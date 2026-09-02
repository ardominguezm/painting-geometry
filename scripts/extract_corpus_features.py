from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When this file is executed as
#     python scripts/extract_corpus_features.py
# Python places ``scripts/`` rather than the repository root on sys.path.
# Add the repository root explicitly so imports from ``src`` work in Colab
# and in ordinary command-line execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from tqdm import tqdm

from src.baselines import edge_features, glcm_features
from src.curvature import multiscale_curvature_features
from src.orientation import structure_tensor_features
from src.preprocessing import preprocess

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def discover_images(root: Path, max_per_artist: int | None = None):
    if not root.exists():
        raise FileNotFoundError(f'Corpus root does not exist: {root}')

    for artist_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        artist = artist_dir.name
        paths = [
            path for path in sorted(artist_dir.rglob('*'))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        if max_per_artist is not None:
            paths = paths[:max_per_artist]
        for path in paths:
            yield artist, path


def extract_one(path: Path, artist: str, long_side: int, sigmas: tuple[float, ...]):
    _, I = preprocess(path, long_side=long_side)

    row = {
        'artist': artist,
        'filename': path.name,
        'path': str(path),
        'resolution_long_side': long_side,
    }

    edge = edge_features(I)
    row.update({f'edge__{k}': v for k, v in edge.items()})

    texture = glcm_features(I)
    row.update({f'texture__{k}': v for k, v in texture.items()})

    orient = structure_tensor_features(I, sigma=2.0)
    row.update({f'orient__{k}': v for k, v in orient.items()})

    curvature, _ = multiscale_curvature_features(I, sigmas=sigmas)
    row.update({f'curv__{k}': v for k, v in curvature.items()})

    return row


def run(
    root: Path,
    output: Path,
    long_side: int,
    sigmas: tuple[float, ...],
    max_per_artist: int | None = None,
):
    items = list(discover_images(root, max_per_artist=max_per_artist))
    if not items:
        raise RuntimeError(f'No images found under {root}')

    print(f'Corpus root: {root}', flush=True)
    print(f'Images discovered: {len(items)}', flush=True)
    print(f'Resolution (long side): {long_side}', flush=True)
    print(f'Curvature scales: {sigmas}', flush=True)
    if max_per_artist is not None:
        print(f'Pilot cap per artist: {max_per_artist}', flush=True)

    rows = []
    failures = []
    progress = tqdm(
        items,
        desc=f'Extracting {root.name}',
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for artist, path in progress:
        try:
            rows.append(extract_one(path, artist, long_side, sigmas))
        except Exception as exc:
            failures.append({'artist': artist, 'path': str(path), 'error': repr(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)

    if failures:
        fail_path = output.with_name(output.stem + '_failures.csv')
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f'Warning: {len(failures)} failures written to {fail_path}', flush=True)
        for failure in failures[:5]:
            print('  sample failure:', failure, flush=True)

    if not rows:
        raise RuntimeError(
            'Feature extraction failed for every image. '
            'Inspect the failures CSV and the sample errors printed above.'
        )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print(f'Saved {len(df)} rows x {df.shape[1]} columns to {output}', flush=True)
    print('Artists:', sorted(df['artist'].unique().tolist()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract multiscale geometry and baseline features from a painting corpus.')
    parser.add_argument('--root', type=Path, required=True, help='Folder containing one subfolder per artist.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-side', type=int, default=512)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument(
        '--max-per-artist',
        type=int,
        default=None,
        help='Optional pilot cap on images per artist. Omit for the full corpus.',
    )
    args = parser.parse_args()

    run(
        args.root,
        args.output,
        args.long_side,
        tuple(args.sigmas),
        max_per_artist=args.max_per_artist,
    )
