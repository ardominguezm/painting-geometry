from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from tqdm import tqdm

from src.baselines import strong_baseline_features
from src.preprocessing import preprocess

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def discover_images(root: Path, max_per_artist: int | None = None):
    if not root.exists():
        raise FileNotFoundError(f'Corpus root does not exist: {root}')
    for artist_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        artist = artist_dir.name
        paths = [
            p for p in sorted(artist_dir.rglob('*'))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if max_per_artist is not None:
            paths = paths[:max_per_artist]
        for path in paths:
            yield artist, path


def extract_one(path: Path, artist: str, long_side: int):
    _, I = preprocess(path, long_side=long_side)
    row = {
        'artist': artist,
        'filename': path.name,
        'path': str(path),
        'resolution_long_side': long_side,
    }
    features = strong_baseline_features(I)
    row.update({f'strong__{k}': v for k, v in features.items()})
    return row


def run(root: Path, output: Path, long_side: int, max_per_artist: int | None):
    items = list(discover_images(root, max_per_artist=max_per_artist))
    if not items:
        raise RuntimeError(f'No images found under {root}')

    print(f'Corpus root: {root}', flush=True)
    print(f'Images discovered: {len(items)}', flush=True)
    print(f'Resolution (long side): {long_side}', flush=True)

    rows = []
    failures = []
    for artist, path in tqdm(items, desc=f'Strong baseline: {root.name}', dynamic_ncols=True):
        try:
            rows.append(extract_one(path, artist, long_side))
        except Exception as exc:
            failures.append({'artist': artist, 'path': str(path), 'error': repr(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        fail_path = output.with_name(output.stem + '_failures.csv')
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f'Warning: {len(failures)} failures -> {fail_path}', flush=True)

    if not rows:
        raise RuntimeError('Strong baseline extraction failed for every image.')

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f'Saved {len(df)} rows x {df.shape[1]} columns to {output}', flush=True)
    print(f'Strong baseline features: {sum(c.startswith("strong__") for c in df.columns)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract stronger non-geometric appearance baselines.')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-side', type=int, default=512)
    parser.add_argument('--max-per-artist', type=int, default=None)
    args = parser.parse_args()
    run(args.root, args.output, args.long_side, args.max_per_artist)
