from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baselines import edge_features, glcm_features
from src.curvature import multiscale_curvature_features
from src.orientation import structure_tensor_features
from src.preprocessing import preprocess

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def discover_images(root: Path):
    for artist_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        artist = artist_dir.name
        for path in sorted(artist_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
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


def run(root: Path, output: Path, long_side: int, sigmas: tuple[float, ...]):
    items = list(discover_images(root))
    if not items:
        raise RuntimeError(f'No images found under {root}')

    rows = []
    failures = []
    for artist, path in tqdm(items, desc=f'Extracting {root.name}'):
        try:
            rows.append(extract_one(path, artist, long_side, sigmas))
        except Exception as exc:
            failures.append({'artist': artist, 'path': str(path), 'error': repr(exc)})

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    if failures:
        fail_path = output.with_name(output.stem + '_failures.csv')
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f'Warning: {len(failures)} failures written to {fail_path}')

    print(f'Saved {len(df)} rows x {df.shape[1]} columns to {output}')
    print('Artists:', sorted(df['artist'].unique().tolist()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract multiscale geometry and baseline features from a painting corpus.')
    parser.add_argument('--root', type=Path, required=True, help='Folder containing one subfolder per artist.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-side', type=int, default=512)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[1.0, 2.0, 4.0, 8.0])
    args = parser.parse_args()

    run(args.root, args.output, args.long_side, tuple(args.sigmas))
