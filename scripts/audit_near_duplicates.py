from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def discover_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f'Corpus root does not exist: {root}')
    rows = []
    for artist_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        artist = artist_dir.name
        for path in sorted(artist_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append((artist, path))
    return rows


def sha1_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_record(split: str, artist: str, path: Path) -> dict:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert('RGB')
        width, height = img.size
        ph = imagehash.phash(img, hash_size=8)
        dh = imagehash.dhash(img, hash_size=8)
    return {
        'split': split,
        'artist': artist,
        'filename': path.name,
        'path': str(path),
        'width': width,
        'height': height,
        'sha1': sha1_file(path),
        'phash': str(ph),
        'dhash': str(dh),
        'phash_u64': np.uint64(int(str(ph), 16)),
        'dhash_u64': np.uint64(int(str(dh), 16)),
    }


def compute_hash_table(split: str, root: Path) -> pd.DataFrame:
    items = discover_images(root)
    records = []
    for artist, path in tqdm(items, desc=f'Hashing {split}', dynamic_ncols=True):
        try:
            records.append(hash_record(split, artist, path))
        except Exception as exc:
            records.append({
                'split': split,
                'artist': artist,
                'filename': path.name,
                'path': str(path),
                'width': np.nan,
                'height': np.nan,
                'sha1': None,
                'phash': None,
                'dhash': None,
                'phash_u64': np.nan,
                'dhash_u64': np.nan,
                'error': repr(exc),
            })
    return pd.DataFrame(records)


_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a[:, None], b[None, :])
    byte_view = xor.view(np.uint8).reshape(xor.shape + (8,))
    return _POPCOUNT_LUT[byte_view].sum(axis=-1).astype(np.uint8)


def find_cross_split_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    phash_threshold: int = 6,
    dhash_threshold: int = 6,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Return a permissive candidate list for visual/manual audit.

    A pair is retained when either pHash OR dHash is within its screening
    threshold. Automatic exclusion is deliberately more conservative and is
    performed later using both distances simultaneously.
    """
    tr = train_df.dropna(subset=['phash_u64', 'dhash_u64']).reset_index(drop=True)
    te = test_df.dropna(subset=['phash_u64', 'dhash_u64']).reset_index(drop=True)
    train_ph = tr['phash_u64'].astype('uint64').to_numpy()
    train_dh = tr['dhash_u64'].astype('uint64').to_numpy()

    rows = []
    for start in tqdm(range(0, len(te), chunk_size), desc='Cross-split similarity', dynamic_ncols=True):
        block = te.iloc[start:start + chunk_size]
        test_ph = block['phash_u64'].astype('uint64').to_numpy()
        test_dh = block['dhash_u64'].astype('uint64').to_numpy()
        pdist = hamming_matrix(test_ph, train_ph)
        ddist = hamming_matrix(test_dh, train_dh)
        ii, jj = np.where((pdist <= phash_threshold) | (ddist <= dhash_threshold))
        for i_local, j in zip(ii, jj):
            test_row = block.iloc[int(i_local)]
            train_row = tr.iloc[int(j)]
            rows.append({
                'test_artist': test_row.artist,
                'test_filename': test_row.filename,
                'test_path': test_row.path,
                'train_artist': train_row.artist,
                'train_filename': train_row.filename,
                'train_path': train_row.path,
                'phash_distance': int(pdist[i_local, j]),
                'dhash_distance': int(ddist[i_local, j]),
                'exact_bytes': bool(test_row.sha1 == train_row.sha1),
                'same_artist': bool(test_row.artist == train_row.artist),
            })
    if not rows:
        return pd.DataFrame(columns=[
            'test_artist', 'test_filename', 'test_path', 'train_artist', 'train_filename',
            'train_path', 'phash_distance', 'dhash_distance', 'exact_bytes', 'same_artist'
        ])
    return pd.DataFrame(rows).sort_values(
        ['exact_bytes', 'phash_distance', 'dhash_distance'],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def make_contact_sheet(candidates: pd.DataFrame, output: Path, max_pairs: int = 24):
    if candidates.empty:
        return
    pairs = candidates.head(max_pairs)
    thumb_w, thumb_h = 220, 180
    label_h = 58
    canvas = Image.new('RGB', (2 * thumb_w, len(pairs) * (thumb_h + label_h)), 'white')
    draw = ImageDraw.Draw(canvas)

    for r, row in enumerate(pairs.itertuples(index=False)):
        y0 = r * (thumb_h + label_h)
        for c, (path, artist, filename) in enumerate([
            (row.train_path, row.train_artist, row.train_filename),
            (row.test_path, row.test_artist, row.test_filename),
        ]):
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert('RGB')
                    img.thumbnail((thumb_w - 8, thumb_h - 8))
                    x = c * thumb_w + (thumb_w - img.width) // 2
                    y = y0 + (thumb_h - img.height) // 2
                    canvas.paste(img, (x, y))
            except Exception:
                pass
            draw.text((c * thumb_w + 4, y0 + thumb_h + 2), f'{artist} / {filename}', fill='black')
        draw.text(
            (4, y0 + thumb_h + 24),
            f'pHash={row.phash_distance}, dHash={row.dhash_distance}, exact={row.exact_bytes}',
            fill='black',
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main(train_root: Path, test_root: Path, output_dir: Path, phash_threshold: int, dhash_threshold: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_hashes = compute_hash_table('train', train_root)
    test_hashes = compute_hash_table('test', test_root)
    train_hashes.to_csv(output_dir / 'train_hashes.csv', index=False)
    test_hashes.to_csv(output_dir / 'test_hashes.csv', index=False)

    candidates = find_cross_split_candidates(
        train_hashes,
        test_hashes,
        phash_threshold=phash_threshold,
        dhash_threshold=dhash_threshold,
    )
    candidates.to_csv(output_dir / 'cross_split_near_duplicates.csv', index=False)
    make_contact_sheet(candidates, output_dir / 'near_duplicate_contact_sheet.jpg')

    exact = int(candidates['exact_bytes'].sum()) if not candidates.empty else 0
    n_test_flagged = int(candidates[['test_artist', 'test_filename']].drop_duplicates().shape[0]) if not candidates.empty else 0
    summary = pd.DataFrame([{
        'train_images': len(train_hashes),
        'test_images': len(test_hashes),
        'candidate_pairs_screened': len(candidates),
        'test_images_with_any_candidate': n_test_flagged,
        'exact_byte_pairs': exact,
        'screen_phash_threshold': phash_threshold,
        'screen_dhash_threshold': dhash_threshold,
        'screen_rule': 'phash<=threshold OR dhash<=threshold',
    }])
    summary.to_csv(output_dir / 'leakage_audit_summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f'Candidates -> {output_dir / "cross_split_near_duplicates.csv"}')
    if not candidates.empty:
        print(candidates.head(20).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit exact and perceptual near-duplicate leakage across corpus splits.')
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--test-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/phase2_leakage'))
    parser.add_argument('--phash-threshold', type=int, default=6)
    parser.add_argument('--dhash-threshold', type=int, default=6)
    args = parser.parse_args()
    main(args.train_root, args.test_root, args.output_dir, args.phash_threshold, args.dhash_threshold)
