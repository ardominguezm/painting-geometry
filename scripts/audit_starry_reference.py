from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def sha1_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def image_hashes(path: Path) -> tuple[str, str, str, int, int]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        width, height = img.size
        ph = str(imagehash.phash(img, hash_size=8))
        dh = str(imagehash.dhash(img, hash_size=8))
    return sha1_file(path), ph, dh, width, height


def hamming_hex(a: str, b: str) -> int:
    return int(int(a, 16) ^ int(b, 16)).bit_count()


def discover_artist(root: Path, artist: str, split: str) -> list[tuple[str, str, Path]]:
    artist_dir = root / artist
    if not artist_dir.exists():
        matches = [p for p in root.iterdir() if p.is_dir() and p.name.lower() == artist.lower()]
        if not matches:
            raise FileNotFoundError(f"Artist folder {artist!r} not found below {root}")
        artist_dir = matches[0]
    return [
        (split, artist_dir.name, p)
        for p in sorted(artist_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def make_contact_sheet(starry_path: Path, candidates: pd.DataFrame, output: Path, max_pairs: int = 20) -> None:
    if candidates.empty:
        return
    rows = candidates.head(max_pairs)
    thumb_w, thumb_h = 240, 190
    label_h = 60
    canvas = Image.new("RGB", (2 * thumb_w, len(rows) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    with Image.open(starry_path) as img:
        star = ImageOps.exif_transpose(img).convert("RGB")

    for r, row in enumerate(rows.itertuples(index=False)):
        y0 = r * (thumb_h + label_h)
        for c, (img_obj, label) in enumerate([
            (star.copy(), f"Uploaded Starry Night / {starry_path.name}"),
            (ImageOps.exif_transpose(Image.open(row.path)).convert("RGB"), f"{row.split}: {row.artist}/{row.filename}"),
        ]):
            try:
                img_obj.thumbnail((thumb_w - 8, thumb_h - 8))
                x = c * thumb_w + (thumb_w - img_obj.width) // 2
                y = y0 + (thumb_h - img_obj.height) // 2
                canvas.paste(img_obj, (x, y))
                draw.text((c * thumb_w + 4, y0 + thumb_h + 2), label[:54], fill="black")
            finally:
                try:
                    img_obj.close()
                except Exception:
                    pass
        draw.text(
            (4, y0 + thumb_h + 28),
            f"pHash={row.phash_distance}, dHash={row.dhash_distance}, exact={row.exact_bytes}, strict={row.strict_exclusion}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main(
    image_path: Path,
    train_root: Path,
    test_root: Path,
    output_dir: Path,
    artist: str,
    screen_phash: int,
    screen_dhash: int,
    exclude_phash: int,
    exclude_dhash: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    star_sha1, star_ph, star_dh, star_w, star_h = image_hashes(image_path)

    items = discover_artist(train_root, artist, "train") + discover_artist(test_root, artist, "test")
    rows = []
    for split, artist_name, path in tqdm(items, desc="Comparing Starry Night to Van Gogh reference", dynamic_ncols=True):
        try:
            sha1, ph, dh, width, height = image_hashes(path)
            pdist = hamming_hex(star_ph, ph)
            ddist = hamming_hex(star_dh, dh)
            exact = bool(sha1 == star_sha1)
            screen = exact or (pdist <= screen_phash) or (ddist <= screen_dhash)
            strict = exact or ((pdist <= exclude_phash) and (ddist <= exclude_dhash))
            rows.append({
                "split": split,
                "artist": artist_name,
                "filename": path.name,
                "path": str(path),
                "width": width,
                "height": height,
                "sha1": sha1,
                "phash": ph,
                "dhash": dh,
                "phash_distance": int(pdist),
                "dhash_distance": int(ddist),
                "exact_bytes": exact,
                "screen_candidate": bool(screen),
                "strict_exclusion": bool(strict),
            })
        except Exception as exc:
            rows.append({
                "split": split,
                "artist": artist_name,
                "filename": path.name,
                "path": str(path),
                "error": repr(exc),
            })

    all_df = pd.DataFrame(rows)
    all_df.to_csv(output_dir / "starry_reference_all_hash_distances.csv", index=False)

    valid = all_df.dropna(subset=["phash_distance", "dhash_distance"]).copy()
    candidates = valid[valid["screen_candidate"].astype(bool)].copy().sort_values(
        ["strict_exclusion", "exact_bytes", "phash_distance", "dhash_distance"],
        ascending=[False, False, True, True],
    )
    candidates.to_csv(output_dir / "starry_reference_candidates.csv", index=False)

    exclusions = valid[valid["strict_exclusion"].astype(bool)].copy().sort_values(
        ["exact_bytes", "phash_distance", "dhash_distance"],
        ascending=[False, True, True],
    )
    exclusions.to_csv(output_dir / "starry_reference_exclusions.csv", index=False)

    make_contact_sheet(image_path, candidates, output_dir / "starry_reference_contact_sheet.jpg")

    summary = pd.DataFrame([{
        "starry_filename": image_path.name,
        "starry_width": star_w,
        "starry_height": star_h,
        "reference_artist": artist,
        "reference_images_scanned": len(items),
        "screen_candidates": len(candidates),
        "strict_exclusions": len(exclusions),
        "exact_byte_matches": int(exclusions.get("exact_bytes", pd.Series(dtype=bool)).sum()) if not exclusions.empty else 0,
        "screen_rule": f"exact OR pHash<={screen_phash} OR dHash<={screen_dhash}",
        "strict_exclusion_rule": f"exact OR (pHash<={exclude_phash} AND dHash<={exclude_dhash})",
    }])
    summary.to_csv(output_dir / "starry_reference_audit_summary.csv", index=False)

    print("\nStarry-reference audit summary:")
    print(summary.to_string(index=False))
    if not candidates.empty:
        print("\nClosest screened candidates:")
        print(candidates[["split", "artist", "filename", "phash_distance", "dhash_distance", "exact_bytes", "strict_exclusion"]].head(20).to_string(index=False))
    else:
        print("\nNo candidates met the permissive screen.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit whether the uploaded Starry Night appears in the Van Gogh reference corpus.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3b_starry_audit"))
    p.add_argument("--artist", type=str, default="VanGogh")
    p.add_argument("--screen-phash", type=int, default=10)
    p.add_argument("--screen-dhash", type=int, default=10)
    p.add_argument("--exclude-phash", type=int, default=4)
    p.add_argument("--exclude-dhash", type=int, default=4)
    args = p.parse_args()
    main(
        args.image,
        args.train_root,
        args.test_root,
        args.output_dir,
        args.artist,
        args.screen_phash,
        args.screen_dhash,
        args.exclude_phash,
        args.exclude_dhash,
    )
