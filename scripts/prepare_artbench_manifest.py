from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def norm_token(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def candidate_split_roots(root: Path, split: str) -> list[Path]:
    out = []
    for p in root.rglob(split):
        if not p.is_dir():
            continue
        child_dirs = [d for d in p.iterdir() if d.is_dir()]
        nonempty = 0
        for d in child_dirs:
            if any(q.is_file() and q.suffix.lower() in IMAGE_EXTS for q in d.rglob("*")):
                nonempty += 1
        if nonempty >= 5:
            out.append(p)
    return out


def resolve_split_root(root: Path, split: str) -> Path:
    cands = candidate_split_roots(root, split)
    if not cands:
        raise FileNotFoundError(f"Could not locate an ImageFolder-style '{split}' directory below {root}")
    scored = []
    for p in cands:
        n_styles = 0
        n_images = 0
        for d in p.iterdir():
            if d.is_dir():
                imgs = [q for q in d.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    n_styles += 1
                    n_images += len(imgs)
        scored.append((n_styles, n_images, p))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return scored[0][2]


def discover_split(split_root: Path, split: str) -> pd.DataFrame:
    rows = []
    for style_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
        style = norm_token(style_dir.name)
        for path in sorted(q for q in style_dir.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTS):
            rows.append(
                {
                    "split": split,
                    "style": style,
                    "style_dir": style_dir.name,
                    "filename": path.name,
                    "stem": path.stem,
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows)


def detect_col(df: pd.DataFrame, aliases: list[str], required: bool = False) -> str | None:
    exact = {norm_token(c): c for c in df.columns}
    for alias in aliases:
        key = norm_token(alias)
        if key in exact:
            return exact[key]
    for c in df.columns:
        n = norm_token(c)
        if any(norm_token(alias) in n for alias in aliases):
            return c
    if required:
        raise KeyError(f"Could not detect any of columns {aliases}; metadata columns={list(df.columns)}")
    return None


def load_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = pd.read_csv(path)
    cols = {
        "filename": detect_col(df, ["filename", "file_name", "image_name", "image", "name", "path"], required=True),
        "artist": detect_col(df, ["artist", "artist_name", "creator", "author"], required=True),
        "style": detect_col(df, ["style", "label", "genre", "class"]),
        "split": detect_col(df, ["split", "set", "subset"]),
        "source": detect_col(df, ["source", "database", "origin"]),
    }
    return df, cols


def add_metadata(manifest: pd.DataFrame, meta: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    m = meta.copy()
    file_col = cols["filename"]
    m["_meta_basename"] = m[file_col].astype(str).map(lambda x: Path(x).name)
    m["_meta_stem"] = m["_meta_basename"].map(lambda x: Path(x).stem)
    m["_meta_basename_norm"] = m["_meta_basename"].map(norm_token)
    m["_meta_stem_norm"] = m["_meta_stem"].map(norm_token)
    if cols["style"]:
        m["_meta_style_norm"] = m[cols["style"]].astype(str).map(norm_token)
    else:
        m["_meta_style_norm"] = ""

    by_base = {}
    by_stem = {}
    for idx, row in m.iterrows():
        by_base.setdefault(row["_meta_basename_norm"], []).append(idx)
        by_stem.setdefault(row["_meta_stem_norm"], []).append(idx)

    artists = []
    sources = []
    matched = []
    meta_indices = []
    ambiguous = []

    for rec in manifest.itertuples(index=False):
        key_base = norm_token(rec.filename)
        key_stem = norm_token(rec.stem)
        cand = list(by_base.get(key_base, []))
        if not cand:
            cand = list(by_stem.get(key_stem, []))

        if len(cand) > 1 and cols["style"]:
            style_matches = [i for i in cand if m.at[i, "_meta_style_norm"] == rec.style]
            if style_matches:
                cand = style_matches

        if len(cand) == 1:
            i = cand[0]
            artists.append(str(m.at[i, cols["artist"]]))
            sources.append(str(m.at[i, cols["source"]]) if cols["source"] else "")
            matched.append(True)
            meta_indices.append(i)
            ambiguous.append(False)
        elif len(cand) > 1:
            i = cand[0]
            artists.append(str(m.at[i, cols["artist"]]))
            sources.append(str(m.at[i, cols["source"]]) if cols["source"] else "")
            matched.append(True)
            meta_indices.append(i)
            ambiguous.append(True)
        else:
            artists.append("")
            sources.append("")
            matched.append(False)
            meta_indices.append(np.nan)
            ambiguous.append(False)

    out = manifest.copy()
    out["artist"] = artists
    out["source"] = sources
    out["metadata_match"] = matched
    out["metadata_ambiguous"] = ambiguous
    out["metadata_index"] = meta_indices
    return out


def main(dataset_root: Path, metadata_csv: Path | None, output: Path):
    train_root = resolve_split_root(dataset_root, "train")
    test_root = resolve_split_root(dataset_root, "test")
    train = discover_split(train_root, "train")
    test = discover_split(test_root, "test")
    manifest = pd.concat([train, test], ignore_index=True)

    print("Resolved train:", train_root)
    print("Resolved test :", test_root)
    print("\nImage counts by split/style:")
    print(manifest.groupby(["split", "style"]).size().unstack(fill_value=0).to_string())

    if metadata_csv is not None and metadata_csv.exists():
        meta, cols = load_metadata(metadata_csv)
        print("\nMetadata columns:", cols)
        manifest = add_metadata(manifest, meta, cols)
        rate = float(manifest["metadata_match"].mean())
        print(f"Metadata match rate: {rate:.3%}")
        if "artist" in manifest:
            print("Unique matched artists:", manifest.loc[manifest["metadata_match"], "artist"].nunique())
            overlap = set(manifest.loc[manifest["split"] == "train", "artist"]) & set(
                manifest.loc[manifest["split"] == "test", "artist"]
            )
            overlap.discard("")
            print("Artist overlap across official train/test:", len(overlap))
    else:
        manifest["artist"] = ""
        manifest["source"] = ""
        manifest["metadata_match"] = False
        manifest["metadata_ambiguous"] = False
        manifest["metadata_index"] = np.nan
        print("\nWARNING: metadata not supplied. Artist-disjoint evaluation will be unavailable.")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print("\nManifest ->", output)
    print("Rows:", len(manifest))
    print("Styles:", sorted(manifest["style"].unique()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build an ArtBench-10 image manifest and attach artist metadata.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("results/artbench_manifest.csv"))
    args = p.parse_args()
    main(args.dataset_root, args.metadata_csv, args.output)
