from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

HF_DATASET = "zguo0525/ArtBench"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_LINKAGE_COVERAGE = 0.995
MIN_UNIQUE_RATE = 0.999


def norm_token(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def detect_col(df: pd.DataFrame, aliases: list[str], required: bool = False) -> str | None:
    exact = {norm_token(c): c for c in df.columns}
    for alias in aliases:
        k = norm_token(alias)
        if k in exact:
            return exact[k]
    for c in df.columns:
        n = norm_token(c)
        if any(norm_token(a) in n for a in aliases):
            return c
    if required:
        raise KeyError(f"Could not detect any of {aliases}; columns={list(df.columns)}")
    return None


def locate_metadata(kaggle_root: Path) -> Path:
    preferred = list(kaggle_root.rglob("ArtBench-10.csv"))
    csvs = preferred + [p for p in kaggle_root.rglob("*.csv") if p not in preferred]
    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=5)
            detect_col(df, ["name", "filename", "file_name", "image"], required=True)
            detect_col(df, ["artist", "artist_name", "creator", "author"], required=True)
            detect_col(df, ["label", "style", "class"], required=True)
            detect_col(df, ["split", "set", "subset"], required=True)
            return p
        except Exception:
            continue
    raise FileNotFoundError(f"Could not locate usable ArtBench metadata below {kaggle_root}")


def load_metadata(kaggle_root: Path):
    p = locate_metadata(kaggle_root)
    df = pd.read_csv(p)
    cols = {
        "filename": detect_col(df, ["name", "filename", "file_name", "image"], required=True),
        "artist": detect_col(df, ["artist", "artist_name", "creator", "author"], required=True),
        "style": detect_col(df, ["label", "style", "class"], required=True),
        "split": detect_col(df, ["split", "set", "subset"], required=True),
    }
    print("Metadata CSV:", p)
    print("Metadata columns detected:", cols)
    print("Metadata rows:", len(df))
    if len(df) != 60000:
        raise RuntimeError(f"Expected 60000 metadata rows, found {len(df)}")
    return p, df, cols


def normalize_split(x: str) -> str:
    s = norm_token(x)
    if s in {"train", "training"}:
        return "train"
    if s in {"test", "testing", "val", "validation"}:
        return "test"
    return s


def metadata_lookup(meta: pd.DataFrame, cols: dict[str, str]) -> tuple[dict, dict]:
    m = meta.copy()
    m["_split"] = m[cols["split"]].astype(str).map(normalize_split)
    m["_style"] = m[cols["style"]].astype(str).map(norm_token)
    m["_name"] = m[cols["filename"]].astype(str).map(lambda x: Path(x).name)
    m["_base"] = m["_name"].map(norm_token)
    m["_stem"] = m["_name"].map(lambda x: norm_token(Path(x).stem))

    by_base: dict[tuple[str, str, str], list[int]] = {}
    by_stem: dict[tuple[str, str, str], list[int]] = {}
    for i, r in m.iterrows():
        by_base.setdefault((r["_split"], r["_style"], r["_base"]), []).append(i)
        by_stem.setdefault((r["_split"], r["_style"], r["_stem"]), []).append(i)
    return by_base, by_stem


def raw_path_name(raw_image) -> str:
    if not isinstance(raw_image, dict):
        return ""
    p = raw_image.get("path")
    if p is None:
        return ""
    s = str(p).strip()
    return Path(s).name if s else ""


def recover_names_from_hf_embedded_paths(ds, raw_ds, label_names, meta, cols, audit_rows):
    """Recover metadata-linked filenames from paths embedded in the HF parquet Image field.

    The linkage is accepted only if each split has >=99.5% metadata coverage and essentially unique
    recovered names. A very small unlinked remainder is retained for style-only analyses, but is
    explicitly excluded from artist-dependent analyses because the later manifest assigns it no artist.
    """
    by_base, by_stem = metadata_lookup(meta, cols)
    assignments: dict[str, dict[int, str]] = {}
    all_ok = True

    for split in ["train", "test"]:
        labels = np.asarray(ds[split]["label"], dtype=int)
        assn: dict[int, str] = {}
        n_with_path = 0
        n_exact = 0
        n_stem = 0
        n_amb = 0
        examples = []

        for i in tqdm(range(len(ds[split])), desc=f"Audit HF embedded paths {split}", dynamic_ncols=True):
            raw = raw_ds[split][i]["image"]
            fn = raw_path_name(raw)
            if not fn:
                continue
            n_with_path += 1
            if len(examples) < 8:
                examples.append(fn)
            style = label_names[int(labels[i])]
            hits = by_base.get((split, style, norm_token(fn)), [])
            if len(hits) == 1:
                j = hits[0]
                assn[i] = str(meta.at[j, cols["filename"]])
                n_exact += 1
                continue
            hits = by_stem.get((split, style, norm_token(Path(fn).stem)), [])
            if len(hits) == 1:
                j = hits[0]
                assn[i] = str(meta.at[j, cols["filename"]])
                n_stem += 1
            elif len(hits) > 1:
                n_amb += 1

        n = len(ds[split])
        coverage = len(assn) / n
        unique_names = len(set(Path(v).name for v in assn.values()))
        unique_rate = unique_names / max(len(assn), 1)
        n_unlinked = n - len(assn)
        print(
            f"HF embedded-path audit {split}: paths={n_with_path}/{n}, matched={len(assn)}/{n} "
            f"({coverage:.3%}), exact={n_exact}, stem={n_stem}, ambiguous={n_amb}, "
            f"unlinked={n_unlinked}, unique={unique_rate:.3%}"
        )
        print("  path examples:", examples)
        audit_rows.append({
            "audit": "hf_embedded_path",
            "split": split,
            "n": n,
            "n_with_path": n_with_path,
            "n_matched": len(assn),
            "n_unlinked": n_unlinked,
            "coverage": coverage,
            "unique_rate": unique_rate,
            "n_exact": n_exact,
            "n_stem": n_stem,
            "n_ambiguous": n_amb,
        })
        if not (coverage >= MIN_LINKAGE_COVERAGE and unique_rate >= MIN_UNIQUE_RATE):
            all_ok = False
        assignments[split] = assn

    if not all_ok:
        return None
    total_matched = sum(len(v) for v in assignments.values())
    print(
        "Validated direct HF embedded-filename linkage ✓ "
        f"({total_matched}/60000 metadata-linked; {60000-total_matched} retained style-only)"
    )
    return assignments


def pil_rgb(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.open(x).convert("RGB")


def materialize(ds, raw_ds, label_names, assignments, output_root: Path, audit_csv: Path):
    """Materialize all 60k images.

    Metadata-linked rows use the exact ArtBench filename. The tiny unlinked remainder receives a
    synthetic filename beginning ``__hf_unlinked__`` so prepare_artbench_manifest.py cannot attach an
    uncertain artist accidentally. Those images remain valid for style/corpus analyses.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    unlinked_rows = []

    for split in ["train", "test"]:
        labels = np.asarray(ds[split]["label"], dtype=int)
        n_linked = len(assignments[split])
        print(
            f"Materialization policy {split}: {n_linked}/{len(ds[split])} metadata-linked; "
            f"{len(ds[split])-n_linked} unlinked rows retained for style-only analyses."
        )

        for i in tqdm(range(len(ds[split])), desc=f"Materialize HF 256 {split}", dynamic_ncols=True):
            style = label_names[int(labels[i])]
            raw = raw_ds[split][i]["image"]
            embedded_fn = raw_path_name(raw)
            linked = i in assignments[split]

            if linked:
                official_fn = Path(assignments[split][i]).name
                target_name = official_fn
            else:
                official_fn = ""
                suffix = Path(embedded_fn).suffix.lower()
                if suffix not in IMAGE_EXTS:
                    suffix = ".jpg"
                target_name = f"__hf_unlinked__{split}_{i:05d}{suffix}"
                unlinked_rows.append({
                    "split": split,
                    "style": style,
                    "hf_index": i,
                    "embedded_filename": embedded_fn,
                    "synthetic_filename": target_name,
                })

            target = output_root / split / style / target_name
            target.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(raw, dict) and raw.get("bytes"):
                if not target.exists() or target.stat().st_size <= 100:
                    target.write_bytes(raw["bytes"])
                physical = target
            elif isinstance(raw, dict) and raw.get("path") and Path(str(raw["path"])).exists():
                if not target.exists() or target.stat().st_size <= 100:
                    shutil.copy2(Path(str(raw["path"])), target)
                physical = target
            else:
                physical = target.with_suffix(".png")
                if not physical.exists() or physical.stat().st_size <= 100:
                    pil_rgb(ds[split][i]["image"]).save(physical, format="PNG")

            rows.append({
                "split": split,
                "style": style,
                "metadata_linked": linked,
                "official_filename": official_fn,
                "embedded_filename": embedded_fn,
                "physical_filename": physical.name,
                "path": str(physical),
            })

    mat = pd.DataFrame(rows)
    if len(mat) != 60000:
        raise RuntimeError(f"Materialized {len(mat)} rows, expected 60000")
    if int(mat["metadata_linked"].sum()) < int(MIN_LINKAGE_COVERAGE * len(mat)):
        raise RuntimeError("Materialized metadata linkage fell below the pre-specified 99.5% guardrail")

    mat_path = audit_csv.with_name("hf256_materialization_manifest.csv")
    mat.to_csv(mat_path, index=False)
    unlinked_path = audit_csv.with_name("hf256_unlinked_rows.csv")
    pd.DataFrame(unlinked_rows).to_csv(unlinked_path, index=False)

    print("HF 256 ImageFolder ready ✓", output_root)
    print("Rows:", len(mat))
    print("Metadata-linked rows:", int(mat["metadata_linked"].sum()))
    print("Style-only unlinked rows:", int((~mat["metadata_linked"]).sum()))
    print("Materialization manifest ->", mat_path)
    print("Unlinked-row audit ->", unlinked_path)


def main(kaggle_root: Path, output_root: Path, audit_csv: Path):
    try:
        from datasets import Image as HFImage, load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets first: pip install 'datasets>=3.0'") from exc

    _, meta, cols = load_metadata(kaggle_root)

    print("Loading 256px mirror from Hugging Face:", HF_DATASET)
    ds = load_dataset(HF_DATASET)
    if len(ds["train"]) != 50000 or len(ds["test"]) != 10000:
        raise RuntimeError(
            f"Unexpected HF sizes train={len(ds['train'])}, test={len(ds['test'])}; expected 50000/10000"
        )

    label_names = [norm_token(x) for x in ds["train"].features["label"].names]
    print("HF label names:", label_names)
    if len(label_names) != 10:
        raise RuntimeError(f"Expected 10 HF styles, got {label_names}")

    raw_ds = {
        split: ds[split].cast_column("image", HFImage(decode=False))
        for split in ["train", "test"]
    }

    audit_rows: list[dict] = []
    assignments = recover_names_from_hf_embedded_paths(
        ds, raw_ds, label_names, meta, cols, audit_rows
    )

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    print("Filename-recovery audit ->", audit_csv)

    if assignments is None:
        raise RuntimeError(
            "The Hugging Face 256px parquet does not preserve enough original ArtBench filenames to "
            "link images to artist metadata with the pre-specified guardrails. Stopping rather than "
            "fabricate artist identities."
        )

    materialize(ds, raw_ds, label_names, assignments, output_root, audit_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--audit-csv", type=Path, required=True)
    a = p.parse_args()
    main(a.kaggle_root, a.output_root, a.audit_csv)
