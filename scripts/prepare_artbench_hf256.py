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
    """Use filenames embedded in the HF Image parquet field if they were preserved.

    Every path is matched against ArtBench-10.csv within the same split and style. We accept this
    linkage only with >=99.5% coverage in both splits and essentially unique recovered names.
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
        print(
            f"HF embedded-path audit {split}: paths={n_with_path}/{n}, matched={len(assn)}/{n} "
            f"({coverage:.3%}), exact={n_exact}, stem={n_stem}, ambiguous={n_amb}, unique={unique_rate:.3%}"
        )
        print("  path examples:", examples)
        audit_rows.append({
            "audit": "hf_embedded_path",
            "split": split,
            "n": n,
            "n_with_path": n_with_path,
            "n_matched": len(assn),
            "coverage": coverage,
            "unique_rate": unique_rate,
            "n_exact": n_exact,
            "n_stem": n_stem,
            "n_ambiguous": n_amb,
        })
        if not (coverage >= 0.995 and unique_rate >= 0.999):
            all_ok = False
        assignments[split] = assn

    if not all_ok:
        return None
    print("Validated direct HF embedded-filename linkage ✓")
    return assignments


def pil_rgb(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.open(x).convert("RGB")


def materialize(ds, raw_ds, label_names, assignments, output_root: Path, audit_csv: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for split in ["train", "test"]:
        labels = np.asarray(ds[split]["label"], dtype=int)
        if len(assignments[split]) != len(ds[split]):
            raise RuntimeError(f"Incomplete assignment for {split}: {len(assignments[split])}/{len(ds[split])}")

        for i in tqdm(range(len(ds[split])), desc=f"Materialize HF 256 {split}", dynamic_ncols=True):
            style = label_names[int(labels[i])]
            official_fn = Path(assignments[split][i]).name
            raw = raw_ds[split][i]["image"]
            target = output_root / split / style / official_fn
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
                "official_filename": official_fn,
                "physical_filename": physical.name,
                "path": str(physical),
            })

    mat = pd.DataFrame(rows)
    if len(mat) != 60000:
        raise RuntimeError(f"Materialized {len(mat)} rows, expected 60000")
    mat.to_csv(audit_csv.with_name("hf256_materialization_manifest.csv"), index=False)
    print("HF 256 ImageFolder ready ✓", output_root)
    print("Rows:", len(mat))


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
            "link the 60,000 images to artist metadata exactly. The independent CIFAR-row-order test "
            "already failed at chance level (~10% style agreement). We therefore stop rather than "
            "fabricate artist identities. The scientifically defensible fallback is to use all 60,000 "
            "for style/corpus-level analyses and retain the existing metadata-linked 4,000-image sample "
            "for artist-disjoint generalisation."
        )

    materialize(ds, raw_ds, label_names, assignments, output_root, audit_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--audit-csv", type=Path, required=True)
    a = p.parse_args()
    main(a.kaggle_root, a.output_root, a.audit_csv)
