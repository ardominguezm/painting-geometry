from __future__ import annotations

import argparse
import pickle
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


def find_batch_root(kaggle_root: Path) -> Path:
    hits = list(kaggle_root.rglob("data_batch_1"))
    for p in hits:
        root = p.parent
        if all((root / f"data_batch_{i}").exists() for i in range(1, 6)) and (root / "test_batch").exists():
            return root
    raise FileNotFoundError(f"Could not find ArtBench python batches below {kaggle_root}")


def unpickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def bget(d, *names):
    for name in names:
        if name in d:
            return d[name]
        b = name.encode() if isinstance(name, str) else name
        if b in d:
            return d[b]
    return None


def decode_filename(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def load_python_index(batch_root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta_path = batch_root / "meta"
    if not meta_path.exists():
        candidates = [p for p in batch_root.iterdir() if p.name in {"meta", "batches.meta"}]
        if not candidates:
            raise FileNotFoundError(f"No ArtBench meta file in {batch_root}")
        meta_path = candidates[0]
    meta = unpickle(meta_path)
    styles_raw = bget(meta, "styles", "label_names")
    if styles_raw is None:
        raise KeyError(f"Could not find styles in {meta_path}")
    styles = [norm_token(decode_filename(x)) for x in styles_raw]
    print("Python-batch styles:", styles)

    rows = []
    rgb32_by_key: dict[str, np.ndarray] = {}
    specs = [("train", batch_root / f"data_batch_{i}") for i in range(1, 6)] + [("test", batch_root / "test_batch")]
    global_order = {"train": 0, "test": 0}

    for split, p in specs:
        d = unpickle(p)
        data = np.asarray(bget(d, "data"))
        labels = np.asarray(bget(d, "labels"), dtype=int)
        filenames_raw = bget(d, "filenames")
        if filenames_raw is None:
            raise KeyError(
                f"{p} does not contain filenames. Filename-aware matching to artist metadata cannot be validated."
            )
        filenames = [decode_filename(x) for x in filenames_raw]
        if len(data) != len(labels) or len(data) != len(filenames):
            raise RuntimeError(f"Inconsistent batch lengths in {p}")
        imgs = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
        for i in range(len(data)):
            style = styles[int(labels[i])]
            fn = Path(filenames[i]).name
            key = f"{split}||{style}||{fn}"
            rows.append(
                {
                    "split": split,
                    "style": style,
                    "label_python": int(labels[i]),
                    "filename": fn,
                    "python_order": global_order[split],
                    "key": key,
                }
            )
            rgb32_by_key[key] = imgs[i]
            global_order[split] += 1

    idx = pd.DataFrame(rows)
    if len(idx) != 60000:
        raise RuntimeError(f"Expected 60000 python-batch rows, found {len(idx)}")
    if idx["key"].duplicated().any():
        dup = idx[idx["key"].duplicated(keep=False)].head(20)
        raise RuntimeError(f"Duplicate split/style/filename keys in python batches:\n{dup}")
    print("Python index ready:", idx.shape)
    return idx, rgb32_by_key


def pil_rgb(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.open(x).convert("RGB")


def mae32(pil: Image.Image, ref: np.ndarray, resample) -> float:
    a = np.asarray(pil.convert("RGB").resize((32, 32), resample=resample), dtype=np.float32)
    b = np.asarray(ref, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def candidate_assignment(
    hf_split,
    split_name: str,
    label_names: list[str],
    py_idx: pd.DataFrame,
    rgb32_by_key: dict[str, np.ndarray],
    audit_rows: list[dict],
) -> dict[int, str]:
    """
    Recover official filenames for the HF 256px rows.

    We test two deterministic orderings of the official 32px batches inside each split/style:
      A) original CIFAR/Python-batch order restricted to the style;
      B) lexicographic filename order, matching standard ImageFolder traversal.

    The chosen ordering must be strongly supported by image-content agreement after resizing HF 256px
    images to 32px. If not, the script stops instead of attaching artists to uncertain images.
    """
    labels = np.asarray(hf_split["label"], dtype=int)
    mapping: dict[int, str] = {}
    filters = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "box": Image.Resampling.BOX,
    }

    for label_id, raw_name in enumerate(label_names):
        style = norm_token(raw_name)
        hf_indices = np.flatnonzero(labels == label_id)
        py = py_idx[(py_idx["split"] == split_name) & (py_idx["style"] == style)].copy()
        if len(hf_indices) != len(py):
            raise RuntimeError(
                f"Count mismatch for {split_name}/{style}: HF={len(hf_indices)} Python={len(py)}"
            )
        candidates = {
            "python_order": py.sort_values("python_order").reset_index(drop=True),
            "filename_order": py.sort_values("filename", kind="stable").reset_index(drop=True),
        }
        n = len(hf_indices)
        sample_pos = np.unique(np.linspace(0, n - 1, min(32, n), dtype=int))
        scores = []
        for order_name, cdf in candidates.items():
            for filt_name, filt in filters.items():
                vals = []
                shifted = []
                for pos in sample_pos:
                    hidx = int(hf_indices[pos])
                    pil = pil_rgb(hf_split[hidx]["image"])
                    row = cdf.iloc[int(pos)]
                    ref = rgb32_by_key[row["key"]]
                    vals.append(mae32(pil, ref, filt))
                    row_shift = cdf.iloc[int((pos + 1) % n)]
                    ref_shift = rgb32_by_key[row_shift["key"]]
                    shifted.append(mae32(pil, ref_shift, filt))
                med = float(np.median(vals))
                p95 = float(np.quantile(vals, 0.95))
                mismatch_med = float(np.median(shifted))
                ratio = med / max(mismatch_med, 1e-9)
                scores.append((med, p95, ratio, order_name, filt_name))
        scores.sort(key=lambda x: (x[2], x[0]))
        med, p95, ratio, order_name, filt_name = scores[0]
        audit_rows.append(
            {
                "split": split_name,
                "style": style,
                "n": n,
                "chosen_order": order_name,
                "best_resize_filter": filt_name,
                "median_mae_0_255": med,
                "p95_mae_0_255": p95,
                "median_to_shifted_ratio": ratio,
            }
        )
        print(
            f"Mapping audit {split_name}/{style}: order={order_name}, filter={filt_name}, "
            f"median MAE={med:.2f}, p95={p95:.2f}, ratio-vs-shift={ratio:.3f}"
        )
        # Strong guardrail: the aligned images must be much closer than a deliberately wrong pairing.
        if not (ratio < 0.55 and med < 40.0):
            raise RuntimeError(
                f"Could not validate filename recovery for {split_name}/{style}. "
                f"Best median MAE={med:.2f}, shifted ratio={ratio:.3f}. "
                "Stopping rather than risking incorrect artist assignments."
            )
        chosen = candidates[order_name]
        for pos, hidx in enumerate(hf_indices):
            mapping[int(hidx)] = str(chosen.iloc[int(pos)]["filename"])
    return mapping


def image_bytes_or_save(raw_image, out_path: Path) -> None:
    """Write HF image without recompression when encoded bytes are exposed; otherwise save losslessly as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 100:
        return

    if isinstance(raw_image, dict):
        b = raw_image.get("bytes")
        path = raw_image.get("path")
        if b:
            out_path.write_bytes(b)
            return
        if path and Path(path).exists():
            shutil.copy2(path, out_path)
            return

    pil = pil_rgb(raw_image)
    # Saving JPEG again would alter local fine-scale structure. Use PNG when bytes are unavailable.
    png_path = out_path.with_suffix(".png")
    pil.save(png_path, format="PNG")
    if png_path != out_path:
        # The manifest builder uses the physical filename; caller will record this alternate name.
        return


def main(kaggle_root: Path, output_root: Path, audit_csv: Path):
    try:
        from datasets import Image as HFImage, load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets first: pip install 'datasets>=3.0'") from exc

    batch_root = find_batch_root(kaggle_root)
    print("ArtBench python batch root:", batch_root)
    py_idx, rgb32_by_key = load_python_index(batch_root)

    print("Loading 256px mirror from Hugging Face:", HF_DATASET)
    ds = load_dataset(HF_DATASET)
    if len(ds["train"]) != 50000 or len(ds["test"]) != 10000:
        raise RuntimeError(
            f"Unexpected HF sizes train={len(ds['train'])}, test={len(ds['test'])}; expected 50000/10000"
        )
    label_names = [norm_token(x) for x in ds["train"].features["label"].names]
    print("HF label names:", label_names)
    if set(label_names) != set(py_idx["style"].unique()):
        raise RuntimeError(
            f"Style-name mismatch HF={label_names} vs python={sorted(py_idx['style'].unique())}"
        )

    audit_rows: list[dict] = []
    assignments = {}
    for split in ["train", "test"]:
        assignments[split] = candidate_assignment(
            ds[split], split, label_names, py_idx, rgb32_by_key, audit_rows
        )
    audit = pd.DataFrame(audit_rows)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_csv, index=False)
    print("Filename-recovery audit ->", audit_csv)

    # Try no-decode access so encoded parquet image bytes can be written without recompression.
    raw_ds = {}
    for split in ["train", "test"]:
        raw_ds[split] = ds[split].cast_column("image", HFImage(decode=False))

    if output_root.exists():
        # Reuse previously materialized valid images; do not delete partial progress.
        print("Reusing materialization root if files already exist:", output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    materialized_rows = []
    for split in ["train", "test"]:
        labels = np.asarray(ds[split]["label"], dtype=int)
        for i in tqdm(range(len(ds[split])), desc=f"Materialize HF 256 {split}", dynamic_ncols=True):
            style = label_names[int(labels[i])]
            official_fn = assignments[split][i]
            raw = raw_ds[split][i]["image"]
            suffix = Path(official_fn).suffix.lower()
            if suffix not in IMAGE_EXTS:
                suffix = ".jpg"
            target = output_root / split / style / Path(official_fn).name
            if isinstance(raw, dict) and raw.get("bytes"):
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size <= 100:
                    target.write_bytes(raw["bytes"])
                physical_fn = target.name
                physical_path = target
            elif isinstance(raw, dict) and raw.get("path") and Path(raw["path"]).exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size <= 100:
                    shutil.copy2(raw["path"], target)
                physical_fn = target.name
                physical_path = target
            else:
                # Rare fallback: lossless PNG, while keeping the official stem for metadata matching.
                pil = pil_rgb(ds[split][i]["image"])
                physical_path = target.with_suffix(".png")
                physical_path.parent.mkdir(parents=True, exist_ok=True)
                if not physical_path.exists() or physical_path.stat().st_size <= 100:
                    pil.save(physical_path, format="PNG")
                physical_fn = physical_path.name
            materialized_rows.append(
                {
                    "split": split,
                    "style": style,
                    "official_filename": official_fn,
                    "physical_filename": physical_fn,
                    "path": str(physical_path),
                }
            )

    mat = pd.DataFrame(materialized_rows)
    if len(mat) != 60000:
        raise RuntimeError(f"Materialized {len(mat)} rows, expected 60000")
    mat.to_csv(audit_csv.with_name("hf256_materialization_manifest.csv"), index=False)
    print("HF 256 ImageFolder ready ✓", output_root)
    print("Rows:", len(mat))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--audit-csv", type=Path, required=True)
    a = p.parse_args()
    main(a.kaggle_root, a.output_root, a.audit_csv)
