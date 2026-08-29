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


def decode_text(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def find_batch_root(kaggle_root: Path) -> Path:
    hits = list(kaggle_root.rglob("data_batch_1"))
    for p in hits:
        root = p.parent
        if all((root / f"data_batch_{i}").exists() for i in range(1, 6)) and (root / "test_batch").exists():
            return root
    raise FileNotFoundError(f"Could not find ArtBench python batches below {kaggle_root}")


def find_metadata_csv(kaggle_root: Path) -> Path:
    preferred = list(kaggle_root.rglob("ArtBench-10.csv"))
    if preferred:
        return preferred[0]
    csvs = list(kaggle_root.rglob("*.csv"))
    for p in csvs:
        try:
            cols = [norm_token(c) for c in pd.read_csv(p, nrows=2).columns]
        except Exception:
            continue
        if any(c in cols for c in ["artist", "artist_name", "creator", "author"]):
            return p
    raise FileNotFoundError(f"Could not find ArtBench metadata CSV below {kaggle_root}")


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


def detect_col(df: pd.DataFrame, aliases: list[str], required: bool = False) -> str | None:
    exact = {norm_token(c): c for c in df.columns}
    for a in aliases:
        k = norm_token(a)
        if k in exact:
            return exact[k]
    for c in df.columns:
        n = norm_token(c)
        if any(norm_token(a) in n for a in aliases):
            return c
    if required:
        raise KeyError(f"Could not detect columns {aliases}; metadata columns={list(df.columns)}")
    return None


def load_python_index(batch_root: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load the official 32px CIFAR-python distribution.

    The public Kaggle mirror omits CIFAR's optional `filenames` key.  We therefore keep
    the exact batch row order and later recover filenames from ArtBench-10.csv only if
    the metadata/style sequence proves that the two are row-aligned.
    """
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
    styles = [norm_token(decode_text(x)) for x in styles_raw]
    print("Python-batch styles:", styles)

    rows = []
    images = []
    specs = [("train", batch_root / f"data_batch_{i}") for i in range(1, 6)] + [("test", batch_root / "test_batch")]
    global_order = {"train": 0, "test": 0}
    any_filenames = False

    for split, p in specs:
        d = unpickle(p)
        data = np.asarray(bget(d, "data"))
        labels = np.asarray(bget(d, "labels"), dtype=int)
        filenames_raw = bget(d, "filenames")
        if len(data) != len(labels):
            raise RuntimeError(f"Inconsistent data/label lengths in {p}")
        if filenames_raw is not None:
            any_filenames = True
            filenames = [Path(decode_text(x)).name for x in filenames_raw]
        else:
            filenames = [""] * len(data)

        imgs = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
        batch_name = p.name
        for i in range(len(data)):
            rows.append(
                {
                    "split": split,
                    "style": styles[int(labels[i])],
                    "label_python": int(labels[i]),
                    "filename": filenames[i],
                    "python_order": global_order[split],
                    "batch_name": batch_name,
                    "batch_row": int(i),
                }
            )
            images.append(imgs[i])
            global_order[split] += 1

    idx = pd.DataFrame(rows)
    rgb32 = np.stack(images, axis=0)
    if len(idx) != 60000:
        raise RuntimeError(f"Expected 60000 python-batch rows, found {len(idx)}")
    print("Python index ready:", idx.shape, "filenames embedded:", any_filenames)
    return idx, rgb32, styles


def metadata_style_series(meta: pd.DataFrame, style_col: str | None) -> pd.Series | None:
    if style_col is None:
        return None
    return meta[style_col].astype(str).map(norm_token)


def split_mask_from_metadata(meta: pd.DataFrame, split_col: str, split: str) -> pd.Series:
    vals = meta[split_col].astype(str).map(norm_token)
    if split == "train":
        accepted = {"train", "training", "tr"}
    else:
        accepted = {"test", "testing", "te", "validation", "val"}
    return vals.isin(accepted)


def recover_filenames_from_metadata(
    py_idx: pd.DataFrame,
    kaggle_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover filename for every official CIFAR row using only a validated row-order relation.

    We never infer artist/filename from style alone.  Candidate CSV orderings are accepted only
    when their style sequence agrees with the official CIFAR labels essentially perfectly and
    the train/test row counts are exactly 50k/10k.  Because the CIFAR label sequence is heavily
    interleaved, this is a strong row-alignment check rather than merely a class-count check.
    """
    metadata_csv = find_metadata_csv(kaggle_root)
    meta = pd.read_csv(metadata_csv)
    filename_col = detect_col(meta, ["filename", "file_name", "image_name", "image", "name", "path"], required=True)
    artist_col = detect_col(meta, ["artist", "artist_name", "creator", "author"], required=True)
    style_col = detect_col(meta, ["style", "label", "genre", "class"])
    split_col = detect_col(meta, ["split", "set", "subset"])
    print("Metadata CSV:", metadata_csv)
    print("Metadata columns detected:", {"filename": filename_col, "artist": artist_col, "style": style_col, "split": split_col})
    print("Metadata rows:", len(meta))

    if len(meta) < 60000:
        raise RuntimeError(f"Metadata has only {len(meta)} rows; expected at least 60000")

    style_s = metadata_style_series(meta, style_col)
    candidates: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

    if split_col is not None:
        tr = meta[split_mask_from_metadata(meta, split_col, "train")].copy()
        te = meta[split_mask_from_metadata(meta, split_col, "test")].copy()
        if len(tr) == 50000 and len(te) == 10000:
            candidates.append(("metadata_split_original_order", tr, te))

    # Common alternative: CSV itself is ordered train rows followed by test rows.
    if len(meta) >= 60000:
        candidates.append(("metadata_first50k_last10k", meta.iloc[:50000].copy(), meta.iloc[50000:60000].copy()))
        candidates.append(("metadata_last50k_first10k", meta.iloc[10000:60000].copy(), meta.iloc[:10000].copy()))

    py_train = py_idx[py_idx["split"] == "train"].sort_values("python_order").reset_index(drop=True)
    py_test = py_idx[py_idx["split"] == "test"].sort_values("python_order").reset_index(drop=True)
    diagnostics = []
    accepted = None

    for name, tr, te in candidates:
        if len(tr) != 50000 or len(te) != 10000:
            continue
        if style_col is None:
            diagnostics.append({"candidate": name, "train_style_agreement": np.nan, "test_style_agreement": np.nan, "accepted": False, "reason": "no style column"})
            continue
        tr_styles = tr[style_col].astype(str).map(norm_token).to_numpy()
        te_styles = te[style_col].astype(str).map(norm_token).to_numpy()
        agr_tr = float(np.mean(tr_styles == py_train["style"].to_numpy()))
        agr_te = float(np.mean(te_styles == py_test["style"].to_numpy()))
        transitions = int(np.sum(py_train["style"].to_numpy()[1:] != py_train["style"].to_numpy()[:-1]))
        ok = agr_tr >= 0.9999 and agr_te >= 0.9999 and transitions > 1000
        diagnostics.append({"candidate": name, "train_style_agreement": agr_tr, "test_style_agreement": agr_te, "python_train_style_transitions": transitions, "accepted": ok})
        print(f"Metadata-order audit {name}: train style agreement={agr_tr:.6f}, test={agr_te:.6f}, train transitions={transitions}")
        if ok and accepted is None:
            accepted = (name, tr.reset_index(drop=True), te.reset_index(drop=True))

    diag = pd.DataFrame(diagnostics)
    if accepted is None:
        raise RuntimeError(
            "The Kaggle CIFAR batches omit filenames and no metadata row-order candidate could be "
            "validated against the interleaved official style-label sequence. Stopping rather than "
            "attach uncertain artist identities. Diagnostics:\n" + diag.to_string(index=False)
        )

    name, tr, te = accepted
    print("Validated metadata↔CIFAR row alignment ✓", name)
    for split, pydf, mdf in [("train", py_train, tr), ("test", py_test, te)]:
        filenames = mdf[filename_col].astype(str).map(lambda x: Path(x).name).to_numpy()
        if pd.Series(filenames).duplicated().any():
            # Duplicated basename can be legitimate across styles, so validate compound key later.
            print(f"Note: duplicated basenames exist in {split}; compound split/style/filename key will be checked.")
        py_idx.loc[pydf.index if False else [], "filename"] = []  # no-op; explicit assignment below
        order_to_filename = dict(zip(pydf["python_order"].astype(int), filenames))
        mask = py_idx["split"] == split
        py_idx.loc[mask, "filename"] = py_idx.loc[mask, "python_order"].astype(int).map(order_to_filename)

    if py_idx["filename"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Filename recovery left empty rows")
    py_idx["key"] = py_idx["split"].astype(str) + "||" + py_idx["style"].astype(str) + "||" + py_idx["filename"].astype(str)
    if py_idx["key"].duplicated().any():
        dup = py_idx[py_idx["key"].duplicated(keep=False)].head(20)
        raise RuntimeError(f"Recovered duplicate split/style/filename keys:\n{dup}")
    return py_idx, diag


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
    rgb32: np.ndarray,
    audit_rows: list[dict],
) -> dict[int, str]:
    """Recover official filenames for HF 256px rows and validate with independent pixel evidence."""
    labels = np.asarray(hf_split["label"], dtype=int)
    mapping: dict[int, str] = {}
    filters = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "box": Image.Resampling.BOX,
    }

    # py_idx row number equals rgb32 row number from load_python_index.
    work = py_idx.copy()
    work["rgb_row"] = np.arange(len(work), dtype=int)

    for label_id, raw_name in enumerate(label_names):
        style = norm_token(raw_name)
        hf_indices = np.flatnonzero(labels == label_id)
        py = work[(work["split"] == split_name) & (work["style"] == style)].copy()
        if len(hf_indices) != len(py):
            raise RuntimeError(f"Count mismatch for {split_name}/{style}: HF={len(hf_indices)} Python={len(py)}")

        candidates = {
            "python_order": py.sort_values("python_order").reset_index(drop=True),
            "filename_order": py.sort_values("filename", kind="stable").reset_index(drop=True),
        }
        n = len(hf_indices)
        sample_pos = np.unique(np.linspace(0, n - 1, min(48, n), dtype=int))
        scores = []
        for order_name, cdf in candidates.items():
            for filt_name, filt in filters.items():
                vals = []
                shifted = []
                for pos in sample_pos:
                    hidx = int(hf_indices[pos])
                    pil = pil_rgb(hf_split[hidx]["image"])
                    row = cdf.iloc[int(pos)]
                    ref = rgb32[int(row["rgb_row"])]
                    vals.append(mae32(pil, ref, filt))
                    row_shift = cdf.iloc[int((pos + 1) % n)]
                    ref_shift = rgb32[int(row_shift["rgb_row"])]
                    shifted.append(mae32(pil, ref_shift, filt))
                med = float(np.median(vals))
                p95 = float(np.quantile(vals, 0.95))
                mismatch_med = float(np.median(shifted))
                ratio = med / max(mismatch_med, 1e-9)
                scores.append((ratio, med, p95, order_name, filt_name))
        scores.sort(key=lambda x: (x[0], x[1]))
        ratio, med, p95, order_name, filt_name = scores[0]
        audit_rows.append({
            "split": split_name,
            "style": style,
            "n": n,
            "chosen_order": order_name,
            "best_resize_filter": filt_name,
            "median_mae_0_255": med,
            "p95_mae_0_255": p95,
            "median_to_shifted_ratio": ratio,
        })
        print(
            f"Mapping audit {split_name}/{style}: order={order_name}, filter={filt_name}, "
            f"median MAE={med:.2f}, p95={p95:.2f}, ratio-vs-shift={ratio:.3f}"
        )
        if not (ratio < 0.55 and med < 40.0):
            raise RuntimeError(
                f"Could not validate HF↔official filename recovery for {split_name}/{style}. "
                f"Best median MAE={med:.2f}, shifted ratio={ratio:.3f}. "
                "Stopping rather than risking incorrect artist assignments."
            )
        chosen = candidates[order_name]
        for pos, hidx in enumerate(hf_indices):
            mapping[int(hidx)] = str(chosen.iloc[int(pos)]["filename"])
    return mapping


def main(kaggle_root: Path, output_root: Path, audit_csv: Path):
    try:
        from datasets import Image as HFImage, load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets first: pip install 'datasets>=3.0'") from exc

    batch_root = find_batch_root(kaggle_root)
    print("ArtBench python batch root:", batch_root)
    py_idx, rgb32, _ = load_python_index(batch_root)

    if py_idx["filename"].fillna("").astype(str).str.strip().eq("").any():
        print("CIFAR batches omit filenames; validating filename recovery from ArtBench-10.csv row order...")
        py_idx, order_diag = recover_filenames_from_metadata(py_idx, kaggle_root)
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        order_diag.to_csv(audit_csv.with_name("metadata_cifar_order_audit.csv"), index=False)
    else:
        print("CIFAR batches contain filenames; metadata-order recovery not needed.")

    print("Loading 256px mirror from Hugging Face:", HF_DATASET)
    ds = load_dataset(HF_DATASET)
    if len(ds["train"]) != 50000 or len(ds["test"]) != 10000:
        raise RuntimeError(f"Unexpected HF sizes train={len(ds['train'])}, test={len(ds['test'])}; expected 50000/10000")
    label_names = [norm_token(x) for x in ds["train"].features["label"].names]
    print("HF label names:", label_names)
    if set(label_names) != set(py_idx["style"].unique()):
        raise RuntimeError(f"Style-name mismatch HF={label_names} vs python={sorted(py_idx['style'].unique())}")

    audit_rows: list[dict] = []
    assignments = {}
    for split in ["train", "test"]:
        assignments[split] = candidate_assignment(ds[split], split, label_names, py_idx, rgb32, audit_rows)
    audit = pd.DataFrame(audit_rows)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_csv, index=False)
    print("HF↔official filename-recovery audit ->", audit_csv)

    raw_ds = {split: ds[split].cast_column("image", HFImage(decode=False)) for split in ["train", "test"]}

    if output_root.exists():
        print("Reusing materialization root if files already exist:", output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    materialized_rows = []
    for split in ["train", "test"]:
        labels = np.asarray(ds[split]["label"], dtype=int)
        for i in tqdm(range(len(ds[split])), desc=f"Materialize HF 256 {split}", dynamic_ncols=True):
            style = label_names[int(labels[i])]
            official_fn = assignments[split][i]
            raw = raw_ds[split][i]["image"]
            target = output_root / split / style / Path(official_fn).name
            target.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(raw, dict) and raw.get("bytes"):
                if not target.exists() or target.stat().st_size <= 100:
                    target.write_bytes(raw["bytes"])
                physical_fn = target.name
                physical_path = target
            elif isinstance(raw, dict) and raw.get("path") and Path(raw["path"]).exists():
                if not target.exists() or target.stat().st_size <= 100:
                    shutil.copy2(raw["path"], target)
                physical_fn = target.name
                physical_path = target
            else:
                pil = pil_rgb(ds[split][i]["image"])
                physical_path = target.with_suffix(".png")
                if not physical_path.exists() or physical_path.stat().st_size <= 100:
                    pil.save(physical_path, format="PNG")
                physical_fn = physical_path.name

            materialized_rows.append({
                "split": split,
                "style": style,
                "official_filename": official_fn,
                "physical_filename": physical_fn,
                "path": str(physical_path),
            })

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
