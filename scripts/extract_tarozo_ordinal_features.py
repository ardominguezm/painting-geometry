from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from tqdm import tqdm

try:
    import ordpy
except ImportError as exc:
    raise ImportError(
        "Phase VI requires ordpy>=1.2.0 (two_by_two_patterns). Install with: pip install 'ordpy>=1.2.0'"
    ) from exc


def locate_imagefolder_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    direct = dataset_root
    if (direct / "train").is_dir() and (direct / "test").is_dir():
        return direct
    candidates = []
    for tr in dataset_root.rglob("train"):
        if tr.is_dir() and (tr.parent / "test").is_dir():
            n_classes = sum(p.is_dir() for p in tr.iterdir())
            if n_classes >= 8:
                candidates.append(tr.parent)
    if not candidates:
        raise FileNotFoundError(f"Could not locate ArtBench train/test ImageFolder below {dataset_root}")
    candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def resolve_image_path(root: Path, split: str, style: str, filename: str) -> Path:
    p = root / str(split) / str(style) / str(filename)
    if p.exists():
        return p
    matches = list(root.rglob(str(filename)))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Image not found: {split}/{style}/{filename}")
    for m in matches:
        if str(split) in m.parts and str(style) in m.parts:
            return m
    return matches[0]


def load_tarozo_grayscale(path: Path) -> np.ndarray:
    """Closest public-paper reproduction: 24-bit RGB -> skimage standard luminance grayscale."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return np.asarray(rgb2gray(rgb), dtype=np.float64)


def sanitize_pattern_key(key: str) -> str:
    return str(key).replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "").replace(" ", "")


def extract_one(gray: np.ndarray, tie_precision=None) -> dict[str, float]:
    p75 = ordpy.two_by_two_patterns(
        gray,
        taux=1,
        tauy=1,
        overlapping=True,
        tie_patterns=True,
        group_patterns=False,
        tie_precision=tie_precision,
    )
    p11 = ordpy.two_by_two_patterns(
        gray,
        taux=1,
        tauy=1,
        overlapping=True,
        tie_patterns=True,
        group_patterns=True,
        tie_precision=tie_precision,
    )
    p24 = ordpy.two_by_two_patterns(
        gray,
        taux=1,
        tauy=1,
        overlapping=True,
        tie_patterns=False,
        group_patterns=False,
        tie_precision=tie_precision,
    )

    if len(p75) != 75:
        raise RuntimeError(f"Expected 75 tied ordinal patterns, got {len(p75)}")
    if len(p11) != 11:
        raise RuntimeError(f"Expected 11 pattern groups, got {len(p11)}")
    if len(p24) != 24:
        raise RuntimeError(f"Expected 24 standard ordinal patterns, got {len(p24)}")

    probs24 = np.asarray(list(p24.values()), dtype=float)
    H, C = ordpy.complexity_entropy(probs24, dx=2, dy=2, probs=True)

    out: dict[str, float] = {}
    for pattern, prob in p75.items():
        out[f"ord75__{sanitize_pattern_key(pattern)}"] = float(prob)
    for group, prob in p11.items():
        out[f"ord11__{str(group)}"] = float(prob)
    for pattern, prob in p24.items():
        out[f"ord24__{sanitize_pattern_key(pattern)}"] = float(prob)
    out["ordhc__H"] = float(H)
    out["ordhc__C"] = float(C)

    tie_mass = 0.0
    for pattern, prob in p75.items():
        digits = sanitize_pattern_key(pattern)
        if len(set(digits)) < 4:
            tie_mass += float(prob)
    out["ordmeta__tie_pattern_mass"] = float(tie_mass)
    out["ordmeta__type_A_0000"] = float(p75.get("[0000]", 0.0))
    out["ordmeta__sum75"] = float(sum(p75.values()))
    out["ordmeta__sum11"] = float(sum(p11.values()))
    out["ordmeta__sum24"] = float(sum(p24.values()))
    return out


def main(
    features_path: Path,
    dataset_root: Path,
    output: Path,
    checkpoint_every: int,
    tie_precision: int | None,
):
    if not hasattr(ordpy, "two_by_two_patterns"):
        raise RuntimeError(
            f"Installed ordpy {getattr(ordpy, '__version__', 'unknown')} lacks two_by_two_patterns; require >=1.2.0"
        )

    base = pd.read_csv(features_path)
    required = {"split", "style", "filename"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Feature matrix missing required metadata columns: {sorted(missing)}")

    root = locate_imagefolder_root(dataset_root)
    print("ArtBench ImageFolder root:", root)
    print("Input rows:", len(base))
    print("ordpy version:", getattr(ordpy, "__version__", "unknown"))
    print("tie_precision:", tie_precision)

    output.parent.mkdir(parents=True, exist_ok=True)

    ordinal_rows: list[dict] = []
    failures: list[dict] = []
    for i, row in tqdm(base.iterrows(), total=len(base), desc="Tarozo ordinal features"):
        try:
            p = resolve_image_path(root, row["split"], row["style"], row["filename"])
            gray = load_tarozo_grayscale(p)
            feats = extract_one(gray, tie_precision=tie_precision)
            feats["__row_index"] = int(i)
            ordinal_rows.append(feats)
        except Exception as exc:
            failures.append(
                {
                    "__row_index": int(i),
                    "split": row.get("split", ""),
                    "style": row.get("style", ""),
                    "filename": row.get("filename", ""),
                    "error": repr(exc),
                }
            )
        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            pd.DataFrame(ordinal_rows).to_csv(output.with_suffix(".ordinal_checkpoint.csv"), index=False)
            pd.DataFrame(failures).to_csv(output.with_suffix(".failures.csv"), index=False)

    ord_df = pd.DataFrame(ordinal_rows).set_index("__row_index").sort_index()
    if failures:
        pd.DataFrame(failures).to_csv(output.with_suffix(".failures.csv"), index=False)
        print(f"WARNING: {len(failures)} failures. Rows with failures will be dropped from enriched output.")

    keep_idx = base.index.intersection(ord_df.index)
    enriched = pd.concat([base.loc[keep_idx].reset_index(drop=True), ord_df.loc[keep_idx].reset_index(drop=True)], axis=1)
    enriched.to_csv(output, index=False)

    meta_cols = [c for c in enriched.columns if c.startswith("ordmeta__")]
    print("Output:", output)
    print("Shape:", enriched.shape)
    print("O75 features:", sum(c.startswith("ord75__") for c in enriched.columns))
    print("O11 features:", sum(c.startswith("ord11__") for c in enriched.columns))
    print("O24 features:", sum(c.startswith("ord24__") for c in enriched.columns))
    print("HC features:", sum(c.startswith("ordhc__") for c in enriched.columns))
    if meta_cols:
        print("Ordinal diagnostics (mean):")
        print(enriched[meta_cols].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Tarozo et al. two-by-two ordinal-pattern features for the ArtBench pilot.")
    p.add_argument("--features", type=Path, required=True, help="Phase-IV artbench_pilot_features.csv")
    p.add_argument("--dataset-root", type=Path, required=True, help="Directory containing extracted ArtBench ImageFolder")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--tie-precision", type=int, default=None)
    args = p.parse_args()
    main(args.features, args.dataset_root, args.output, args.checkpoint_every, args.tie_precision)
