from __future__ import annotations

import argparse
import itertools
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
    if (dataset_root / "train").is_dir() and (dataset_root / "test").is_dir():
        return dataset_root
    candidates = []
    for tr in dataset_root.rglob("train"):
        if tr.is_dir() and (tr.parent / "test").is_dir():
            n_classes = sum(p.is_dir() for p in tr.iterdir())
            if n_classes >= 8:
                candidates.append(tr.parent)
    if not candidates:
        raise FileNotFoundError(f"Could not locate ArtBench train/test ImageFolder below {dataset_root}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


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
    return (
        str(key)
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" ", "")
    )


def pattern_string(pattern: tuple[int, int, int, int]) -> str:
    return "[" + "".join(str(int(x)) for x in pattern) + "]"


def all_tie_patterns() -> list[tuple[int, int, int, int]]:
    """All 75 dense weak-order rank patterns of four elements."""
    out = []
    for tup in itertools.product(range(4), repeat=4):
        used = set(tup)
        if used == set(range(max(tup) + 1)):
            out.append(tuple(int(x) for x in tup))
    if len(out) != 75:
        raise RuntimeError(f"Internal error: generated {len(out)} tie-aware patterns instead of 75")
    return out


TIE_PATTERNS = all_tie_patterns()
TIE_PATTERN_STRINGS = [pattern_string(p) for p in TIE_PATTERNS]
TIE_CODES = np.asarray([p[0] * 64 + p[1] * 16 + p[2] * 4 + p[3] for p in TIE_PATTERNS], dtype=np.int16)
STANDARD_PATTERNS = [tuple(int(x) for x in p) for p in itertools.permutations(range(4))]
STANDARD_PATTERN_STRINGS = [pattern_string(p) for p in STANDARD_PATTERNS]
GROUP_NAMES = list("ABCDEFGHIJK")


def build_ordinal_maps():
    """Use ordpy itself once on the 75 canonical patterns to reproduce its grouping/tie-break maps exactly."""
    group_of: dict[str, str] = {}
    standard_of: dict[str, str] = {}
    for pat, key in zip(TIE_PATTERNS, TIE_PATTERN_STRINGS):
        arr = np.asarray(pat, dtype=float).reshape(2, 2)
        g = ordpy.two_by_two_patterns(
            arr, taux=1, tauy=1, overlapping=True,
            tie_patterns=True, group_patterns=True,
        )
        group_of[key] = max(g, key=g.get)
        s = ordpy.two_by_two_patterns(
            arr, taux=1, tauy=1, overlapping=True,
            tie_patterns=False, group_patterns=False,
        )
        if len(s) != 1:
            raise RuntimeError(f"Unexpected standard mapping for {key}: {s}")
        standard_of[key] = str(next(iter(s.keys())))
    return group_of, standard_of


GROUP_OF, STANDARD_OF = build_ordinal_maps()


def dense_rank_codes_2x2(gray: np.ndarray, tie_precision: int | None = None) -> np.ndarray:
    """Vectorized dense ranks for every overlapping 2x2 patch, exactly matching tie-aware ordinal ranking."""
    a = np.asarray(gray)
    if tie_precision is not None:
        a = np.round(a, int(tie_precision))
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError(f"Expected a 2D image at least 2x2, got {a.shape}")

    patches = np.stack(
        [
            a[:-1, :-1].ravel(),
            a[:-1, 1:].ravel(),
            a[1:, :-1].ravel(),
            a[1:, 1:].ravel(),
        ],
        axis=1,
    )

    order = np.argsort(patches, axis=1, kind="stable")
    sorted_vals = np.take_along_axis(patches, order, axis=1)
    sorted_ranks = np.zeros(order.shape, dtype=np.uint8)
    sorted_ranks[:, 1:] = np.cumsum(sorted_vals[:, 1:] != sorted_vals[:, :-1], axis=1, dtype=np.uint8)
    ranks = np.empty_like(sorted_ranks)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)

    return (
        ranks[:, 0].astype(np.int16) * 64
        + ranks[:, 1].astype(np.int16) * 16
        + ranks[:, 2].astype(np.int16) * 4
        + ranks[:, 3].astype(np.int16)
    )


def fast_p75(gray: np.ndarray, tie_precision: int | None = None) -> dict[str, float]:
    codes = dense_rank_codes_2x2(gray, tie_precision=tie_precision)
    counts = np.bincount(codes, minlength=256)
    denom = float(codes.size)
    return {key: float(counts[int(code)] / denom) for key, code in zip(TIE_PATTERN_STRINGS, TIE_CODES)}


def aggregate_from_p75(p75: dict[str, float]):
    p11 = {g: 0.0 for g in GROUP_NAMES}
    p24 = {k: 0.0 for k in STANDARD_PATTERN_STRINGS}
    for key, prob in p75.items():
        p11[GROUP_OF[key]] += float(prob)
        p24[STANDARD_OF[key]] = p24.get(STANDARD_OF[key], 0.0) + float(prob)
    return p11, p24


def validate_fast_implementation() -> None:
    """Small exact-equivalence tests against ordpy before processing the corpus."""
    rng = np.random.default_rng(20260826)
    for case in range(4):
        # Integer-valued synthetic images create many ties and are therefore a stringent test.
        arr = rng.integers(0, 7, size=(9 + case, 11 + case)).astype(float)
        fast75 = fast_p75(arr)
        slow75_raw = ordpy.two_by_two_patterns(
            arr, taux=1, tauy=1, overlapping=True,
            tie_patterns=True, group_patterns=False,
        )
        slow75 = {k: 0.0 for k in TIE_PATTERN_STRINGS}
        for k, v in slow75_raw.items():
            slow75[str(k)] = float(v)

        f75 = np.asarray([fast75[k] for k in TIE_PATTERN_STRINGS])
        s75 = np.asarray([slow75[k] for k in TIE_PATTERN_STRINGS])
        if not np.allclose(f75, s75, atol=1e-12, rtol=0):
            raise RuntimeError(f"Fast OP75 validation failed in synthetic case {case}; max diff={np.max(np.abs(f75-s75))}")

        fast11, fast24 = aggregate_from_p75(fast75)
        slow11 = ordpy.two_by_two_patterns(
            arr, taux=1, tauy=1, overlapping=True,
            tie_patterns=True, group_patterns=True,
        )
        slow24_raw = ordpy.two_by_two_patterns(
            arr, taux=1, tauy=1, overlapping=True,
            tie_patterns=False, group_patterns=False,
        )
        slow24 = {k: 0.0 for k in STANDARD_PATTERN_STRINGS}
        for k, v in slow24_raw.items():
            slow24[str(k)] = float(v)

        if not np.allclose(
            [fast11[g] for g in GROUP_NAMES],
            [float(slow11.get(g, 0.0)) for g in GROUP_NAMES],
            atol=1e-12, rtol=0,
        ):
            raise RuntimeError(f"Fast OP11 validation failed in synthetic case {case}")
        if not np.allclose(
            [fast24[k] for k in STANDARD_PATTERN_STRINGS],
            [slow24[k] for k in STANDARD_PATTERN_STRINGS],
            atol=1e-12, rtol=0,
        ):
            raise RuntimeError(f"Fast OP24 validation failed in synthetic case {case}")
    print("Fast ordinal extractor validated exactly against ordpy on tie-rich synthetic images.")


def extract_one(gray: np.ndarray, tie_precision=None) -> dict[str, float]:
    p75 = fast_p75(gray, tie_precision=tie_precision)
    p11, p24 = aggregate_from_p75(p75)

    probs24 = np.asarray([p24[k] for k in STANDARD_PATTERN_STRINGS], dtype=float)
    H, C = ordpy.complexity_entropy(probs24, dx=2, dy=2, probs=True)

    out: dict[str, float] = {}
    for pattern in TIE_PATTERN_STRINGS:
        out[f"ord75__{sanitize_pattern_key(pattern)}"] = float(p75[pattern])
    for group in GROUP_NAMES:
        out[f"ord11__{group}"] = float(p11[group])
    for pattern in STANDARD_PATTERN_STRINGS:
        out[f"ord24__{sanitize_pattern_key(pattern)}"] = float(p24[pattern])
    out["ordhc__H"] = float(H)
    out["ordhc__C"] = float(C)

    tie_mass = sum(prob for pattern, prob in p75.items() if len(set(sanitize_pattern_key(pattern))) < 4)
    out["ordmeta__tie_pattern_mass"] = float(tie_mass)
    out["ordmeta__type_A_0000"] = float(p75["[0000]"])
    out["ordmeta__sum75"] = float(sum(p75.values()))
    out["ordmeta__sum11"] = float(sum(p11.values()))
    out["ordmeta__sum24"] = float(sum(p24.values()))
    return out


def normalize_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Older checkpoints can be sparse because ordpy returns only observed patterns; missing probabilities are zeros."""
    if df.empty:
        return df
    expected_prob_cols = (
        [f"ord75__{sanitize_pattern_key(k)}" for k in TIE_PATTERN_STRINGS]
        + [f"ord11__{g}" for g in GROUP_NAMES]
        + [f"ord24__{sanitize_pattern_key(k)}" for k in STANDARD_PATTERN_STRINGS]
    )
    for c in expected_prob_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[expected_prob_cols] = df[expected_prob_cols].fillna(0.0)
    return df


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

    validate_fast_implementation()

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
    checkpoint = output.with_suffix(".ordinal_checkpoint.csv")
    failure_path = output.with_suffix(".failures.csv")

    completed: dict[int, dict] = {}
    if checkpoint.exists():
        prior = normalize_checkpoint(pd.read_csv(checkpoint))
        if "__row_index" not in prior.columns:
            raise RuntimeError(f"Checkpoint lacks __row_index: {checkpoint}")
        for rec in prior.to_dict("records"):
            completed[int(rec["__row_index"])] = rec
        print(f"RESUME: loaded {len(completed)} completed paintings from {checkpoint}")

    pending = [i for i in range(len(base)) if i not in completed]
    print(f"Remaining paintings: {len(pending)} / {len(base)}")

    failures: list[dict] = []
    since_save = 0
    for i in tqdm(pending, total=len(pending), desc="Fast Tarozo ordinal features"):
        row = base.iloc[i]
        try:
            p = resolve_image_path(root, row["split"], row["style"], row["filename"])
            gray = load_tarozo_grayscale(p)
            feats = extract_one(gray, tie_precision=tie_precision)
            feats["__row_index"] = int(i)
            completed[int(i)] = feats
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
        since_save += 1
        if checkpoint_every > 0 and since_save >= checkpoint_every:
            ck = normalize_checkpoint(pd.DataFrame([completed[k] for k in sorted(completed)]))
            ck.to_csv(checkpoint, index=False)
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            print(f"Checkpoint: {len(completed)}/{len(base)} completed")
            since_save = 0

    ord_df = normalize_checkpoint(pd.DataFrame([completed[k] for k in sorted(completed)]))
    if ord_df.empty:
        raise RuntimeError("No ordinal features were extracted.")
    ord_df.to_csv(checkpoint, index=False)
    ord_df = ord_df.set_index("__row_index").sort_index()

    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"WARNING: {len(failures)} failures. Rows with failures will be dropped from enriched output.")
    elif failure_path.exists():
        failure_path.unlink()

    keep_idx = base.index.intersection(ord_df.index)
    enriched = pd.concat(
        [base.loc[keep_idx].reset_index(drop=True), ord_df.loc[keep_idx].reset_index(drop=True)],
        axis=1,
    )
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
