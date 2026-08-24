from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def stratified_bootstrap_indices(y_true, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true)
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def macro_f1_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        vals[i] = f1_score(y[idx], pred[idx], average="macro", zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(y_true, pred_new, pred_ref, n_boot: int = 5000, seed: int = 123) -> dict[str, float]:
    y = np.asarray(y_true)
    a = np.asarray(pred_new)
    b = np.asarray(pred_ref)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        vals[i] = (
            f1_score(y[idx], a[idx], average="macro", zero_division=0)
            - f1_score(y[idx], b[idx], average="macro", zero_division=0)
        )
    obs = (
        f1_score(y, a, average="macro", zero_division=0)
        - f1_score(y, b, average="macro", zero_division=0)
    )
    return {
        "delta_macro_f1": float(obs),
        "delta_ci_low": float(np.percentile(vals, 2.5)),
        "delta_ci_high": float(np.percentile(vals, 97.5)),
        "bootstrap_p_improvement": float((1 + np.sum(vals <= 0.0)) / (n_boot + 1)),
        "n_boot": int(n_boot),
    }


def clean_mask(predictions: pd.DataFrame, excluded_path: Path | None) -> np.ndarray:
    mask = np.ones(len(predictions), dtype=bool)
    if excluded_path is None or not excluded_path.exists():
        return mask
    excluded = pd.read_csv(excluded_path)
    if excluded.empty:
        return mask
    keys = set(zip(excluded["artist"].astype(str), excluded["filename"].astype(str)))
    current = zip(predictions["artist"].astype(str), predictions["filename"].astype(str))
    return np.array([key not in keys for key in current], dtype=bool)


def main(
    predictions_path: Path,
    scale_results_path: Path,
    output_dir: Path,
    excluded_test_path: Path | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(predictions_path)
    original = pd.read_csv(scale_results_path)
    mask = clean_mask(pred, excluded_test_path)
    clean = pred.loc[mask].reset_index(drop=True)
    y = clean["artist"].to_numpy()

    experiment_cols = [c for c in pred.columns if c not in {"artist", "filename"}]
    rows = []
    for name in experiment_cols:
        p = clean[name].to_numpy()
        lo, hi = macro_f1_ci(y, p)
        cv_match = original.loc[original["experiment"] == name, "best_cv_macro_f1"]
        rows.append({
            "experiment": name,
            "n_test_clean": int(len(clean)),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "macro_f1_ci_low": lo,
            "macro_f1_ci_high": hi,
            "best_cv_macro_f1": float(cv_match.iloc[0]) if len(cv_match) else np.nan,
        })
    clean_results = pd.DataFrame(rows)
    clean_results.to_csv(output_dir / "scale_ablation_results_clean.csv", index=False)

    single_names = [x for x in ["S1", "S2", "S4", "S8"] if x in experiment_cols]
    cv_map = dict(zip(original["experiment"], original["best_cv_macro_f1"]))
    best_single = max(single_names, key=lambda x: cv_map.get(x, -np.inf))

    comparisons = [
        ("S1248", best_single, "all_scales_vs_best_single_by_training_cv"),
        ("S1248_orient", "S1248", "orientation_increment"),
        ("S1248", "S124", "sigma8_increment_given_124"),
    ]
    delta_rows = []
    for new, ref, label in comparisons:
        if new not in clean.columns or ref not in clean.columns:
            continue
        d = paired_delta(y, clean[new].to_numpy(), clean[ref].to_numpy())
        d.update({
            "comparison": label,
            "new_model": new,
            "reference": ref,
            "n_test_clean": int(len(clean)),
        })
        delta_rows.append(d)
    pd.DataFrame(delta_rows).to_csv(output_dir / "scale_ablation_deltas_clean.csv", index=False)

    per_artist_rows = []
    for name in experiment_cols:
        for artist in sorted(clean["artist"].astype(str).unique()):
            keep = clean["artist"].astype(str).to_numpy() == artist
            per_artist_rows.append({
                "experiment": name,
                "artist": artist,
                "n": int(np.sum(keep)),
                "recall": float(np.mean(clean.loc[keep, name].astype(str).to_numpy() == y[keep])),
            })
    pd.DataFrame(per_artist_rows).to_csv(output_dir / "scale_ablation_per_artist_clean.csv", index=False)

    metadata = pd.DataFrame([{
        "n_test_raw": len(pred),
        "n_test_clean": len(clean),
        "n_excluded": int((~mask).sum()),
        "best_single_by_training_cv": best_single,
        "exclusions_applied": bool(excluded_test_path is not None and excluded_test_path.exists()),
    }])
    metadata.to_csv(output_dir / "scale_ablation_clean_metadata.csv", index=False)

    print("\nLeakage-clean scale results:")
    print(clean_results.sort_values("macro_f1", ascending=False).to_string(index=False))
    print("\nLeakage-clean paired deltas:")
    print(pd.DataFrame(delta_rows).to_string(index=False))
    print("\nMetadata:")
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Recompute Phase-III scale metrics after Phase-II validation exclusions.")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--scale-results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3b_scale_clean"))
    p.add_argument("--excluded-test", type=Path, default=None)
    args = p.parse_args()
    main(args.predictions, args.scale_results, args.output_dir, args.excluded_test)
