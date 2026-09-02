from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SOURCE_SPECIFIC_STYLES = {"surrealism", "ukiyo_e"}
SCALE_TAGS = {1.0: "1p0", 2.0: "2p0", 4.0: "4p0", 8.0: "8p0"}


def feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(prefix))


def curvature_scale_columns(df: pd.DataFrame, scales: tuple[float, ...]) -> list[str]:
    cols: list[str] = []
    for sigma in scales:
        tag = SCALE_TAGS[float(sigma)]
        prefix = f"geom__curv__kappa_ref_s{tag}_"
        cols.extend(c for c in df.columns if c.startswith(prefix))
    return sorted(set(cols))


def group_bootstrap_indices(y_true, groups, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    y = np.asarray(y_true)
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    all_classes = set(np.unique(y))
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(max_tries):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        if set(np.unique(y[idx])) == all_classes:
            rng.shuffle(idx)
            return idx
    return np.arange(len(y))


def bootstrap_f1(y_true, y_pred, groups, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = group_bootstrap_indices(y, groups, rng)
        vals[i] = f1_score(y[idx], pred[idx], average="macro", zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_group_delta(y_true, pred_new, pred_ref, groups, n_boot: int = 5000, seed: int = 123) -> dict[str, float]:
    y = np.asarray(y_true)
    a = np.asarray(pred_new)
    b = np.asarray(pred_ref)
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = group_bootstrap_indices(y, groups, rng)
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
        "bootstrap_p_new_gt_ref": float((1 + np.sum(vals <= 0.0)) / (len(vals) + 1)),
        "n_boot": int(n_boot),
    }


def build_pipeline(k: int | None = None) -> Pipeline:
    selector = "passthrough" if k is None else SelectKBest(score_func=f_classif, k=int(k))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("clf", SVC(kernel="rbf", cache_size=2048)),
        ]
    )


def parameter_grid() -> dict[str, list]:
    return {
        "clf__C": [1.0, 3.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.03],
    }


def experiment_map(df: pd.DataFrame, matched_baseline_k: int) -> dict[str, dict]:
    b = feature_columns(df, "base__")
    s1 = curvature_scale_columns(df, (1.0,))
    s2 = curvature_scale_columns(df, (2.0,))
    s4 = curvature_scale_columns(df, (4.0,))
    s8 = curvature_scale_columns(df, (8.0,))
    s12 = sorted(set(s1 + s2))
    s48 = sorted(set(s4 + s8))
    sall = sorted(set(s1 + s2 + s4 + s8))

    if not b or min(map(len, [s1, s2, s4, s8])) == 0:
        raise RuntimeError("Required baseline or scale-specific curvature columns are missing.")

    k = min(int(matched_baseline_k), len(b))

    exps = {
        "G_s1": {"cols": s1, "k": None, "family": "geometry_only", "scale_set": "1"},
        "G_s2": {"cols": s2, "k": None, "family": "geometry_only", "scale_set": "2"},
        "G_s4": {"cols": s4, "k": None, "family": "geometry_only", "scale_set": "4"},
        "G_s8": {"cols": s8, "k": None, "family": "geometry_only", "scale_set": "8"},
        "G_fine_s12": {"cols": s12, "k": None, "family": "geometry_only", "scale_set": "1+2"},
        "G_coarse_s48": {"cols": s48, "k": None, "family": "geometry_only", "scale_set": "4+8"},
        "G_all_s1248": {"cols": sall, "k": None, "family": "geometry_only", "scale_set": "1+2+4+8"},
        "B_strong": {"cols": b, "k": k, "family": "baseline", "scale_set": "none"},
        "BG_s1_kB": {"cols": sorted(set(b + s1)), "k": k, "family": "baseline_plus_geometry", "scale_set": "1"},
        "BG_s2_kB": {"cols": sorted(set(b + s2)), "k": k, "family": "baseline_plus_geometry", "scale_set": "2"},
        "BG_s4_kB": {"cols": sorted(set(b + s4)), "k": k, "family": "baseline_plus_geometry", "scale_set": "4"},
        "BG_s8_kB": {"cols": sorted(set(b + s8)), "k": k, "family": "baseline_plus_geometry", "scale_set": "8"},
        "BG_fine_s12_kB": {"cols": sorted(set(b + s12)), "k": k, "family": "baseline_plus_geometry", "scale_set": "1+2"},
        "BG_coarse_s48_kB": {"cols": sorted(set(b + s48)), "k": k, "family": "baseline_plus_geometry", "scale_set": "4+8"},
        "BG_all_s1248_kB": {"cols": sorted(set(b + sall)), "k": k, "family": "baseline_plus_geometry", "scale_set": "1+2+4+8"},
    }
    return exps


def artist_disjoint_scale_eval(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    matched_baseline_k: int,
    outer_folds: int,
    inner_folds: int,
    n_jobs: int,
    metric_boot: int,
    delta_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["artist"] = work["artist"].fillna("").astype(str).str.strip()
    work = work[work["artist"].ne("")].reset_index(drop=True)
    if work.empty:
        raise RuntimeError(f"{dataset_name}: no usable artist metadata.")

    style_artist_counts = work.groupby("style")["artist"].nunique()
    if (style_artist_counts < outer_folds).any():
        raise RuntimeError(
            f"{dataset_name}: not enough artists for {outer_folds} outer folds: "
            f"{style_artist_counts.to_dict()}"
        )

    y = work["style"].astype(str).to_numpy()
    groups = work["artist"].astype(str).to_numpy()
    exps = experiment_map(work, matched_baseline_k)

    outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    frozen_outer = list(outer.split(np.zeros((len(work), 1)), y, groups=groups))

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    oof_map: dict[str, np.ndarray] = {}

    for exp_idx, (name, spec) in enumerate(exps.items(), start=1):
        cols = spec["cols"]
        k = spec["k"]
        k_use = None if k is None else min(int(k), len(cols))
        oof = np.empty(len(work), dtype=object)
        fold_scores: list[float] = []
        print(f"[{dataset_name}] {exp_idx:02d}/{len(exps)} {name}: {len(cols)} -> {k_use or len(cols)} features")

        for fold, (tr_idx, te_idx) in enumerate(frozen_outer, start=1):
            Xtr = work.iloc[tr_idx][cols]
            Xte = work.iloc[te_idx][cols]
            ytr, yte = y[tr_idx], y[te_idx]
            gtr, gte = groups[tr_idx], groups[te_idx]

            inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=100 + fold)
            search = GridSearchCV(
                build_pipeline(k_use),
                parameter_grid(),
                scoring="f1_macro",
                cv=inner,
                refit=True,
                n_jobs=n_jobs,
                verbose=0,
            )
            search.fit(Xtr, ytr, groups=gtr)
            pred = search.predict(Xte)
            oof[te_idx] = pred
            score = float(f1_score(yte, pred, average="macro", zero_division=0))
            fold_scores.append(score)
            fold_rows.append(
                {
                    "dataset": dataset_name,
                    "experiment": name,
                    "family": spec["family"],
                    "scale_set": spec["scale_set"],
                    "fold": fold,
                    "n_train": len(tr_idx),
                    "n_test": len(te_idx),
                    "n_train_artists": len(np.unique(gtr)),
                    "n_test_artists": len(np.unique(gte)),
                    "macro_f1": score,
                    "best_inner_cv_macro_f1": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                }
            )

        oof_map[name] = oof
        lo, hi = bootstrap_f1(y, oof, groups, n_boot=metric_boot, seed=42)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "protocol": "artist_disjoint_nested_cv",
                "experiment": name,
                "family": spec["family"],
                "scale_set": spec["scale_set"],
                "n_images": len(work),
                "n_artists": len(np.unique(groups)),
                "n_features_input": len(cols),
                "n_features_selected": len(cols) if k_use is None else k_use,
                "macro_f1_oof": float(f1_score(y, oof, average="macro", zero_division=0)),
                "macro_f1_group_boot_ci_low": lo,
                "macro_f1_group_boot_ci_high": hi,
                "fold_macro_f1_mean": float(np.mean(fold_scores)),
                "fold_macro_f1_std": float(np.std(fold_scores, ddof=1)),
            }
        )

    oof_df = work[["style", "artist", "filename", "split"]].copy()
    for name, pred in oof_map.items():
        oof_df[name] = pred
    oof_df.to_csv(output_dir / f"{dataset_name}_phase4b_oof_predictions.csv", index=False)

    comparisons = [
        ("G_s1", "G_s8", "geometry_fine_extreme_vs_coarse_extreme"),
        ("G_fine_s12", "G_coarse_s48", "geometry_fine_pair_vs_coarse_pair"),
        ("BG_fine_s12_kB", "BG_coarse_s48_kB", "complementarity_fine_pair_vs_coarse_pair"),
        ("BG_s1_kB", "B_strong", "increment_scale_1_over_baseline"),
        ("BG_s2_kB", "B_strong", "increment_scale_2_over_baseline"),
        ("BG_s4_kB", "B_strong", "increment_scale_4_over_baseline"),
        ("BG_s8_kB", "B_strong", "increment_scale_8_over_baseline"),
        ("BG_fine_s12_kB", "B_strong", "increment_fine_1_2_over_baseline"),
        ("BG_coarse_s48_kB", "B_strong", "increment_coarse_4_8_over_baseline"),
        ("BG_all_s1248_kB", "B_strong", "increment_all_scales_over_baseline"),
    ]

    delta_rows: list[dict] = []
    for new, ref, label in comparisons:
        d = paired_group_delta(y, oof_map[new], oof_map[ref], groups, n_boot=delta_boot, seed=123)
        d.update(
            {
                "dataset": dataset_name,
                "protocol": "artist_disjoint_nested_cv",
                "comparison": label,
                "new_model": new,
                "reference": ref,
                "n_images": len(work),
                "n_artists": len(np.unique(groups)),
            }
        )
        delta_rows.append(d)

    per_style_rows: list[dict] = []
    report_models = [
        "G_s1", "G_s2", "G_s4", "G_s8", "G_fine_s12", "G_coarse_s48",
        "B_strong", "BG_s1_kB", "BG_s2_kB", "BG_s4_kB", "BG_s8_kB",
        "BG_fine_s12_kB", "BG_coarse_s48_kB",
    ]
    for model in report_models:
        pred = oof_map[model]
        for style in sorted(np.unique(y)):
            true_bin = (y == style).astype(int)
            pred_bin = (pred == style).astype(int)
            per_style_rows.append(
                {
                    "dataset": dataset_name,
                    "experiment": model,
                    "style": style,
                    "n_style_images": int(np.sum(y == style)),
                    "f1_one_vs_rest": float(f1_score(true_bin, pred_bin, zero_division=0)),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(delta_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(per_style_rows),
    )


def main(
    features_path: Path,
    output_dir: Path,
    matched_baseline_k: int,
    outer_folds: int,
    inner_folds: int,
    n_jobs: int,
    metric_boot: int,
    delta_boot: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path)
    df["style"] = df["style"].astype(str)
    if "artist" not in df.columns:
        raise RuntimeError("Artist metadata are required for Phase IVb.")

    datasets = {
        "artbench10_all": df,
        "artbench10_wikiart8": df[~df["style"].isin(SOURCE_SPECIFIC_STYLES)].reset_index(drop=True),
    }

    all_results, all_deltas, all_folds, all_per_style = [], [], [], []
    for dataset_name, sub in datasets.items():
        print(f"\n=== {dataset_name}: {len(sub)} images, {sub['style'].nunique()} styles ===")
        result, delta, folds, per_style = artist_disjoint_scale_eval(
            sub,
            dataset_name,
            output_dir,
            matched_baseline_k,
            outer_folds,
            inner_folds,
            n_jobs,
            metric_boot,
            delta_boot,
        )
        all_results.append(result)
        all_deltas.append(delta)
        all_folds.append(folds)
        all_per_style.append(per_style)

    results = pd.concat(all_results, ignore_index=True)
    deltas = pd.concat(all_deltas, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    per_style = pd.concat(all_per_style, ignore_index=True)

    results.to_csv(output_dir / "phase4b_scale_hierarchy_results.csv", index=False)
    deltas.to_csv(output_dir / "phase4b_scale_hierarchy_deltas.csv", index=False)
    folds.to_csv(output_dir / "phase4b_scale_hierarchy_fold_results.csv", index=False)
    per_style.to_csv(output_dir / "phase4b_scale_hierarchy_per_style.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "features_path": str(features_path),
                "n_input_images": len(df),
                "n_input_styles": int(df["style"].nunique()),
                "n_input_artists": int(df["artist"].fillna("").astype(str).str.strip().replace("", np.nan).nunique()),
                "matched_baseline_k": int(matched_baseline_k),
                "outer_folds": int(outer_folds),
                "inner_folds": int(inner_folds),
                "metric_group_boot": int(metric_boot),
                "delta_group_boot": int(delta_boot),
                "primary_scale_contrast": "G_fine_s12 - G_coarse_s48",
                "primary_complementarity_scale_contrast": "BG_fine_s12_kB - BG_coarse_s48_kB",
                "interpretation": "All uncertainty uses artist-level group bootstrap; outer test artists are unseen in training.",
            }
        ]
    )
    metadata.to_csv(output_dir / "phase4b_metadata.csv", index=False)

    print("\nPhase IVb summary:")
    show = results[["dataset", "experiment", "scale_set", "macro_f1_oof", "macro_f1_group_boot_ci_low", "macro_f1_group_boot_ci_high"]]
    print(show.to_string(index=False))
    print("\nPre-specified paired contrasts:")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase IVb: scale hierarchy of ArtBench style information under artist-disjoint evaluation.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase4b_artbench_scale_hierarchy"))
    p.add_argument("--matched-baseline-k", type=int, default=90)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--metric-boot", type=int, default=2000)
    p.add_argument("--delta-boot", type=int, default=5000)
    args = p.parse_args()
    main(
        args.features,
        args.output_dir,
        args.matched_baseline_k,
        args.outer_folds,
        args.inner_folds,
        args.n_jobs,
        args.metric_boot,
        args.delta_boot,
    )
