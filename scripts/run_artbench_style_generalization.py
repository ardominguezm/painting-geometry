from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SOURCE_SPECIFIC_STYLES = {"surrealism", "ukiyo_e"}


def feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(prefix))


def stratified_bootstrap_indices(y_true, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true)
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def group_bootstrap_indices(y_true, groups, rng: np.random.Generator, max_tries: int = 50) -> np.ndarray:
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


def bootstrap_metric(y_true, y_pred, groups=None, n_boot: int = 1000, seed: int = 42):
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = (
            stratified_bootstrap_indices(y, rng)
            if groups is None
            else group_bootstrap_indices(y, groups, rng)
        )
        vals[i] = f1_score(y[idx], p[idx], average="macro", zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(y_true, pred_new, pred_ref, groups=None, n_boot: int = 2000, seed: int = 123):
    y = np.asarray(y_true)
    a = np.asarray(pred_new)
    b = np.asarray(pred_ref)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = (
            stratified_bootstrap_indices(y, rng)
            if groups is None
            else group_bootstrap_indices(y, groups, rng)
        )
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


def build_pipeline(k: int | None):
    selector = "passthrough" if k is None else SelectKBest(score_func=f_classif, k=k)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("clf", SVC(kernel="rbf", cache_size=2048)),
        ]
    )


def grid():
    return {
        "clf__C": [1.0, 3.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.03],
    }


def experiment_map(df: pd.DataFrame, matched_k: int):
    b = feature_columns(df, "base__")
    g = feature_columns(df, "geom__")
    bg = sorted(set(b + g))
    return {
        "B_strong_full": (b, None),
        "G_geometry_full": (g, None),
        "BG_combined_full": (bg, None),
        f"B_strong_k{min(matched_k, len(b))}": (b, min(matched_k, len(b))),
        f"G_geometry_k{min(matched_k, len(g))}": (g, min(matched_k, len(g))),
        f"BG_combined_k{min(matched_k, len(bg))}": (bg, min(matched_k, len(bg))),
    }


def official_split_eval(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    matched_k: int,
    cv_folds: int,
    n_jobs: int,
    metric_boot: int,
    delta_boot: int,
):
    train = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    test = df[df["split"].astype(str) == "test"].reset_index(drop=True)
    y_train = train["style"].astype(str).to_numpy()
    y_test = test["style"].astype(str).to_numpy()
    exps = experiment_map(df, matched_k)
    rows, preds = [], {}

    for name, (cols, k) in exps.items():
        if not cols:
            continue
        k_use = None if k is None else min(int(k), len(cols))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        search = GridSearchCV(
            build_pipeline(k_use), grid(), scoring="f1_macro", cv=cv, refit=True, n_jobs=n_jobs, verbose=0
        )
        search.fit(train[cols], y_train)
        pred = search.predict(test[cols])
        preds[name] = pred
        lo, hi = bootstrap_metric(y_test, pred, n_boot=metric_boot)
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": "official_image_split",
                "experiment": name,
                "n_train": len(train),
                "n_test": len(test),
                "n_features_input": len(cols),
                "n_features_selected": len(cols) if k_use is None else k_use,
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "macro_f1_ci_low": lo,
                "macro_f1_ci_high": hi,
                "best_cv_macro_f1": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    pred_df = test[["style", "artist", "filename", "split"]].copy()
    for name, pred in preds.items():
        pred_df[name] = pred
    pred_df.to_csv(output_dir / f"{dataset_name}_official_predictions.csv", index=False)

    b_full = "B_strong_full"
    bg_full = "BG_combined_full"
    b_k = f"B_strong_k{min(matched_k, len(feature_columns(df, 'base__')))}"
    bg_k = f"BG_combined_k{min(matched_k, len(feature_columns(df, 'base__') + feature_columns(df, 'geom__')))}"
    deltas = []
    for new, ref in [(bg_full, b_full), (bg_k, b_k)]:
        if new in preds and ref in preds:
            d = paired_delta(y_test, preds[new], preds[ref], n_boot=delta_boot)
            d.update({"dataset": dataset_name, "protocol": "official_image_split", "new_model": new, "reference": ref})
            deltas.append(d)
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def artist_disjoint_eval(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    matched_k: int,
    outer_folds: int,
    inner_folds: int,
    n_jobs: int,
    metric_boot: int,
    delta_boot: int,
):
    work = df.copy()
    work["artist"] = work["artist"].fillna("").astype(str).str.strip()
    work = work[work["artist"].ne("")].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    counts = work.groupby("style")["artist"].nunique()
    if (counts < outer_folds).any():
        raise RuntimeError(f"Not enough unique artists for {outer_folds}-fold artist-disjoint CV: {counts.to_dict()}")

    y = work["style"].astype(str).to_numpy()
    groups = work["artist"].astype(str).to_numpy()
    exps = experiment_map(work, matched_k)
    outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    summary_rows, fold_rows, oof_cols = [], [], {}

    for name, (cols, k) in exps.items():
        k_use = None if k is None else min(int(k), len(cols))
        oof = np.empty(len(work), dtype=object)
        fold_scores = []

        for fold, (tr_idx, te_idx) in enumerate(outer.split(work[cols], y, groups=groups), start=1):
            Xtr = work.iloc[tr_idx][cols]
            Xte = work.iloc[te_idx][cols]
            ytr, yte = y[tr_idx], y[te_idx]
            gtr, gte = groups[tr_idx], groups[te_idx]

            inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=100 + fold)
            search = GridSearchCV(
                build_pipeline(k_use), grid(), scoring="f1_macro", cv=inner, refit=True, n_jobs=n_jobs, verbose=0
            )
            search.fit(Xtr, ytr, groups=gtr)
            pred = search.predict(Xte)
            oof[te_idx] = pred
            score = float(f1_score(yte, pred, average="macro", zero_division=0))
            fold_scores.append(score)
            fold_rows.append(
                {
                    "dataset": dataset_name,
                    "protocol": "artist_disjoint_nested_cv",
                    "experiment": name,
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

        oof_cols[name] = oof
        lo, hi = bootstrap_metric(y, oof, groups=groups, n_boot=metric_boot)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "protocol": "artist_disjoint_nested_cv",
                "experiment": name,
                "n_images": len(work),
                "n_artists": len(np.unique(groups)),
                "n_features_input": len(cols),
                "n_features_selected": len(cols) if k_use is None else k_use,
                "macro_f1_oof": float(f1_score(y, oof, average="macro", zero_division=0)),
                "macro_f1_group_boot_ci_low": lo,
                "macro_f1_group_boot_ci_high": hi,
                "fold_macro_f1_mean": float(np.mean(fold_scores)),
                "fold_macro_f1_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            }
        )

    oof_df = work[["style", "artist", "filename", "split"]].copy()
    for name, pred in oof_cols.items():
        oof_df[name] = pred
    oof_df.to_csv(output_dir / f"{dataset_name}_artist_disjoint_oof_predictions.csv", index=False)

    b_full = "B_strong_full"
    bg_full = "BG_combined_full"
    b_k = f"B_strong_k{min(matched_k, len(feature_columns(work, 'base__')))}"
    bg_k = f"BG_combined_k{min(matched_k, len(feature_columns(work, 'base__') + feature_columns(work, 'geom__')))}"
    deltas = []
    for new, ref in [(bg_full, b_full), (bg_k, b_k)]:
        if new in oof_cols and ref in oof_cols:
            d = paired_delta(y, oof_cols[new], oof_cols[ref], groups=groups, n_boot=delta_boot)
            d.update(
                {
                    "dataset": dataset_name,
                    "protocol": "artist_disjoint_nested_cv",
                    "new_model": new,
                    "reference": ref,
                    "n_images": len(work),
                    "n_artists": len(np.unique(groups)),
                }
            )
            deltas.append(d)

    return pd.DataFrame(summary_rows), pd.DataFrame(deltas), pd.DataFrame(fold_rows)


def run_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    matched_k: int,
    cv_folds: int,
    outer_folds: int,
    inner_folds: int,
    n_jobs: int,
    metric_boot: int,
    delta_boot: int,
):
    off_res, off_delta = official_split_eval(
        df, dataset_name, output_dir, matched_k, cv_folds, n_jobs, metric_boot, delta_boot
    )
    artist_match_rate = float(df["artist"].fillna("").astype(str).str.strip().ne("").mean())
    if artist_match_rate >= 0.90:
        grp_res, grp_delta, grp_folds = artist_disjoint_eval(
            df, dataset_name, output_dir, matched_k, outer_folds, inner_folds, n_jobs, metric_boot, delta_boot
        )
    else:
        print(f"WARNING {dataset_name}: artist metadata coverage {artist_match_rate:.1%}; artist-disjoint CV skipped.")
        grp_res, grp_delta, grp_folds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return off_res, off_delta, grp_res, grp_delta, grp_folds


def main(
    features_path: Path,
    output_dir: Path,
    matched_k: int,
    cv_folds: int,
    outer_folds: int,
    inner_folds: int,
    n_jobs: int,
    metric_boot: int,
    delta_boot: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path)
    df["style"] = df["style"].astype(str)
    if "artist" not in df:
        df["artist"] = ""

    datasets = {
        "artbench10_all": df,
        "artbench10_wikiart8": df[~df["style"].isin(SOURCE_SPECIFIC_STYLES)].reset_index(drop=True),
    }

    all_off, all_off_d, all_grp, all_grp_d, all_folds = [], [], [], [], []
    for name, sub in datasets.items():
        print(f"\n=== {name}: {len(sub)} images, {sub['style'].nunique()} styles ===")
        out = run_dataset(
            sub, name, output_dir, matched_k, cv_folds, outer_folds, inner_folds,
            n_jobs, metric_boot, delta_boot
        )
        for target, item in zip([all_off, all_off_d, all_grp, all_grp_d, all_folds], out):
            if not item.empty:
                target.append(item)

    def save(parts, filename):
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(output_dir / filename, index=False)

    save(all_off, "artbench_official_results.csv")
    save(all_off_d, "artbench_official_deltas.csv")
    save(all_grp, "artbench_artist_disjoint_results.csv")
    save(all_grp_d, "artbench_artist_disjoint_deltas.csv")
    save(all_folds, "artbench_artist_disjoint_fold_results.csv")

    pd.DataFrame(
        [
            {
                "n_images": len(df),
                "n_styles": df["style"].nunique(),
                "artist_metadata_coverage": float(df["artist"].fillna("").astype(str).str.strip().ne("").mean()),
                "n_unique_artists": int(df.loc[df["artist"].fillna("").astype(str).str.strip().ne(""), "artist"].nunique()),
                "matched_k": matched_k,
                "cv_folds_official": cv_folds,
                "outer_folds_artist_disjoint": outer_folds,
                "inner_folds_artist_disjoint": inner_folds,
                "wikiart8_excludes": ",".join(sorted(SOURCE_SPECIFIC_STYLES)),
                "metric_bootstrap": metric_boot,
                "delta_bootstrap": delta_boot,
            }
        ]
    ).to_csv(output_dir / "artbench_phase4_metadata.csv", index=False)

    if all_grp:
        show = pd.concat(all_grp, ignore_index=True)
        print("\n=== ARTIST-DISJOINT SUMMARY ===")
        print(show[["dataset", "experiment", "n_images", "n_artists", "macro_f1_oof", "macro_f1_group_boot_ci_low", "macro_f1_group_boot_ci_high"]].to_string(index=False))
    if all_grp_d:
        print("\n=== ARTIST-DISJOINT DELTAS ===")
        print(pd.concat(all_grp_d, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase-IV ArtBench style generalization experiments.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase4_artbench"))
    p.add_argument("--matched-k", type=int, default=40)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--metric-bootstrap", type=int, default=1000)
    p.add_argument("--delta-bootstrap", type=int, default=2000)
    args = p.parse_args()
    main(
        args.features, args.output_dir, args.matched_k, args.cv_folds,
        args.outer_folds, args.inner_folds, args.n_jobs,
        args.metric_bootstrap, args.delta_bootstrap,
    )
