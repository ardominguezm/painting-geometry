from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def stratified_bootstrap_indices(y_true, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true)
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def macro_f1_ci(y_true, y_pred, n_boot=2000, seed=42):
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        vals[i] = f1_score(y[idx], pred[idx], average="macro", zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(y_true, pred_new, pred_ref, n_boot=5000, seed=123):
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


def build_search(cv_folds=3, n_jobs=-1):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", cache_size=2048)),
    ])
    grid = {
        "clf__C": [1.0, 3.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.03],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return GridSearchCV(
        pipe,
        param_grid=grid,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        verbose=0,
    )


def columns_for_scales(df: pd.DataFrame, scales: tuple[float, ...]) -> list[str]:
    cols = []
    for s in scales:
        tag = str(float(s)).replace(".", "p")
        prefix = f"curv__kappa_s{tag}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    return sorted(cols)


def main(train_path: Path, test_path: Path, output_dir: Path, cv_folds: int, n_jobs: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y_train = train["artist"].to_numpy()
    y_test = test["artist"].to_numpy()

    orient_cols = sorted(c for c in train.columns if c.startswith("orient__"))
    experiments = [
        ("S1", (1.0,), False),
        ("S2", (2.0,), False),
        ("S4", (4.0,), False),
        ("S8", (8.0,), False),
        ("S12", (1.0, 2.0), False),
        ("S24", (2.0, 4.0), False),
        ("S48", (4.0, 8.0), False),
        ("S124", (1.0, 2.0, 4.0), False),
        ("S248", (2.0, 4.0, 8.0), False),
        ("S1248", (1.0, 2.0, 4.0, 8.0), False),
        ("S1248_orient", (1.0, 2.0, 4.0, 8.0), True),
    ]

    rows = []
    pred_map = {}
    cv_scores = {}
    for name, scales, add_orient in experiments:
        cols = columns_for_scales(train, scales)
        if add_orient:
            cols = sorted(set(cols + orient_cols))
        if not cols:
            raise RuntimeError(f"No features found for {name}")
        search = build_search(cv_folds=cv_folds, n_jobs=n_jobs)
        search.fit(train[cols], y_train)
        pred = search.predict(test[cols])
        lo, hi = macro_f1_ci(y_test, pred)
        pred_map[name] = pred
        cv_scores[name] = float(search.best_score_)
        rows.append({
            "experiment": name,
            "scales": "+".join(str(int(s)) if float(s).is_integer() else str(s) for s in scales),
            "with_orientation": bool(add_orient),
            "n_features": len(cols),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "macro_f1_ci_low": lo,
            "macro_f1_ci_high": hi,
            "best_cv_macro_f1": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })
        print(name, rows[-1]["macro_f1"], search.best_params_)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "scale_ablation_results.csv", index=False)

    single_names = ["S1", "S2", "S4", "S8"]
    best_single = max(single_names, key=lambda x: cv_scores[x])
    comparisons = [
        ("S1248", best_single, "all_scales_vs_best_single_by_training_cv"),
        ("S1248_orient", "S1248", "orientation_increment"),
    ]
    delta_rows = []
    for new, ref, label in comparisons:
        d = paired_delta(y_test, pred_map[new], pred_map[ref])
        d.update({"comparison": label, "new_model": new, "reference": ref})
        delta_rows.append(d)
    pd.DataFrame(delta_rows).to_csv(output_dir / "scale_ablation_deltas.csv", index=False)

    pred_df = test[["artist", "filename"]].copy()
    for name, pred in pred_map.items():
        pred_df[name] = pred
    pred_df.to_csv(output_dir / "scale_ablation_predictions.csv", index=False)

    print("\nBest single scale by training CV:", best_single)
    print(results[["experiment", "n_features", "best_cv_macro_f1", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase III scale ablation using Phase-I geometry features.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3_scale"))
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    main(args.train, args.test, args.output_dir, args.cv_folds, args.n_jobs)
