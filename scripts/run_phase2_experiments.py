from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for prefix in prefixes:
        cols.extend(c for c in df.columns if c.startswith(prefix))
    return sorted(set(cols))


def merge_feature_tables(geometry: pd.DataFrame, strong: pd.DataFrame) -> pd.DataFrame:
    keys = ['artist', 'filename']
    keep = keys + [c for c in strong.columns if c.startswith('strong__')]
    if geometry.duplicated(keys).any() or strong.duplicated(keys).any():
        raise ValueError('Duplicate artist/filename keys detected before feature merge.')
    merged = geometry.merge(strong[keep], on=keys, how='inner', validate='one_to_one')
    if len(merged) != len(geometry):
        missing = len(geometry) - len(merged)
        raise ValueError(f'Strong baseline merge lost {missing} geometry rows.')
    return merged


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})


def build_clean_mask(
    test_df: pd.DataFrame,
    candidates_path: Path | None,
    phash_threshold: int,
    dhash_threshold: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    mask = np.ones(len(test_df), dtype=bool)
    if candidates_path is None or not candidates_path.exists():
        return mask, pd.DataFrame()

    candidates = pd.read_csv(candidates_path)
    if candidates.empty:
        return mask, candidates

    exact = _as_bool(candidates['exact_bytes'])
    flagged = candidates[
        exact
        | (
            (candidates['phash_distance'] <= phash_threshold)
            & (candidates['dhash_distance'] <= dhash_threshold)
        )
    ].copy()
    if flagged.empty:
        return mask, flagged

    flagged_keys = set(zip(flagged['test_artist'].astype(str), flagged['test_filename'].astype(str)))
    keys = list(zip(test_df['artist'].astype(str), test_df['filename'].astype(str)))
    mask = np.array([key not in flagged_keys for key in keys], dtype=bool)
    return mask, flagged


def stratified_bootstrap_indices(y_true, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true)
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def metric_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42):
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        vals.append(f1_score(y[idx], pred[idx], average='macro', zero_division=0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta(y_true, pred_new, pred_ref, n_boot: int = 5000, seed: int = 123):
    y = np.asarray(y_true)
    a = np.asarray(pred_new)
    b = np.asarray(pred_ref)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        deltas[i] = (
            f1_score(y[idx], a[idx], average='macro', zero_division=0)
            - f1_score(y[idx], b[idx], average='macro', zero_division=0)
        )
    observed = (
        f1_score(y, a, average='macro', zero_division=0)
        - f1_score(y, b, average='macro', zero_division=0)
    )
    return {
        'delta_macro_f1': float(observed),
        'delta_ci_low': float(np.percentile(deltas, 2.5)),
        'delta_ci_high': float(np.percentile(deltas, 97.5)),
        'bootstrap_p_improvement': float((1 + np.sum(deltas <= 0.0)) / (n_boot + 1)),
        'n_boot': int(n_boot),
    }


def build_search(k: int | None, cv_folds: int, random_state: int, n_jobs: int):
    selector = 'passthrough' if k is None else SelectKBest(score_func=f_classif, k=k)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', selector),
        ('clf', SVC(kernel='rbf', cache_size=2048)),
    ])
    grid = {
        'clf__C': [1.0, 3.0, 10.0],
        'clf__gamma': ['scale', 0.01, 0.03],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return GridSearchCV(
        pipe,
        param_grid=grid,
        scoring='f1_macro',
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False,
    )


def evaluate_predictions(y_true, pred, experiment: str, eval_set: str, n_features: int, selected_k, cv_score, best_params):
    ci_lo, ci_hi = metric_ci(y_true, pred)
    return {
        'experiment': experiment,
        'eval_set': eval_set,
        'n_input_features': int(n_features),
        'n_selected_features': int(selected_k if selected_k is not None else n_features),
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'macro_f1_ci_low': ci_lo,
        'macro_f1_ci_high': ci_hi,
        'best_cv_macro_f1': float(cv_score),
        'best_params': json.dumps(best_params, sort_keys=True),
    }


def main(
    geometry_train_path: Path,
    geometry_test_path: Path,
    strong_train_path: Path,
    strong_test_path: Path,
    output_dir: Path,
    leakage_candidates: Path | None,
    clean_phash: int,
    clean_dhash: int,
    matched_k: int,
    cv_folds: int,
    n_jobs: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_train = pd.read_csv(geometry_train_path)
    geometry_test = pd.read_csv(geometry_test_path)
    strong_train = pd.read_csv(strong_train_path)
    strong_test = pd.read_csv(strong_test_path)

    train = merge_feature_tables(geometry_train, strong_train)
    test = merge_feature_tables(geometry_test, strong_test)
    if sorted(train['artist'].unique()) != sorted(test['artist'].unique()):
        raise ValueError('Train/test artist supports do not match.')

    clean_mask, flagged = build_clean_mask(test, leakage_candidates, clean_phash, clean_dhash)
    flagged.to_csv(output_dir / 'flagged_test_leakage_pairs.csv', index=False)
    test.loc[~clean_mask, ['artist', 'filename']].to_csv(output_dir / 'excluded_test_images.csv', index=False)

    legacy_cols = feature_columns(train, ('edge__', 'texture__'))
    geometry_cols = feature_columns(train, ('curv__', 'orient__'))
    strong_cols = feature_columns(train, ('strong__',))
    combined_cols = sorted(set(strong_cols + geometry_cols))

    if not geometry_cols or not strong_cols:
        raise RuntimeError('Missing geometry or strong-baseline feature families.')

    k_strong = min(matched_k, len(strong_cols))
    k_geometry = min(matched_k, len(geometry_cols))
    k_combined = min(matched_k, len(combined_cols))

    experiments = {
        'L_legacy_compact': (legacy_cols, None),
        'B_strong_full': (strong_cols, None),
        'G_geometry_full': (geometry_cols, None),
        'BG_combined_full': (combined_cols, None),
        f'B_strong_k{k_strong}': (strong_cols, k_strong),
        f'G_geometry_k{k_geometry}': (geometry_cols, k_geometry),
        f'BG_combined_k{k_combined}': (combined_cols, k_combined),
    }

    y_train = train['artist'].to_numpy()
    y_test = test['artist'].to_numpy()
    rows = []
    predictions: dict[str, np.ndarray] = {}
    selected_rows = []

    for exp_index, (name, (cols, k)) in enumerate(experiments.items(), start=1):
        if not cols:
            print(f'Skipping {name}: no features')
            continue
        print(f'\n[{exp_index}/{len(experiments)}] {name}: {len(cols)} input features; k={k}')
        search = build_search(k=k, cv_folds=cv_folds, random_state=42, n_jobs=n_jobs)
        search.fit(train[cols], y_train)
        pred = search.predict(test[cols])
        predictions[name] = pred
        print('Best CV macro-F1:', search.best_score_)
        print('Best params:', search.best_params_)

        rows.append(evaluate_predictions(
            y_test, pred, name, 'raw', len(cols), k, search.best_score_, search.best_params_
        ))
        if clean_mask.sum() > 0:
            rows.append(evaluate_predictions(
                y_test[clean_mask], pred[clean_mask], name, 'clean', len(cols), k,
                search.best_score_, search.best_params_
            ))

        if k is not None:
            selector = search.best_estimator_.named_steps['selector']
            support = selector.get_support()
            scores = getattr(selector, 'scores_', np.full(len(cols), np.nan))
            for feature, keep, score in zip(cols, support, scores):
                if keep:
                    selected_rows.append({
                        'experiment': name,
                        'feature': feature,
                        'anova_f': float(score) if np.isfinite(score) else np.nan,
                    })

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / 'phase2_results.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / 'phase2_selected_features.csv', index=False)

    pred_df = test[['artist', 'filename']].copy()
    pred_df['clean_eval'] = clean_mask
    for name, pred in predictions.items():
        pred_df[name] = pred
    pred_df.to_csv(output_dir / 'phase2_predictions.csv', index=False)

    comparisons = [
        ('BG_combined_full', 'B_strong_full'),
        (f'BG_combined_k{k_combined}', f'B_strong_k{k_strong}'),
        ('G_geometry_full', 'B_strong_full'),
    ]
    delta_rows = []
    for new_name, ref_name in comparisons:
        if new_name not in predictions or ref_name not in predictions:
            continue
        for eval_name, mask in [('raw', np.ones(len(test), dtype=bool)), ('clean', clean_mask)]:
            if mask.sum() == 0:
                continue
            delta = paired_delta(y_test[mask], predictions[new_name][mask], predictions[ref_name][mask])
            delta.update({
                'eval_set': eval_name,
                'new_model': new_name,
                'reference': ref_name,
                'n_test': int(mask.sum()),
            })
            delta_rows.append(delta)
    pd.DataFrame(delta_rows).to_csv(output_dir / 'phase2_deltas.csv', index=False)

    per_artist_rows = []
    for name, pred in predictions.items():
        for eval_name, mask in [('raw', np.ones(len(test), dtype=bool)), ('clean', clean_mask)]:
            yy = y_test[mask]
            pp = pred[mask]
            for artist in np.unique(yy):
                class_f1 = f1_score(
                    yy,
                    pp,
                    labels=[artist],
                    average=None,
                    zero_division=0,
                )[0]
                per_artist_rows.append({
                    'experiment': name,
                    'eval_set': eval_name,
                    'artist': artist,
                    'n': int(np.sum(yy == artist)),
                    'f1': float(class_f1),
                })
    pd.DataFrame(per_artist_rows).to_csv(output_dir / 'phase2_per_artist_f1.csv', index=False)

    metadata = pd.DataFrame([{
        'n_train': len(train),
        'n_test_raw': len(test),
        'n_test_clean': int(clean_mask.sum()),
        'n_excluded_for_leakage': int((~clean_mask).sum()),
        'legacy_features': len(legacy_cols),
        'strong_features': len(strong_cols),
        'geometry_features': len(geometry_cols),
        'combined_features': len(combined_cols),
        'matched_k_requested': matched_k,
        'cv_folds': cv_folds,
        'clean_phash_threshold': clean_phash,
        'clean_dhash_threshold': clean_dhash,
    }])
    metadata.to_csv(output_dir / 'phase2_metadata.csv', index=False)

    print('\n=== PHASE II RESULTS ===')
    print(results[['experiment', 'eval_set', 'n_input_features', 'n_selected_features', 'macro_f1', 'macro_f1_ci_low', 'macro_f1_ci_high', 'best_cv_macro_f1']].to_string(index=False))
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        print('\n=== PAIRED DELTAS ===')
        print(delta_df.to_string(index=False))
    print('\n=== METADATA ===')
    print(metadata.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Phase II leakage-aware strong-baseline experiments.')
    parser.add_argument('--geometry-train', type=Path, required=True)
    parser.add_argument('--geometry-test', type=Path, required=True)
    parser.add_argument('--strong-train', type=Path, required=True)
    parser.add_argument('--strong-test', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/phase2'))
    parser.add_argument('--leakage-candidates', type=Path, default=None)
    parser.add_argument('--clean-phash', type=int, default=4)
    parser.add_argument('--clean-dhash', type=int, default=4)
    parser.add_argument('--matched-k', type=int, default=40)
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()
    main(
        args.geometry_train,
        args.geometry_test,
        args.strong_train,
        args.strong_test,
        args.output_dir,
        args.leakage_candidates,
        args.clean_phash,
        args.clean_dhash,
        args.matched_k,
        args.cv_folds,
        args.n_jobs,
    )
