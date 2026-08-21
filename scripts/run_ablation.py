from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for p in prefixes:
        cols.extend([c for c in df.columns if c.startswith(p)])
    return sorted(set(cols))


def build_model() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10.0, gamma='scale')),
    ])


def stratified_bootstrap_indices(y_true, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap within each class so macro-F1 resamples preserve class support."""
    y = np.asarray(y_true)
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y_true, rng)
        vals.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_delta(
    y_true,
    pred_new,
    pred_reference,
    n_boot: int = 5000,
    seed: int = 123,
):
    """Paired bootstrap CI for Macro-F1(new) - Macro-F1(reference)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = np.asarray(pred_new)
    b = np.asarray(pred_reference)
    deltas = []
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        fa = f1_score(y[idx], a[idx], average='macro', zero_division=0)
        fb = f1_score(y[idx], b[idx], average='macro', zero_division=0)
        deltas.append(fa - fb)
    deltas = np.asarray(deltas)
    return {
        'delta_macro_f1': float(
            f1_score(y, a, average='macro', zero_division=0)
            - f1_score(y, b, average='macro', zero_division=0)
        ),
        'delta_ci_low': float(np.percentile(deltas, 2.5)),
        'delta_ci_high': float(np.percentile(deltas, 97.5)),
        'bootstrap_p_improvement': float(np.mean(deltas <= 0.0)),
        'n_boot': int(n_boot),
    }


def evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]):
    X_train = train_df[cols]
    y_train = train_df['artist']
    X_test = test_df[cols]
    y_test = test_df['artist']

    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro = f1_score(y_test, pred, average='macro', zero_division=0)
    ci_lo, ci_hi = bootstrap_ci(
        y_test.to_numpy(),
        pred,
        lambda a, b: f1_score(a, b, average='macro', zero_division=0),
    )
    metrics = {
        'n_features': len(cols),
        'accuracy': float(acc),
        'macro_f1': float(macro),
        'macro_f1_ci_low': ci_lo,
        'macro_f1_ci_high': ci_hi,
    }
    return metrics, pred


def main(train_csv: Path, test_csv: Path, output_csv: Path):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    experiments = {
        'E1_edges': ('edge__',),
        'E2_texture': ('texture__',),
        'E3_curvature': ('curv__',),
        'E4_geometry': ('curv__', 'orient__'),
        'E5_baseline': ('edge__', 'texture__'),
        'E6_combined': ('edge__', 'texture__', 'curv__', 'orient__'),
    }

    rows = []
    predictions: dict[str, np.ndarray] = {}
    for name, prefixes in experiments.items():
        cols = feature_columns(train_df, prefixes)
        if not cols:
            raise RuntimeError(f'No columns found for {name}: {prefixes}')
        result, pred = evaluate(train_df, test_df, cols)
        predictions[name] = pred
        result['experiment'] = name
        result['prefixes'] = '+'.join(prefixes)
        rows.append(result)
        print(name, result)

    out = pd.DataFrame(rows)[[
        'experiment', 'prefixes', 'n_features', 'accuracy', 'macro_f1',
        'macro_f1_ci_low', 'macro_f1_ci_high'
    ]]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    pred_df = test_df[['artist', 'filename']].copy()
    for name, pred in predictions.items():
        pred_df[name] = pred
    pred_path = output_csv.with_name('ablation_predictions.csv')
    pred_df.to_csv(pred_path, index=False)

    delta = paired_bootstrap_delta(
        test_df['artist'].to_numpy(),
        predictions['E6_combined'],
        predictions['E5_baseline'],
    )
    delta.update({'new_model': 'E6_combined', 'reference': 'E5_baseline'})
    delta_path = output_csv.with_name('ablation_delta.csv')
    pd.DataFrame([delta]).to_csv(delta_path, index=False)

    print('\nSaved:', output_csv)
    print(out.to_string(index=False))
    print('\nPaired improvement E6 - E5:')
    print(pd.DataFrame([delta]).to_string(index=False))
    print('\nPredictions:', pred_path)
    print('Delta:', delta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run edge/texture/geometry ablation experiments.')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/ablation_results.csv'))
    args = parser.parse_args()
    main(args.train, args.test, args.output)
