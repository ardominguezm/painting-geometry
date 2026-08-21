from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    cols = []
    for p in prefixes:
        cols.extend([c for c in df.columns if c.startswith(p)])
    return sorted(set(cols))


def build_model() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10.0, gamma='scale')),
    ])


def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]):
    X_train = train_df[cols]
    y_train = train_df['artist']
    X_test = test_df[cols]
    y_test = test_df['artist']

    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro = f1_score(y_test, pred, average='macro')
    ci_lo, ci_hi = bootstrap_ci(
        y_test.to_numpy(), pred,
        lambda a, b: f1_score(a, b, average='macro', zero_division=0),
    )
    return {
        'n_features': len(cols),
        'accuracy': float(acc),
        'macro_f1': float(macro),
        'macro_f1_ci_low': ci_lo,
        'macro_f1_ci_high': ci_hi,
    }


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
    for name, prefixes in experiments.items():
        cols = feature_columns(train_df, prefixes)
        if not cols:
            raise RuntimeError(f'No columns found for {name}: {prefixes}')
        result = evaluate(train_df, test_df, cols)
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
    print('\nSaved:', output_csv)
    print(out.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run edge/texture/geometry ablation experiments.')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/ablation_results.csv'))
    args = parser.parse_args()
    main(args.train, args.test, args.output)
