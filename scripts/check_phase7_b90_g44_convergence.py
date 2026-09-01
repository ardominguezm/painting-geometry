from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def b90_g44_columns(df: pd.DataFrame) -> list[str]:
    base = [c for c in df.columns if c.startswith('base__')]
    curv = [c for c in df.columns if c.startswith('geom__curv__')]
    orient = [c for c in df.columns if c.startswith('geom__orient__')]
    cols = base + curv + orient
    if len(base) != 90 or len(curv) != 40 or len(orient) != 4 or len(cols) != 134:
        raise RuntimeError(
            f'Unexpected B90/G44 dimensions: B={len(base)}, K={len(curv)}, O={len(orient)}, total={len(cols)}'
        )
    return cols


def build_model(max_iter: int) -> Pipeline:
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('clf', LinearSVC(
            C=1.0,
            dual=False,
            tol=1e-3,
            max_iter=max_iter,
            random_state=42,
        )),
    ])


def main(features: Path, checkpoint: Path, output_dir: Path, fold: int, max_iter: int) -> None:
    print('Reading frozen feature matrix...', flush=True)
    df = pd.read_csv(features)
    df['artist'] = df['artist'].fillna('').astype(str).str.strip()
    d = df[df['artist'].ne('')].reset_index(drop=True)
    print('Usable artist-linked rows:', len(d), 'artists:', d['artist'].nunique(), flush=True)

    y = d['style'].astype(str).to_numpy()
    g = d['artist'].astype(str).to_numpy()
    cols = b90_g44_columns(d)
    X = d[cols].to_numpy(np.float32)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260829)
    splits = list(splitter.split(np.zeros((len(d), 1)), y, groups=g))
    tr, te = splits[fold]

    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing original checkpoint: {checkpoint}')
    z = np.load(checkpoint, allow_pickle=False)
    old_pred = z['pred'].astype(str)
    old_fold = z['fold'].astype(np.int16)
    if len(old_pred) != len(d) or len(old_fold) != len(d):
        raise RuntimeError('Checkpoint length does not match current filtered dataset.')
    if not np.all(old_fold[te] == fold):
        raise RuntimeError(f'Original checkpoint does not contain complete fold {fold} predictions.')

    p_old = old_pred[te]
    y_te = y[te]
    old_f1 = float(f1_score(y_te, p_old, average='macro'))
    old_acc = float(accuracy_score(y_te, p_old))

    print(f'Refitting B90_G44 fold {fold} with max_iter={max_iter} ...', flush=True)
    model = build_model(max_iter)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X[tr], y[tr])
    elapsed = time.perf_counter() - t0
    p_new = model.predict(X[te]).astype(str)
    clf = model.named_steps['clf']
    conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    n_iter = int(np.max(np.atleast_1d(clf.n_iter_)))

    new_f1 = float(f1_score(y_te, p_new, average='macro'))
    new_acc = float(accuracy_score(y_te, p_new))
    n_changed = int(np.sum(p_new != p_old))
    agreement = float(np.mean(p_new == p_old))
    delta_f1 = new_f1 - old_f1
    delta_acc = new_acc - old_acc

    # The original confirmatory analysis used tol=1e-3 and max_iter=5000.
    # We regard the reported fold as numerically stable if the longer fit
    # converges and changes fold macro-F1 by <1e-3. Prediction agreement is
    # reported descriptively but is not itself used as the pass criterion.
    stable = (not conv_warn) and (abs(delta_f1) < 1e-3)

    result = {
        'dataset': 'artbench10_all',
        'representation': 'B90_G44',
        'fold': int(fold),
        'n_train': int(len(tr)),
        'n_test': int(len(te)),
        'C': 1.0,
        'dual': False,
        'tol': 1e-3,
        'original_max_iter': 5000,
        'refit_max_iter': int(max_iter),
        'refit_n_iter': n_iter,
        'refit_convergence_warning': bool(conv_warn),
        'fit_seconds': float(elapsed),
        'original_macro_f1': old_f1,
        'refit_macro_f1': new_f1,
        'delta_macro_f1': float(delta_f1),
        'original_accuracy': old_acc,
        'refit_accuracy': new_acc,
        'delta_accuracy': float(delta_acc),
        'n_prediction_changes': n_changed,
        'prediction_agreement': agreement,
        'stability_threshold_abs_delta_macro_f1': 1e-3,
        'numerically_stable': bool(stable),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(output_dir / 'b90_g44_fold1_convergence_sensitivity.csv', index=False)
    (output_dir / 'b90_g44_fold1_convergence_sensitivity.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8'
    )

    print('\nConvergence sensitivity result', flush=True)
    for k, v in result.items():
        print(f'  {k}: {v}', flush=True)
    print('\nVERDICT:', 'PASS ✓' if stable else 'REVIEW REQUIRED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--max-iter', type=int, default=20000)
    a = p.parse_args()
    main(a.features, a.checkpoint, a.output_dir, a.fold, a.max_iter)
