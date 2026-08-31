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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

EXCLUDE = {'surrealism', 'ukiyo_e'}
FIXED_C = 1.0
TOL = 1e-3
MAX_ITER = 5000


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = p[o]
    q = r * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    z = np.empty_like(q)
    z[o] = np.clip(q, 0, 1)
    return z


def scols(df, s):
    t = f'_s{float(s):.1f}'.replace('.', 'p')
    return [c for c in df if c.startswith('geom__curv__') and t in c]


def reps(df):
    B = [c for c in df if c.startswith('base__')]
    K = [c for c in df if c.startswith('geom__curv__')]
    O = [c for c in df if c.startswith('geom__orient__')]
    OP = [c for c in df if c.startswith('ord75__')]
    S = {s: scols(df, s) for s in [1, 2, 4, 8]}
    R = {
        'B90': B,
        'K40': K,
        'G44': K + O,
        'B90_K40': B + K,
        'B90_G44': B + K + O,
        'OP75': OP,
        'OP75_K40': OP + K,
        'B90_OP75': B + OP,
        'B90_OP75_K40': B + OP + K,
        'K_s1': S[1],
        'K_s2': S[2],
        'K_s4': S[4],
        'K_s8': S[8],
        'K_fine_s1_s2': S[1] + S[2],
        'K_coarse_s4_s8': S[4] + S[8],
    }
    return {k: v for k, v in R.items() if v}


def pipe():
    # n_samples >> n_features in every frozen representation, so the primal
    # liblinear problem is the appropriate deterministic formulation.
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('clf', LinearSVC(
            C=FIXED_C,
            dual=False,
            tol=TOL,
            max_iter=MAX_ITER,
            random_state=42,
        )),
    ])


def _atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def macro_f1_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(axis=-1)


def artist_bootstrap_delta(y, pred_new, pred_ref, groups, n=5000, seed=42, batch=250):
    """Exact artist-cluster bootstrap using per-artist confusion matrices.

    This is mathematically equivalent to resampling artists and concatenating
    their paintings, but avoids rebuilding ~60k-row arrays 5000 times.
    """
    y = np.asarray(y, str)
    pred_new = np.asarray(pred_new, str)
    pred_ref = np.asarray(pred_ref, str)
    groups = np.asarray(groups, str)
    labels = np.unique(np.concatenate([y, pred_new, pred_ref]))
    li = {lab: i for i, lab in enumerate(labels)}
    yi = np.array([li[x] for x in y], dtype=np.int16)
    ai = np.array([li[x] for x in pred_new], dtype=np.int16)
    bi = np.array([li[x] for x in pred_ref], dtype=np.int16)

    artists, gi = np.unique(groups, return_inverse=True)
    na, nc = len(artists), len(labels)
    cms_a = np.zeros((na, nc, nc), dtype=np.int32)
    cms_b = np.zeros((na, nc, nc), dtype=np.int32)
    np.add.at(cms_a, (gi, yi, ai), 1)
    np.add.at(cms_b, (gi, yi, bi), 1)
    flat_a = cms_a.reshape(na, -1)
    flat_b = cms_b.reshape(na, -1)

    obs = float(f1_score(y, pred_new, average='macro') - f1_score(y, pred_ref, average='macro'))
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    probs = np.full(na, 1.0 / na)
    pos = 0
    while pos < n:
        m = min(batch, n - pos)
        counts = rng.multinomial(na, probs, size=m)
        ca = (counts @ flat_a).reshape(m, nc, nc)
        cb = (counts @ flat_b).reshape(m, nc, nc)
        vals[pos:pos + m] = macro_f1_from_cm(ca) - macro_f1_from_cm(cb)
        pos += m
    lo, hi = np.quantile(vals, [.025, .975])
    p = (1 + int((vals <= 0).sum())) / (n + 1)
    return obs, float(lo), float(hi), float(p)


def run(df, label, out, outer=5, nboot=5000):
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / '_representation_checkpoints'
    ckpt.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d['artist'] = d['artist'].fillna('').astype(str).str.strip()
    d = d[d['artist'].ne('')].reset_index(drop=True)
    y = d['style'].astype(str).to_numpy()
    g = d['artist'].astype(str).to_numpy()
    R = reps(d)

    sp = StratifiedGroupKFold(n_splits=outer, shuffle=True, random_state=20260829)
    splits = list(sp.split(np.zeros((len(d), 1)), y, groups=g))

    preds = d[['split', 'style', 'artist', 'filename']].copy()
    common_outer_fold = np.full(len(d), -1, dtype=np.int16)
    for k, (_, te) in enumerate(splits):
        common_outer_fold[te] = k
    preds['outer_fold'] = common_outer_fold

    rows, fr = [], []
    for name, cols in R.items():
        print(f'{label} {name} {len(cols)}', flush=True)
        pred_path = ckpt / f'{name}.npz'
        foldmeta_path = ckpt / f'{name}.folds.json'
        P = np.full(len(d), '', dtype='<U64')
        F = np.full(len(d), -1, dtype=np.int16)
        fold_rows = []

        if pred_path.exists():
            try:
                z = np.load(pred_path, allow_pickle=False)
                p0 = z['pred'].astype(str)
                f0 = z['fold'].astype(np.int16)
                if len(p0) == len(d) and len(f0) == len(d):
                    P, F = p0, f0
                    print(f'  resume checkpoint: {(F >= 0).sum()}/{len(d)} predictions', flush=True)
            except Exception as exc:
                print('  checkpoint warning:', repr(exc), flush=True)
        if foldmeta_path.exists():
            try:
                fold_rows = json.loads(foldmeta_path.read_text(encoding='utf-8'))
            except Exception:
                fold_rows = []

        X = d[cols].to_numpy(np.float32)
        for k, (tr, te) in enumerate(splits):
            if np.all(F[te] == k) and np.all(P[te] != ''):
                print(f'  fold {k}: resume ✓', flush=True)
                continue

            print(f'  fold {k}: fitting fixed C={FIXED_C} ...', flush=True)
            t0 = time.perf_counter()
            m = pipe()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                m.fit(X[tr], y[tr])
            elapsed = time.perf_counter() - t0
            p = m.predict(X[te]).astype(str)
            P[te] = p
            F[te] = k
            clf = m.named_steps['clf']
            conv = any(issubclass(w.category, ConvergenceWarning) for w in caught)

            fold_rows = [r for r in fold_rows if int(r.get('fold', -1)) != k]
            fold_rows.append(dict(
                dataset=label,
                representation=name,
                fold=k,
                C=FIXED_C,
                n_train=len(tr),
                n_test=len(te),
                macro_f1=float(f1_score(y[te], p, average='macro')),
                accuracy=float(accuracy_score(y[te], p)),
                fit_seconds=float(elapsed),
                n_iter=int(np.max(np.atleast_1d(clf.n_iter_))),
                convergence_warning=bool(conv),
            ))
            fold_rows = sorted(fold_rows, key=lambda r: int(r['fold']))
            _atomic_npz(pred_path, pred=P, fold=F)
            _atomic_json(foldmeta_path, fold_rows)
            pd.DataFrame(fold_rows).to_csv(ckpt / f'{name}.folds.partial.csv', index=False)
            print(f'  fold {k}: checkpoint ✓ ({elapsed:.1f}s, n_iter={fold_rows[-1]["n_iter"]}, conv_warn={conv})', flush=True)

        if np.any(F < 0) or np.any(P == ''):
            raise RuntimeError(f'Incomplete prediction checkpoint for {label}/{name}')

        preds['pred__' + name] = P
        fr.extend(fold_rows)
        row = dict(
            dataset=label,
            representation=name,
            n_features=len(cols),
            n_images=len(d),
            n_artists=int(d['artist'].nunique()),
            C=FIXED_C,
            macro_f1=float(f1_score(y, P, average='macro')),
            accuracy=float(accuracy_score(y, P)),
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'linear_probe_results.partial.csv', index=False)
        pd.DataFrame(fr).to_csv(out / 'linear_probe_fold_results.partial.csv', index=False)
        print(f'  representation complete ✓ macro-F1={row["macro_f1"]:.6f}', flush=True)

    res = pd.DataFrame(rows)
    # K_all is exactly the same 40 curvature descriptors as K40; report the
    # alias without refitting so the scale table remains explicit.
    if (res['representation'] == 'K40').any():
        alias = res[res['representation'] == 'K40'].iloc[0].copy()
        alias['representation'] = 'K_all'
        res = pd.concat([res, pd.DataFrame([alias])], ignore_index=True)
        preds['pred__K_all'] = preds['pred__K40']

    res.to_csv(out / 'linear_probe_results.csv', index=False)
    pd.DataFrame(fr).to_csv(out / 'linear_probe_fold_results.csv', index=False)
    preds.to_csv(out / 'linear_probe_predictions.csv', index=False)

    H = [
        ('B90_K40', 'B90', 'H1 curvature beyond appearance'),
        ('B90_G44', 'B90', 'H1b full geometry beyond appearance'),
        ('OP75_K40', 'OP75', 'H2 curvature beyond ordinal'),
        ('B90_OP75_K40', 'B90_OP75', 'H3 curvature beyond appearance+ordinal'),
    ]
    dr = []
    for i, (a, b, h) in enumerate(H):
        print(f'  bootstrap {h} ...', flush=True)
        z = artist_bootstrap_delta(
            y,
            preds['pred__' + a].to_numpy(),
            preds['pred__' + b].to_numpy(),
            g,
            n=nboot,
            seed=42 + i,
        )
        dr.append(dict(
            dataset=label,
            hypothesis=h,
            new_model=a,
            reference=b,
            delta_macro_f1=z[0],
            ci_low=z[1],
            ci_high=z[2],
            p_one_sided=z[3],
            n_boot=nboot,
        ))
        pd.DataFrame(dr).to_csv(out / 'confirmatory_deltas.partial.csv', index=False)

    D = pd.DataFrame(dr)
    D['q_bh'] = bh(D['p_one_sided'].to_numpy())
    D.to_csv(out / 'confirmatory_deltas.csv', index=False)

    scale = ['K_s1', 'K_s2', 'K_s4', 'K_s8', 'K_fine_s1_s2', 'K_coarse_s4_s8', 'K_all']
    res[res['representation'].isin(scale)].to_csv(out / 'scale_probe_results.csv', index=False)
    return res, D


def main(inp, out, outer, nboot):
    print('Reading frozen 60k feature matrix...', flush=True)
    df = pd.read_csv(inp)
    print('Feature matrix:', df.shape, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    allr, alld = [], []
    datasets = [
        (df, 'artbench10_all'),
        (df[~df['style'].astype(str).isin(EXCLUDE)].copy(), 'artbench10_wikiart8'),
    ]
    for d, label in datasets:
        r, z = run(d, label, out / label, outer=outer, nboot=nboot)
        allr.append(r)
        alld.append(z)
        pd.concat(allr, ignore_index=True).to_csv(out / 'phase7_fixed_results.partial.csv', index=False)
        pd.concat(alld, ignore_index=True).to_csv(out / 'phase7_fixed_deltas.partial.csv', index=False)

    pd.concat(allr, ignore_index=True).to_csv(out / 'phase7_fixed_results_all.csv', index=False)
    pd.concat(alld, ignore_index=True).to_csv(out / 'phase7_fixed_deltas_all.csv', index=False)
    (out / 'analysis_plan.json').write_text(json.dumps({
        'classifier': 'SimpleImputer(median) + StandardScaler + LinearSVC',
        'regularization': {'C': FIXED_C, 'selection': 'common fixed value; no inner hyperparameter tuning'},
        'solver': {'dual': False, 'tol': TOL, 'max_iter': MAX_ITER},
        'outer_folds': outer,
        'n_boot': nboot,
        'group': 'artist',
        'metric': 'macro-F1',
        'checkpointing': 'per representation and outer fold',
        'bootstrap': 'artist-cluster bootstrap via equivalent per-artist confusion-matrix aggregation',
    }, indent=2), encoding='utf-8')
    print('\nPHASE VII FIXED LINEAR PROBE COMPLETE ✓', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--n-boot', type=int, default=5000)
    a = p.parse_args()
    main(a.features, a.output_dir, a.outer_folds, a.n_boot)
