from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.curvature_v2 import relative_scale_curvature_features
from src.orientation import structure_tensor_features
from src.preprocessing import preprocess


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def discover_by_artist(root: Path) -> dict[str, list[Path]]:
    out = {}
    for artist_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = sorted(
            p for p in artist_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if paths:
            out[artist_dir.name] = paths
    if not out:
        raise RuntimeError(f"No artist folders/images found under {root}")
    return out


def sample_manifest(root: Path, per_artist: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = discover_by_artist(root)
    rows = []
    for artist, paths in groups.items():
        n = min(int(per_artist), len(paths))
        chosen_idx = np.sort(rng.choice(len(paths), size=n, replace=False))
        for i in chosen_idx:
            path = paths[int(i)]
            rows.append({
                "artist": artist,
                "filename": path.name,
                "path": str(path),
            })
    return pd.DataFrame(rows)


def extract_one(path: Path, long_side: int, sigma_refs, reference_long_side: int):
    _, I = preprocess(path, long_side=long_side)
    curv = relative_scale_curvature_features(
        I,
        long_side=long_side,
        sigma_refs=sigma_refs,
        reference_long_side=reference_long_side,
        return_maps=False,
    )

    orient_sigma_px = 2.0 * float(long_side) / float(reference_long_side)
    orient = structure_tensor_features(I, sigma=orient_sigma_px)

    out = {f"v2__curv__{k}": v for k, v in curv.items()}
    out.update({f"v2__orient__{k}": v for k, v in orient.items()})
    return out


def icc3_1(matrix: np.ndarray) -> float:
    """ICC(3,1): two-way mixed, single measurement, consistency."""
    X = np.asarray(matrix, dtype=float)
    keep = np.all(np.isfinite(X), axis=1)
    X = X[keep]
    n, k = X.shape if X.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return np.nan

    grand = np.mean(X)
    row_mean = np.mean(X, axis=1)
    col_mean = np.mean(X, axis=0)

    ms_subject = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    residual = X - row_mean[:, None] - col_mean[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denom = ms_subject + (k - 1) * ms_error
    if denom <= 0:
        return np.nan
    return float((ms_subject - ms_error) / denom)


def matched_matrix(df: pd.DataFrame, feature: str, resolutions: list[int]) -> tuple[pd.DataFrame, list[int]]:
    piv = df.pivot_table(
        index=["artist", "filename"],
        columns="resolution",
        values=feature,
        aggfunc="first",
    )
    available = [r for r in resolutions if r in piv.columns]
    piv = piv[available].dropna()
    return piv, available


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    q25, q75 = np.percentile(x, [25, 75])
    s = (q75 - q25) / 1.349
    if s <= 1e-12:
        s = np.std(x, ddof=1)
    return float(s) if s > 1e-12 else np.nan


def summarize_stability(df: pd.DataFrame, resolutions: list[int], reference_resolution: int) -> pd.DataFrame:
    features = sorted(c for c in df.columns if c.startswith("v2__"))
    rows = []
    for feature in features:
        piv, available = matched_matrix(df, feature, resolutions)
        row = {
            "feature": feature,
            "n_complete": len(piv),
            "icc3_1": icc3_1(piv.to_numpy()) if len(available) >= 2 else np.nan,
        }

        for a, b in [(256, 512), (512, 1024), (256, 1024)]:
            key = f"spearman_{a}_{b}"
            if a in piv.columns and b in piv.columns and len(piv) >= 3:
                rho, p = spearmanr(piv[a], piv[b], nan_policy="omit")
                row[key] = float(rho)
                row[f"{key}_p"] = float(p)
            else:
                row[key] = np.nan
                row[f"{key}_p"] = np.nan

        if reference_resolution in piv.columns:
            scale = robust_scale(piv[reference_resolution].to_numpy())
            for r in available:
                if r == reference_resolution:
                    continue
                if np.isfinite(scale) and scale > 0:
                    drift = np.median(np.abs(piv[r] - piv[reference_resolution])) / scale
                    row[f"median_abs_drift_iqrscale_{r}_vs_{reference_resolution}"] = float(drift)
                else:
                    row[f"median_abs_drift_iqrscale_{r}_vs_{reference_resolution}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(
    root: Path,
    output_dir: Path,
    per_artist: int,
    seed: int,
    resolutions: list[int],
    sigma_refs: list[float],
    reference_long_side: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = sample_manifest(root, per_artist=per_artist, seed=seed)
    manifest.to_csv(output_dir / "resolution_sample_manifest.csv", index=False)

    rows = []
    total = len(manifest) * len(resolutions)
    with tqdm(total=total, desc="Phase III resolution robustness", dynamic_ncols=True) as bar:
        for rec in manifest.itertuples(index=False):
            for resolution in resolutions:
                row = {
                    "artist": rec.artist,
                    "filename": rec.filename,
                    "path": rec.path,
                    "resolution": int(resolution),
                }
                try:
                    row.update(
                        extract_one(
                            Path(rec.path),
                            long_side=int(resolution),
                            sigma_refs=sigma_refs,
                            reference_long_side=reference_long_side,
                        )
                    )
                    row["error"] = None
                except Exception as exc:
                    row["error"] = repr(exc)
                rows.append(row)
                bar.update(1)

    features = pd.DataFrame(rows)
    features.to_csv(output_dir / "resolution_features_v2.csv", index=False)

    ok = features[features["error"].isna()].copy()
    summary = summarize_stability(
        ok,
        resolutions=[int(r) for r in resolutions],
        reference_resolution=int(reference_long_side),
    )
    summary.to_csv(output_dir / "resolution_robustness_summary.csv", index=False)

    metadata = pd.DataFrame([{
        "n_manifest_images": len(manifest),
        "per_artist_requested": per_artist,
        "seed": seed,
        "resolutions": ",".join(map(str, resolutions)),
        "sigma_refs_at_512": ",".join(map(str, sigma_refs)),
        "reference_long_side": reference_long_side,
        "curvature_definition": "true derivative-of-Gaussian; sigma_px*kappa; relative sigma matched across resolutions",
        "grad_quantile_mask": 0.20,
    }])
    metadata.to_csv(output_dir / "resolution_robustness_metadata.csv", index=False)

    key_cols = [
        c for c in [
            "feature", "n_complete", "icc3_1",
            "spearman_256_512", "spearman_512_1024", "spearman_256_1024"
        ]
        if c in summary.columns
    ]
    print("\nResolution robustness summary:")
    print(summary[key_cols].sort_values("icc3_1", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Resolution robustness using scale-normalized level-set curvature.")
    p.add_argument("--root", type=Path, required=True, help="One-folder-per-artist corpus root.")
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3_resolution"))
    p.add_argument("--per-artist", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolutions", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--sigma-refs", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--reference-long-side", type=int, default=512)
    args = p.parse_args()
    main(
        args.root,
        args.output_dir,
        args.per_artist,
        args.seed,
        args.resolutions,
        args.sigma_refs,
        args.reference_long_side,
    )
