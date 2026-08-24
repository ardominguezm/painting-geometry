from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.curvature import multiscale_curvature_features
from src.orientation import structure_tensor_features
from src.preprocessing import preprocess


def exclusion_keys(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    split_col = "split" if "split" in df.columns else None
    keys = set()
    for row in df.itertuples(index=False):
        split = str(getattr(row, split_col)) if split_col is not None else "any"
        keys.add((split, str(getattr(row, "artist")), str(getattr(row, "filename"))))
    return keys


def load_reference(
    train_path: Path,
    test_path: Path,
    phase2_excluded_path: Path | None,
    starry_excluded_path: Path | None,
    artist: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    train = pd.read_csv(train_path).copy()
    test = pd.read_csv(test_path).copy()
    train["split"] = "train"
    test["split"] = "test"

    counts = {
        "train_initial": len(train),
        "test_initial": len(test),
        "phase2_test_exclusions": 0,
        "starry_reference_exclusions": 0,
    }

    if phase2_excluded_path is not None and phase2_excluded_path.exists():
        excluded = pd.read_csv(phase2_excluded_path)
        if not excluded.empty:
            keys = set(zip(excluded["artist"].astype(str), excluded["filename"].astype(str)))
            mask = np.array([
                (str(a), str(f)) not in keys
                for a, f in zip(test["artist"], test["filename"])
            ])
            counts["phase2_test_exclusions"] = int((~mask).sum())
            test = test.loc[mask].copy()

    combined = pd.concat([train, test], ignore_index=True)
    combined = combined[combined["artist"].astype(str) == str(artist)].copy()

    starry_keys = exclusion_keys(starry_excluded_path)
    if starry_keys:
        mask = []
        for row in combined[["split", "artist", "filename"]].itertuples(index=False):
            k_exact = (str(row.split), str(row.artist), str(row.filename))
            k_any = ("any", str(row.artist), str(row.filename))
            mask.append(k_exact not in starry_keys and k_any not in starry_keys)
        mask = np.asarray(mask, dtype=bool)
        counts["starry_reference_exclusions"] = int((~mask).sum())
        combined = combined.loc[mask].copy()

    if combined.empty:
        raise RuntimeError(f"No reference rows found for artist={artist!r}")
    return combined.reset_index(drop=True), counts


def extract_legacy_geometry(image_path: Path, long_side: int = 512) -> dict[str, float]:
    _, I = preprocess(image_path, long_side=long_side)
    curv, _ = multiscale_curvature_features(
        I,
        sigmas=(1.0, 2.0, 4.0, 8.0),
        eps=1e-8,
        grad_quantile=0.20,
    )
    orient = structure_tensor_features(I, sigma=2.0)
    out = {f"curv__{k}": v for k, v in curv.items()}
    out.update({f"orient__{k}": v for k, v in orient.items()})
    return out


def robust_center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - center[None, :]), axis=0)
    scale = 1.4826 * mad
    q25 = np.nanpercentile(X, 25, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std = np.nanstd(X, axis=0, ddof=1)
    bad = ~np.isfinite(scale) | (scale <= 1e-12)
    scale[bad] = iqr_scale[bad]
    bad = ~np.isfinite(scale) | (scale <= 1e-12)
    scale[bad] = std[bad]
    bad = ~np.isfinite(scale) | (scale <= 1e-12)
    scale[bad] = 1.0
    return center, scale


def percentile_rank(value: float, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not np.isfinite(value):
        return np.nan
    return float((1 + np.sum(ref <= value)) / (len(ref) + 1))


def pca_scores(Z: np.ndarray, z_star: np.ndarray, variance_target: float, max_components: int):
    max_possible = min(Z.shape[1], max(1, Z.shape[0] - 1))
    full = PCA(n_components=max_possible, svd_solver="full")
    scores_full = full.fit_transform(Z)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_target = int(np.searchsorted(cumulative, variance_target, side="left") + 1)
    n_keep = max(2, min(max_components, n_target, max_possible))
    scores = scores_full[:, :n_keep]
    star_score = full.transform(z_star.reshape(1, -1))[0, :n_keep]
    return full, scores, star_score, n_keep, float(cumulative[n_keep - 1])


def covariance_distances(scores: np.ndarray, star_score: np.ndarray, method: str):
    if method == "mcd":
        estimator = MinCovDet(random_state=42, assume_centered=False).fit(scores)
    elif method == "ledoitwolf":
        estimator = LedoitWolf(assume_centered=False).fit(scores)
    else:
        raise ValueError(method)
    ref_d = np.sqrt(np.clip(estimator.mahalanobis(scores), 0.0, None))
    star_d = float(np.sqrt(max(float(estimator.mahalanobis(star_score.reshape(1, -1))[0]), 0.0)))
    return estimator, ref_d, star_d


def pairwise_metric_distance(scores: np.ndarray, star_score: np.ndarray, precision: np.ndarray) -> np.ndarray:
    D = scores - star_score[None, :]
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, precision, D), 0.0, None))


def main(
    image_path: Path,
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    phase2_excluded_path: Path | None,
    starry_excluded_path: Path | None,
    artist: str,
    long_side: int,
    variance_target: float,
    max_pca_components: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ref, counts = load_reference(
        train_path,
        test_path,
        phase2_excluded_path,
        starry_excluded_path,
        artist,
    )

    star_features = extract_legacy_geometry(image_path, long_side=long_side)
    geometry_cols = sorted(c for c in ref.columns if c.startswith("curv__") or c.startswith("orient__"))
    if not geometry_cols:
        raise RuntimeError("No geometry columns in reference tables.")
    missing = [c for c in geometry_cols if c not in star_features]
    if missing:
        raise RuntimeError(f"Starry Night extraction missing {len(missing)} geometry columns.")

    X = ref[geometry_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x = np.array([star_features[c] for c in geometry_cols], dtype=float)
    medians = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isfinite(X), X, medians[None, :])
    x_imp = np.where(np.isfinite(x), x, medians)

    center, scale = robust_center_scale(X_imp)
    Z = (X_imp - center[None, :]) / scale[None, :]
    z_star = (x_imp - center) / scale

    rms_ref = np.sqrt(np.mean(Z * Z, axis=1))
    rms_star = float(np.sqrt(np.mean(z_star * z_star)))
    rms_pct = percentile_rank(rms_star, rms_ref)

    pca, scores, star_score, n_keep, cumulative_var = pca_scores(
        Z,
        z_star,
        variance_target=float(variance_target),
        max_components=int(max_pca_components),
    )

    method_rows = []
    nearest_frames = []
    for method in ["mcd", "ledoitwolf"]:
        try:
            estimator, ref_d, star_d = covariance_distances(scores, star_score, method)
            pct = percentile_rank(star_d, ref_d)
            pair_d = pairwise_metric_distance(scores, star_score, estimator.precision_)
            order = np.argsort(pair_d)[:10]
            nn = ref.iloc[order][["split", "artist", "filename"]].copy()
            nn["method"] = method
            nn["covariance_metric_distance_to_starry"] = pair_d[order]
            nearest_frames.append(nn)
            method_rows.append({
                "method": method,
                "starry_distance": star_d,
                "distance_percentile_within_artist": pct,
                "reference_distance_median": float(np.median(ref_d)),
                "reference_distance_q95": float(np.percentile(ref_d, 95)),
                "fit_status": "ok",
                "error": None,
            })
        except Exception as exc:
            method_rows.append({
                "method": method,
                "starry_distance": np.nan,
                "distance_percentile_within_artist": np.nan,
                "reference_distance_median": np.nan,
                "reference_distance_q95": np.nan,
                "fit_status": "failed",
                "error": repr(exc),
            })

    covariance_df = pd.DataFrame(method_rows)
    covariance_df.to_csv(output_dir / "starry_covariance_distance_methods.csv", index=False)
    if nearest_frames:
        pd.concat(nearest_frames, ignore_index=True).to_csv(
            output_dir / "starry_covariance_nearest_neighbors.csv", index=False
        )

    feature_rows = []
    for j, feature in enumerate(geometry_cols):
        vals = X_imp[:, j]
        pct = percentile_rank(x_imp[j], vals)
        feature_rows.append({
            "feature": feature,
            "starry_value": float(x_imp[j]),
            "reference_median": float(np.median(vals)),
            "reference_q25": float(np.percentile(vals, 25)),
            "reference_q75": float(np.percentile(vals, 75)),
            "percentile": pct,
            "robust_z": float(z_star[j]),
            "abs_robust_z": float(abs(z_star[j])),
        })
    feature_df = pd.DataFrame(feature_rows).sort_values("abs_robust_z", ascending=False)
    feature_df.to_csv(output_dir / "starry_feature_percentiles_after_reference_cleaning.csv", index=False)

    coords = scores[:, :2] if scores.shape[1] >= 2 else np.column_stack([scores[:, 0], np.zeros(len(scores))])
    star_coord = star_score[:2] if len(star_score) >= 2 else np.array([star_score[0], 0.0])
    pca_df = ref[["split", "artist", "filename"]].copy()
    pca_df["kind"] = "reference"
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df = pd.concat([
        pca_df,
        pd.DataFrame([{
            "split": "uploaded",
            "artist": artist,
            "filename": image_path.name,
            "kind": "starry_night",
            "PC1": float(star_coord[0]),
            "PC2": float(star_coord[1]),
        }]),
    ], ignore_index=True)
    pca_df.to_csv(output_dir / "starry_covaware_pca_coordinates.csv", index=False)

    summary_row = {
        "reference_artist": artist,
        "reference_n_after_all_exclusions": len(ref),
        "phase2_test_exclusions": counts["phase2_test_exclusions"],
        "starry_reference_exclusions": counts["starry_reference_exclusions"],
        "long_side": long_side,
        "n_geometry_features": len(geometry_cols),
        "rms_robust_z_distance": rms_star,
        "rms_robust_z_percentile": rms_pct,
        "pca_components_for_covariance": n_keep,
        "pca_cumulative_variance": cumulative_var,
        "n_features_abs_robust_z_ge_2": int(np.sum(np.abs(z_star) >= 2.0)),
        "n_features_abs_robust_z_ge_3": int(np.sum(np.abs(z_star) >= 3.0)),
    }
    for row in method_rows:
        prefix = row["method"]
        summary_row[f"{prefix}_distance"] = row["starry_distance"]
        summary_row[f"{prefix}_distance_percentile"] = row["distance_percentile_within_artist"]
        summary_row[f"{prefix}_fit_status"] = row["fit_status"]

    summary = pd.DataFrame([summary_row])
    summary.to_csv(output_dir / "starry_covaware_position_summary.csv", index=False)
    pd.DataFrame([star_features]).to_csv(output_dir / "starry_geometry_features_recomputed.csv", index=False)

    print("\nCovariance-aware Starry Night positioning:")
    print(summary.to_string(index=False))
    print("\nCovariance methods:")
    print(covariance_df.to_string(index=False))
    print("\nMost extreme features after reference cleaning:")
    print(feature_df[["feature", "percentile", "robust_z", "abs_robust_z"]].head(12).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Covariance-aware positioning of The Starry Night within a duplicate-clean Van Gogh reference corpus.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3b_starry_covaware"))
    p.add_argument("--phase2-excluded-test", type=Path, default=None)
    p.add_argument("--starry-reference-exclusions", type=Path, default=None)
    p.add_argument("--artist", type=str, default="VanGogh")
    p.add_argument("--long-side", type=int, default=512)
    p.add_argument("--variance-target", type=float, default=0.90)
    p.add_argument("--max-pca-components", type=int, default=20)
    args = p.parse_args()
    main(
        args.image,
        args.train,
        args.test,
        args.output_dir,
        args.phase2_excluded_test,
        args.starry_reference_exclusions,
        args.artist,
        args.long_side,
        args.variance_target,
        args.max_pca_components,
    )
