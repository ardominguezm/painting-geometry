from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.curvature import multiscale_curvature_features
from src.orientation import structure_tensor_features
from src.preprocessing import preprocess


def load_clean_reference(
    train_path: Path,
    test_path: Path,
    excluded_test_path: Path | None,
    artist: str,
) -> pd.DataFrame:
    train = pd.read_csv(train_path).copy()
    test = pd.read_csv(test_path).copy()

    if excluded_test_path is not None and excluded_test_path.exists():
        excluded = pd.read_csv(excluded_test_path)
        if not excluded.empty:
            keys = set(zip(excluded["artist"].astype(str), excluded["filename"].astype(str)))
            keep = [
                (str(a), str(f)) not in keys
                for a, f in zip(test["artist"], test["filename"])
            ]
            test = test.loc[keep].copy()

    df = pd.concat([train, test], ignore_index=True)
    ref = df[df["artist"].astype(str) == str(artist)].copy()
    if ref.empty:
        raise RuntimeError(f"No reference rows found for artist={artist!r}")
    return ref


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


def rms_robust_distance(Z: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(Z * Z, axis=1))


def percentile_rank(value: float, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not np.isfinite(value):
        return np.nan
    return float((1 + np.sum(ref <= value)) / (len(ref) + 1))


def main(
    image_path: Path,
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    excluded_test_path: Path | None,
    artist: str,
    long_side: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = load_clean_reference(
        train_path,
        test_path,
        excluded_test_path,
        artist=artist,
    )

    star_features = extract_legacy_geometry(image_path, long_side=long_side)
    geometry_cols = sorted(
        c for c in ref.columns
        if c.startswith("curv__") or c.startswith("orient__")
    )
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

    ref_dist = rms_robust_distance(Z)
    star_dist = float(rms_robust_distance(z_star[None, :])[0])
    distance_percentile = percentile_rank(star_dist, ref_dist)

    euclidean = np.sqrt(np.sum((Z - z_star[None, :]) ** 2, axis=1))
    order = np.argsort(euclidean)
    nn = ref.iloc[order[:10]][["artist", "filename"]].copy()
    nn["robust_z_euclidean_distance"] = euclidean[order[:10]]
    nn.to_csv(output_dir / "starry_night_nearest_neighbors.csv", index=False)

    feature_rows = []
    for j, feature in enumerate(geometry_cols):
        vals = X_imp[:, j]
        pct = percentile_rank(x_imp[j], vals)
        two_sided_extreme = float(2.0 * min(pct, 1.0 - pct)) if np.isfinite(pct) else np.nan
        feature_rows.append({
            "feature": feature,
            "starry_value": float(x_imp[j]),
            "reference_median": float(np.median(vals)),
            "reference_q25": float(np.percentile(vals, 25)),
            "reference_q75": float(np.percentile(vals, 75)),
            "percentile": pct,
            "two_sided_tail_fraction": two_sided_extreme,
            "robust_z": float(z_star[j]),
            "abs_robust_z": float(abs(z_star[j])),
        })
    feature_df = pd.DataFrame(feature_rows).sort_values("abs_robust_z", ascending=False)
    feature_df.to_csv(output_dir / "starry_night_feature_percentiles.csv", index=False)

    n_components = min(2, Z.shape[1], max(1, Z.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(Z)
    star_coord = pca.transform(z_star.reshape(1, -1))[0]

    pca_df = ref[["artist", "filename"]].copy()
    pca_df["kind"] = "reference"
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1] if n_components > 1 else 0.0
    star_row = pd.DataFrame([{
        "artist": artist,
        "filename": image_path.name,
        "kind": "starry_night",
        "PC1": float(star_coord[0]),
        "PC2": float(star_coord[1]) if n_components > 1 else 0.0,
    }])
    pca_df = pd.concat([pca_df, star_row], ignore_index=True)
    pca_df.to_csv(output_dir / "starry_night_pca_coordinates.csv", index=False)

    summary = pd.DataFrame([{
        "reference_artist": artist,
        "reference_n": len(ref),
        "long_side": long_side,
        "n_geometry_features": len(geometry_cols),
        "starry_rms_robust_z_distance": star_dist,
        "distance_percentile_within_artist": distance_percentile,
        "n_features_abs_robust_z_ge_2": int(np.sum(np.abs(z_star) >= 2.0)),
        "n_features_abs_robust_z_ge_3": int(np.sum(np.abs(z_star) >= 3.0)),
        "pca_pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pca_pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]) if n_components > 1 else np.nan,
        "interpretation_note": "Percentile is descriptive within the chosen corpus/reference preprocessing; it is not an authenticity or psychological score.",
    }])
    summary.to_csv(output_dir / "starry_night_position_summary.csv", index=False)

    pd.DataFrame([star_features]).to_csv(
        output_dir / "starry_night_geometry_features.csv",
        index=False,
    )

    print("\nStarry Night position within reference artist:")
    print(summary.to_string(index=False))
    print("\nNearest reference paintings:")
    print(nn.to_string(index=False))
    print("\nMost extreme geometry features:")
    print(feature_df.head(12)[["feature", "percentile", "robust_z", "abs_robust_z"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Position a Starry Night image within a clean artist geometry reference corpus.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3_starry"))
    p.add_argument("--excluded-test", type=Path, default=None)
    p.add_argument("--artist", type=str, default="VanGogh")
    p.add_argument("--long-side", type=int, default=512)
    args = p.parse_args()
    main(
        args.image,
        args.train,
        args.test,
        args.output_dir,
        args.excluded_test,
        args.artist,
        args.long_side,
    )
