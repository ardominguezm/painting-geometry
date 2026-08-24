from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

SOURCE_SPECIFIC_STYLES = {"surrealism", "ukiyo_e"}
SCALES = (1.0, 2.0, 4.0, 8.0)
SCALE_TAGS = {1.0: "1p0", 2.0: "2p0", 4.0: "4p0", 8.0: "8p0"}


def clean_artist(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def curvature_scale_columns(df: pd.DataFrame, sigma: float) -> list[str]:
    tag = SCALE_TAGS[float(sigma)]
    prefix = f"geom__curv__kappa_ref_s{tag}_"
    return sorted(c for c in df.columns if c.startswith(prefix))


def all_curvature_columns(df: pd.DataFrame) -> list[str]:
    return [c for s in SCALES for c in curvature_scale_columns(df, s)]


def robust_center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
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


def robust_z_dataframe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med[None, :])
    center, scale = robust_center_scale(X)
    Z = (X - center[None, :]) / scale[None, :]
    return pd.DataFrame(Z, columns=cols, index=df.index)


def style_fingerprint(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    cols = all_curvature_columns(df)
    Z = robust_z_dataframe(df, cols)
    work = pd.concat([df[["style"]].reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    rows = []
    for style, grp in work.groupby("style", sort=True):
        for sigma in SCALES:
            tag = SCALE_TAGS[sigma]
            for col in curvature_scale_columns(df, sigma):
                vals = grp[col].to_numpy(dtype=float)
                rows.append({
                    "dataset": dataset_name,
                    "style": style,
                    "sigma_ref": sigma,
                    "feature": col,
                    "descriptor": col.replace(f"geom__curv__kappa_ref_s{tag}_", ""),
                    "style_median_robust_z": float(np.nanmedian(vals)),
                    "style_mean_robust_z": float(np.nanmean(vals)),
                    "style_iqr_robust_z": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
                    "n_images": int(len(grp)),
                })
    return pd.DataFrame(rows)


def style_centroids_by_scale(df: pd.DataFrame, sigma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = curvature_scale_columns(df, sigma)
    Z = robust_z_dataframe(df, cols)
    work = pd.concat([df[["style", "artist"]].reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    work["artist"] = work["artist"].map(clean_artist)
    style_cent = work.groupby("style", sort=True)[cols].median()
    as_work = work[work["artist"].ne("")].copy()
    artist_style_cent = as_work.groupby(["style", "artist"], sort=True)[cols].median().reset_index()
    return style_cent, artist_style_cent


def rms_distance_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    X = centroids.to_numpy(dtype=float)
    D = squareform(pdist(X, metric="euclidean")) / np.sqrt(X.shape[1])
    return pd.DataFrame(D, index=centroids.index, columns=centroids.index)


def single_style_subset(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["artist"] = work["artist"].map(clean_artist)
    work = work[work["artist"].ne("")].copy()
    n_styles = work.groupby("artist")["style"].nunique()
    keep = set(n_styles[n_styles == 1].index)
    return work[work["artist"].isin(keep)].reset_index(drop=True)


def nested_variance_partition_single_style(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Nested style -> artist -> painting SS partition on single-style artists."""
    work = single_style_subset(df)
    rows = []
    for sigma in SCALES:
        tag = SCALE_TAGS[sigma]
        for feature in curvature_scale_columns(work, sigma):
            d = work[["style", "artist", feature]].copy()
            d[feature] = pd.to_numeric(d[feature], errors="coerce")
            d = d.dropna(subset=[feature]).reset_index(drop=True)
            y = d[feature].to_numpy(dtype=float)
            grand = float(np.mean(y))
            ss_total = float(np.sum((y - grand) ** 2))
            if ss_total <= 1e-15:
                ss_style = ss_artist = ss_residual = 0.0
            else:
                style_means = d.groupby("style")[feature].mean()
                ss_style = 0.0
                for style, grp in d.groupby("style"):
                    ss_style += len(grp) * float((grp[feature].mean() - grand) ** 2)
                ss_artist = 0.0
                artist_means = {}
                for (style, artist), grp in d.groupby(["style", "artist"]):
                    mu = float(grp[feature].mean())
                    artist_means[(style, artist)] = mu
                    ss_artist += len(grp) * float((mu - style_means.loc[style]) ** 2)
                ss_residual = 0.0
                for (style, artist), grp in d.groupby(["style", "artist"]):
                    mu = artist_means[(style, artist)]
                    vals = grp[feature].to_numpy(dtype=float)
                    ss_residual += float(np.sum((vals - mu) ** 2))
                comp = ss_style + ss_artist + ss_residual
                if comp > 0:
                    factor = ss_total / comp
                    ss_style *= factor
                    ss_artist *= factor
                    ss_residual *= factor
            between = ss_style + ss_artist
            rows.append({
                "dataset": dataset_name,
                "sigma_ref": sigma,
                "feature": feature,
                "descriptor": feature.replace(f"geom__curv__kappa_ref_s{tag}_", ""),
                "n_images_single_style_artists": int(len(d)),
                "n_single_style_artists": int(d["artist"].nunique()),
                "n_styles": int(d["style"].nunique()),
                "style_fraction": float(ss_style / ss_total) if ss_total > 0 else 0.0,
                "artist_within_style_fraction": float(ss_artist / ss_total) if ss_total > 0 else 0.0,
                "painting_residual_fraction": float(ss_residual / ss_total) if ss_total > 0 else 0.0,
                "style_share_of_between_artist_variation": float(ss_style / between) if between > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def single_style_artist_centroids(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    cols = curvature_scale_columns(df, sigma)
    Z = robust_z_dataframe(df, cols)
    work = pd.concat([df[["style", "artist"]].reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    work["artist"] = work["artist"].map(clean_artist)
    work = work[work["artist"].ne("")].copy()
    n_styles = work.groupby("artist")["style"].nunique()
    keep_artists = set(n_styles[n_styles == 1].index)
    work = work[work["artist"].isin(keep_artists)].copy()
    return work.groupby(["artist", "style"], sort=True)[cols].median().reset_index()


def artist_centroid_style_test(artist_cent: pd.DataFrame, feature_cols: list[str], n_perm: int = 4999, seed: int = 42) -> dict[str, float]:
    X = artist_cent[feature_cols].to_numpy(dtype=float)
    labels = artist_cent["style"].astype(str).to_numpy()
    n = len(labels)
    styles = np.unique(labels)
    k = len(styles)
    grand = X.mean(axis=0)
    ss_total = float(np.sum((X - grand) ** 2))
    def between_ss(lab: np.ndarray) -> float:
        ss = 0.0
        for s in np.unique(lab):
            idx = np.flatnonzero(lab == s)
            mu = X[idx].mean(axis=0)
            ss += len(idx) * float(np.sum((mu - grand) ** 2))
        return ss
    ss_between = between_ss(labels)
    ss_within = max(ss_total - ss_between, 0.0)
    dfb = max(k - 1, 1)
    dfw = max(n - k, 1)
    F = (ss_between / dfb) / (ss_within / dfw + 1e-15)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(labels)
        ss_b = between_ss(perm)
        ss_w = max(ss_total - ss_b, 0.0)
        fp = (ss_b / dfb) / (ss_w / dfw + 1e-15)
        if fp >= F:
            ge += 1
    p = (1 + ge) / (int(n_perm) + 1)
    return {
        "n_single_style_artists": int(n),
        "n_styles": int(k),
        "pseudo_F": float(F),
        "eta2_style_artist_centroids": float(eta2),
        "permutation_p": float(p),
        "n_permutations": int(n_perm),
    }


def within_style_dispersion(style_cent: pd.DataFrame, artist_style_cent: pd.DataFrame, sigma: float, dataset_name: str) -> pd.DataFrame:
    cols = style_cent.columns.tolist()
    rows = []
    for style, grp in artist_style_cent.groupby("style", sort=True):
        c = style_cent.loc[style, cols].to_numpy(dtype=float)
        A = grp[cols].to_numpy(dtype=float)
        d = np.sqrt(np.mean((A - c[None, :]) ** 2, axis=1))
        rows.append({
            "dataset": dataset_name,
            "sigma_ref": sigma,
            "style": style,
            "n_artist_style_units": len(grp),
            "median_artist_to_style_centroid_distance": float(np.median(d)),
            "mean_artist_to_style_centroid_distance": float(np.mean(d)),
            "iqr_artist_to_style_centroid_distance": float(np.percentile(d, 75) - np.percentile(d, 25)),
        })
    return pd.DataFrame(rows)


def mantel_spearman(D1: pd.DataFrame, D2: pd.DataFrame, n_perm: int, seed: int) -> tuple[float, float]:
    common = sorted(set(D1.index) & set(D2.index))
    A = D1.loc[common, common].to_numpy(dtype=float)
    B = D2.loc[common, common].to_numpy(dtype=float)
    iu = np.triu_indices(len(common), k=1)
    obs = float(spearmanr(A[iu], B[iu]).statistic)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        p = rng.permutation(len(common))
        Bp = B[p][:, p]
        r = float(spearmanr(A[iu], Bp[iu]).statistic)
        if abs(r) >= abs(obs):
            ge += 1
    return obs, (1 + ge) / (int(n_perm) + 1)


def matrix_correlations(distance_mats: dict[float, pd.DataFrame], dataset_name: str, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    scales = list(SCALES)
    for i, s1 in enumerate(scales):
        for s2 in scales[i:]:
            if s1 == s2:
                rho, p = 1.0, 0.0
            else:
                rho, p = mantel_spearman(distance_mats[s1], distance_mats[s2], n_perm=n_perm, seed=seed + int(10 * s1 + s2))
            rows.append({
                "dataset": dataset_name,
                "sigma_a": s1,
                "sigma_b": s2,
                "n_styles": len(distance_mats[s1]),
                "mantel_spearman_rho": float(rho),
                "permutation_p": float(p),
                "n_permutations": int(n_perm if s1 != s2 else 0),
            })
    return pd.DataFrame(rows)


def closest_farthest_pairs(D: pd.DataFrame, dataset_name: str, sigma: float) -> pd.DataFrame:
    rows = []
    labels = list(D.index)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append({"dataset": dataset_name, "sigma_ref": sigma, "style_a": labels[i], "style_b": labels[j], "distance": float(D.iloc[i, j])})
    out = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    out["rank_nearest"] = np.arange(1, len(out) + 1)
    out["rank_farthest"] = len(out) - np.arange(len(out))
    return out


def plot_fingerprint(fp: pd.DataFrame, out: Path, title: str):
    styles = sorted(fp["style"].unique())
    descriptors = sorted(set(fp["descriptor"]))
    keys = [(s, d) for s in SCALES for d in descriptors]
    mat = np.full((len(styles), len(keys)), np.nan)
    lookup = {(r.style, float(r.sigma_ref), r.descriptor): r.style_median_robust_z for r in fp.itertuples(index=False)}
    for i, style in enumerate(styles):
        for j, (sigma, desc) in enumerate(keys):
            mat[i, j] = lookup.get((style, sigma, desc), np.nan)
    fig, ax = plt.subplots(figsize=(18, max(5.5, 0.48 * len(styles) + 2)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(styles)))
    ax.set_yticklabels(styles)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f"s={int(s)}\n{d}" for s, d in keys], rotation=90, fontsize=7)
    for boundary in [len(descriptors) - 0.5, 2 * len(descriptors) - 0.5, 3 * len(descriptors) - 0.5]:
        ax.axvline(boundary, linewidth=0.8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Style median robust z")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_distance_matrix(D: pd.DataFrame, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D.to_numpy(dtype=float), interpolation="nearest")
    ax.set_xticks(np.arange(len(D.columns)))
    ax.set_xticklabels(D.columns, rotation=90)
    ax.set_yticks(np.arange(len(D.index)))
    ax.set_yticklabels(D.index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMS centroid distance (robust-z units)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_dendrogram(cent: pd.DataFrame, out: Path, title: str):
    Z = linkage(cent.to_numpy(dtype=float), method="ward", metric="euclidean")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=cent.index.tolist(), leaf_rotation=90, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Ward linkage distance")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_variance_summary(vp: pd.DataFrame, out: Path, title: str):
    cols = ["style_fraction", "artist_within_style_fraction", "painting_residual_fraction"]
    agg = vp.groupby("sigma_ref")[cols].median().reindex(SCALES)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bottom = np.zeros(len(agg))
    x = np.arange(len(agg))
    for col in cols:
        vals = agg[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=col.replace("_fraction", "").replace("_", " "))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={int(s)}" for s in agg.index])
    ax.set_ylabel("Median fraction of total sum of squares")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_style_effect_tests(tests: pd.DataFrame, out: Path, title: str):
    sub = tests.sort_values("sigma_ref")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["sigma_ref"], sub["eta2_style_artist_centroids"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(SCALES))
    ax.set_xticklabels([str(int(s)) for s in SCALES])
    ax.set_xlabel("Reference scale sigma")
    ax.set_ylabel("Style eta-squared among single-style artist centroids")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_distance_corr(corr: pd.DataFrame, out: Path, title: str):
    M = pd.DataFrame(np.eye(len(SCALES)), index=SCALES, columns=SCALES, dtype=float)
    for r in corr.itertuples(index=False):
        M.loc[r.sigma_a, r.sigma_b] = r.mantel_spearman_rho
        M.loc[r.sigma_b, r.sigma_a] = r.mantel_spearman_rho
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M.to_numpy(dtype=float), vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(SCALES)))
    ax.set_xticklabels([str(int(s)) for s in SCALES])
    ax.set_yticks(np.arange(len(SCALES)))
    ax.set_yticklabels([str(int(s)) for s in SCALES])
    ax.set_xlabel("sigma")
    ax.set_ylabel("sigma")
    ax.set_title(title)
    for i in range(len(SCALES)):
        for j in range(len(SCALES)):
            ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mantel Spearman rho")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def run_one(df: pd.DataFrame, dataset_name: str, output_dir: Path, n_perm: int, seed: int):
    out = output_dir / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    fp = style_fingerprint(df, dataset_name)
    fp.to_csv(out / "style_geometric_fingerprints.csv", index=False)
    plot_fingerprint(fp, out / "Figure_style_geometric_fingerprint.png", f"{dataset_name}: multiscale geometric fingerprints")
    vp = nested_variance_partition_single_style(df, dataset_name)
    vp.to_csv(out / "nested_single_style_variance_partition_features.csv", index=False)
    vp.groupby("sigma_ref")[["style_fraction", "artist_within_style_fraction", "painting_residual_fraction", "style_share_of_between_artist_variation"]].agg(["median", "mean", "min", "max"]).to_csv(out / "nested_single_style_variance_partition_scale_summary.csv")
    plot_variance_summary(vp, out / "Figure_nested_single_style_variance_partition.png", f"{dataset_name}: nested style/artist/painting variation across scale")
    distance_mats = {}
    tests = []
    separation_rows = []
    centroid_long = []
    dispersion_frames = []
    pair_frames = []
    for sigma in SCALES:
        cent, artist_style_cent = style_centroids_by_scale(df, sigma)
        D = rms_distance_matrix(cent)
        distance_mats[sigma] = D
        tag = str(int(sigma))
        D.to_csv(out / f"style_distance_matrix_sigma{tag}.csv")
        plot_distance_matrix(D, out / f"Figure_style_distance_sigma{tag}.png", f"{dataset_name}: style distances at sigma={tag}")
        plot_dendrogram(cent, out / f"Figure_style_dendrogram_sigma{tag}.png", f"{dataset_name}: geometry-derived style dendrogram, sigma={tag}")
        for style, row in cent.iterrows():
            for col, value in row.items():
                centroid_long.append({"dataset": dataset_name, "sigma_ref": sigma, "style": style, "feature": col, "descriptor": col.replace(f"geom__curv__kappa_ref_s{SCALE_TAGS[sigma]}_", ""), "style_centroid_robust_z": float(value)})
        iu = np.triu_indices(len(D), k=1)
        pair_vals = D.to_numpy(dtype=float)[iu]
        separation_rows.append({"dataset": dataset_name, "sigma_ref": sigma, "n_styles": len(D), "mean_pairwise_style_distance": float(np.mean(pair_vals)), "median_pairwise_style_distance": float(np.median(pair_vals)), "min_pairwise_style_distance": float(np.min(pair_vals)), "max_pairwise_style_distance": float(np.max(pair_vals))})
        dispersion_frames.append(within_style_dispersion(cent, artist_style_cent, sigma, dataset_name))
        pair_frames.append(closest_farthest_pairs(D, dataset_name, sigma))
        test_units = single_style_artist_centroids(df, sigma)
        feat_cols = curvature_scale_columns(df, sigma)
        test = artist_centroid_style_test(test_units, feat_cols, n_perm=n_perm, seed=seed + int(sigma))
        test.update({"dataset": dataset_name, "sigma_ref": sigma})
        tests.append(test)
    pd.DataFrame(centroid_long).to_csv(out / "style_centroids_by_scale.csv", index=False)
    pd.DataFrame(separation_rows).to_csv(out / "style_separation_by_scale.csv", index=False)
    pd.concat(dispersion_frames, ignore_index=True).to_csv(out / "within_style_artist_dispersion_by_scale.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(out / "style_pair_distances_by_scale.csv", index=False)
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out / "single_style_artist_centroid_permutation_tests.csv", index=False)
    plot_style_effect_tests(tests_df, out / "Figure_single_style_artist_centroid_effect.png", f"{dataset_name}: style effect among single-style artist centroids")
    corr = matrix_correlations(distance_mats, dataset_name, n_perm=n_perm, seed=seed + 1000)
    corr.to_csv(out / "distance_matrix_scale_mantel_correlations.csv", index=False)
    plot_distance_corr(corr, out / "Figure_distance_matrix_scale_correlations.png", f"{dataset_name}: reorganization of style-distance geometry across scales")
    n_unique_artists = int(df["artist"].map(clean_artist).replace("", np.nan).nunique())
    n_single_style = int((df.assign(_a=df["artist"].map(clean_artist)).query("_a != ''").groupby("_a")["style"].nunique() == 1).sum())
    return {"dataset": dataset_name, "n_images": int(len(df)), "n_styles": int(df["style"].nunique()), "n_unique_artists": n_unique_artists, "n_single_style_artists_for_permutation_test": n_single_style, "n_permutations_per_test": int(n_perm)}


def main(features_path: Path, output_dir: Path, n_perm: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path).copy()
    df["style"] = df["style"].astype(str)
    if "artist" not in df:
        raise KeyError("Phase V requires the artist column from the Phase-IV manifest.")
    df["artist"] = df["artist"].map(clean_artist)
    missing_scale = [s for s in SCALES if len(curvature_scale_columns(df, s)) == 0]
    if missing_scale:
        raise RuntimeError(f"Missing curvature features for scales: {missing_scale}")
    datasets = {"artbench10_all": df.reset_index(drop=True), "artbench10_wikiart8": df[~df["style"].isin(SOURCE_SPECIFIC_STYLES)].reset_index(drop=True)}
    metadata = []
    for name, sub in datasets.items():
        print(f"\n=== Phase V: {name} | {len(sub)} images | {sub['style'].nunique()} styles ===")
        metadata.append(run_one(sub, name, output_dir, n_perm=n_perm, seed=seed))
    pd.DataFrame(metadata).to_csv(output_dir / "phase5_metadata.csv", index=False)
    summaries = []
    for name in datasets:
        base = output_dir / name
        tests = pd.read_csv(base / "single_style_artist_centroid_permutation_tests.csv")
        sep = pd.read_csv(base / "style_separation_by_scale.csv")
        vp = pd.read_csv(base / "nested_single_style_variance_partition_features.csv")
        vps = vp.groupby("sigma_ref")[["style_fraction", "artist_within_style_fraction", "painting_residual_fraction", "style_share_of_between_artist_variation"]].median().reset_index()
        summaries.append(tests.merge(sep, on=["dataset", "sigma_ref"], how="left").merge(vps, on="sigma_ref", how="left"))
    pd.concat(summaries, ignore_index=True).to_csv(output_dir / "phase5_scale_summary.csv", index=False)
    print("\nPhase V complete ->", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase V: multiscale geometric organization of artistic styles.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase5_style_geometry"))
    p.add_argument("--n-permutations", type=int, default=4999)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.features, args.output_dir, args.n_permutations, args.seed)
