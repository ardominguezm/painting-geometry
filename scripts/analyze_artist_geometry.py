from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def load_clean_geometry(
    train_path: Path,
    test_path: Path,
    excluded_test_path: Path | None = None,
) -> pd.DataFrame:
    train = pd.read_csv(train_path).copy()
    test = pd.read_csv(test_path).copy()
    train["split"] = "train"
    test["split"] = "test"

    if excluded_test_path is not None and excluded_test_path.exists():
        excluded = pd.read_csv(excluded_test_path)
        if not excluded.empty:
            keys = set(zip(excluded["artist"].astype(str), excluded["filename"].astype(str)))
            mask = [
                (str(a), str(f)) not in keys
                for a, f in zip(test["artist"], test["filename"])
            ]
            test = test.loc[mask].copy()

    return pd.concat([train, test], ignore_index=True)


def geometry_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c.startswith("curv__") or c.startswith("orient__")
    )


def epsilon_squared_kw(H: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    eps2 = (H - k + 1.0) / (n - k)
    return float(np.clip(eps2, 0.0, 1.0))


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(2.0 * u / (n1 * n2) - 1.0)


def global_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    artists = sorted(df["artist"].dropna().astype(str).unique())
    rows = []
    for feature in features:
        groups = []
        ns = []
        for artist in artists:
            vals = pd.to_numeric(
                df.loc[df["artist"].astype(str) == artist, feature],
                errors="coerce",
            ).dropna().to_numpy()
            if vals.size:
                groups.append(vals)
                ns.append(vals.size)
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        n = int(sum(ns))
        k = len(groups)
        rows.append({
            "feature": feature,
            "H": float(H),
            "p_value": float(p),
            "n": n,
            "n_artists": k,
            "epsilon_squared": epsilon_squared_kw(float(H), n, k),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        _, q, _, _ = multipletests(out["p_value"].to_numpy(), method="fdr_bh")
        out["q_fdr_bh"] = q
        out = out.sort_values(
            ["epsilon_squared", "q_fdr_bh"],
            ascending=[False, True],
        ).reset_index(drop=True)
    return out


def pairwise_tests(
    df: pd.DataFrame,
    features: list[str],
    top_features: list[str],
) -> pd.DataFrame:
    artists = sorted(df["artist"].dropna().astype(str).unique())
    rows = []

    for feature in top_features:
        feature_rows = []
        for a, b in combinations(artists, 2):
            x = pd.to_numeric(
                df.loc[df["artist"].astype(str) == a, feature],
                errors="coerce",
            ).dropna().to_numpy()
            y = pd.to_numeric(
                df.loc[df["artist"].astype(str) == b, feature],
                errors="coerce",
            ).dropna().to_numpy()
            if x.size == 0 or y.size == 0:
                continue
            u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
            feature_rows.append({
                "feature": feature,
                "artist_a": a,
                "artist_b": b,
                "n_a": int(x.size),
                "n_b": int(y.size),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
                "U": float(u),
                "p_value": float(p),
                "rank_biserial_a_vs_b": rank_biserial_from_u(float(u), len(x), len(y)),
            })

        if feature_rows:
            pvals = np.array([r["p_value"] for r in feature_rows], dtype=float)
            _, q, _, _ = multipletests(pvals, method="fdr_bh")
            for r, qv in zip(feature_rows, q):
                r["q_fdr_bh_within_feature"] = float(qv)
            rows.extend(feature_rows)

    return pd.DataFrame(rows)


def summarize_by_artist(
    df: pd.DataFrame,
    top_features: list[str],
) -> pd.DataFrame:
    rows = []
    for artist, group in df.groupby("artist", sort=True):
        for feature in top_features:
            x = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy()
            if not x.size:
                continue
            q25, med, q75 = np.percentile(x, [25, 50, 75])
            rows.append({
                "artist": artist,
                "feature": feature,
                "n": int(x.size),
                "median": float(med),
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(q75 - q25),
            })
    return pd.DataFrame(rows)


def main(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    excluded_test_path: Path | None,
    top_n: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean_geometry(train_path, test_path, excluded_test_path)
    features = geometry_columns(df)
    if not features:
        raise RuntimeError("No curvature/orientation features found.")

    global_df = global_tests(df, features)
    global_df.to_csv(output_dir / "artist_geometry_global.csv", index=False)

    top_features = global_df.head(int(top_n))["feature"].tolist()
    pd.DataFrame({"feature": top_features}).to_csv(
        output_dir / "artist_geometry_top_features.csv",
        index=False,
    )

    pairwise_df = pairwise_tests(df, features, top_features)
    pairwise_df.to_csv(output_dir / "artist_geometry_pairwise.csv", index=False)

    summary_df = summarize_by_artist(df, top_features)
    summary_df.to_csv(output_dir / "artist_geometry_summary_by_artist.csv", index=False)

    metadata = pd.DataFrame([{
        "n_total_clean": len(df),
        "n_train": int((df["split"] == "train").sum()),
        "n_test_clean": int((df["split"] == "test").sum()),
        "n_artists": int(df["artist"].nunique()),
        "n_geometry_features": len(features),
        "top_n_pairwise": int(top_n),
        "test_exclusions_applied": bool(
            excluded_test_path is not None and excluded_test_path.exists()
        ),
    }])
    metadata.to_csv(output_dir / "artist_geometry_metadata.csv", index=False)

    print("\nTop geometry features by Kruskal-Wallis epsilon^2:")
    print(
        global_df[
            ["feature", "H", "epsilon_squared", "q_fdr_bh"]
        ].head(top_n).to_string(index=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Artist-wise inferential analysis of Phase-I geometry features."
    )
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase3_artist"))
    p.add_argument("--excluded-test", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()
    main(
        args.train,
        args.test,
        args.output_dir,
        args.excluded_test,
        args.top_n,
    )
