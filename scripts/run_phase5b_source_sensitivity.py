from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_phase5_style_geometry import (
    SCALES,
    artist_centroid_style_test,
    curvature_scale_columns,
    single_style_artist_centroids,
)

SOURCE_SPECIFIC_STYLES = ("surrealism", "ukiyo_e")


def evaluate_variant(
    df: pd.DataFrame,
    variant: str,
    dropped_styles: tuple[str, ...],
    n_perm: int,
    seed: int,
) -> list[dict]:
    work = df[~df["style"].isin(dropped_styles)].reset_index(drop=True)
    rows = []
    for sigma in SCALES:
        artist_cent = single_style_artist_centroids(work, sigma)
        cols = curvature_scale_columns(work, sigma)
        test = artist_centroid_style_test(
            artist_cent,
            cols,
            n_perm=n_perm,
            seed=seed + int(sigma),
        )
        test.update(
            {
                "variant": variant,
                "dropped_styles": ";".join(dropped_styles),
                "sigma_ref": sigma,
                "n_images": len(work),
            }
        )
        rows.append(test)
    return rows


def leave_one_style_out_sigma2(
    df: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    sigma = 2.0
    for style in sorted(df["style"].astype(str).unique()):
        work = df[df["style"].astype(str) != style].reset_index(drop=True)
        artist_cent = single_style_artist_centroids(work, sigma)
        cols = curvature_scale_columns(work, sigma)
        test = artist_centroid_style_test(
            artist_cent,
            cols,
            n_perm=n_perm,
            seed=seed + int(sigma),
        )
        test.update(
            {
                "dropped_style": style,
                "sigma_ref": sigma,
                "n_images": len(work),
            }
        )
        rows.append(test)
    return pd.DataFrame(rows).sort_values(
        ["eta2_style_artist_centroids", "permutation_p"], ascending=[True, True]
    )


def main(features: Path, output_dir: Path, n_perm: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features)
    df["style"] = df["style"].astype(str)

    variants = {
        "all10": (),
        "drop_ukiyo_e": ("ukiyo_e",),
        "drop_surrealism": ("surrealism",),
        "drop_both_wikiart8": SOURCE_SPECIFIC_STYLES,
    }

    rows = []
    for name, dropped in variants.items():
        rows.extend(evaluate_variant(df, name, dropped, n_perm, seed))
    primary = pd.DataFrame(rows)

    baseline = primary[primary["variant"] == "all10"][
        ["sigma_ref", "eta2_style_artist_centroids", "pseudo_F"]
    ].rename(
        columns={
            "eta2_style_artist_centroids": "eta2_all10",
            "pseudo_F": "pseudo_F_all10",
        }
    )
    primary = primary.merge(baseline, on="sigma_ref", how="left")
    primary["eta2_change_vs_all10"] = (
        primary["eta2_style_artist_centroids"] - primary["eta2_all10"]
    )
    primary["pseudo_F_change_vs_all10"] = primary["pseudo_F"] - primary["pseudo_F_all10"]
    primary.to_csv(output_dir / "phase5b_source_sensitivity_primary.csv", index=False)

    loo = leave_one_style_out_sigma2(df, n_perm=n_perm, seed=seed)
    all10_s2 = primary[(primary["variant"] == "all10") & (primary["sigma_ref"] == 2.0)].iloc[0]
    loo["eta2_all10_sigma2"] = float(all10_s2["eta2_style_artist_centroids"])
    loo["eta2_change_vs_all10_sigma2"] = (
        loo["eta2_style_artist_centroids"] - loo["eta2_all10_sigma2"]
    )
    loo.to_csv(output_dir / "phase5b_leave_one_style_out_sigma2.csv", index=False)

    print("\nPrimary source-sensitivity control:\n")
    print(
        primary[
            [
                "variant",
                "sigma_ref",
                "n_single_style_artists",
                "n_styles",
                "pseudo_F",
                "eta2_style_artist_centroids",
                "permutation_p",
                "eta2_change_vs_all10",
            ]
        ].to_string(index=False)
    )
    print("\nLeave-one-style-out diagnostic at sigma=2:\n")
    print(
        loo[
            [
                "dropped_style",
                "n_single_style_artists",
                "pseudo_F",
                "eta2_style_artist_centroids",
                "permutation_p",
                "eta2_change_vs_all10_sigma2",
            ]
        ].to_string(index=False)
    )
    print("\nOutputs ->", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase Vb: source-specific sensitivity of the artist-centroid style effect."
    )
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase5b_source_sensitivity"))
    p.add_argument("--n-permutations", type=int, default=4999)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.features, args.output_dir, args.n_permutations, args.seed)
