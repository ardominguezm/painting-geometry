from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import (
    lbp_features,
    multidistance_glcm_features,
    multiscale_gradient_features,
    orientation_histogram_features,
)
from src.curvature_v2 import relative_scale_curvature_features
from src.orientation import structure_tensor_features
from src.preprocessing import preprocess


def balanced_sample(
    df: pd.DataFrame,
    n_per_style: int | None,
    max_per_artist: int | None,
    seed: int,
) -> pd.DataFrame:
    if n_per_style is None or int(n_per_style) <= 0:
        return df.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected = []
    for style, group in df.groupby("style", sort=True):
        g = group.copy()
        target = min(int(n_per_style), len(g))

        artist_ok = "artist" in g.columns and g["artist"].astype(str).str.strip().ne("").any()
        if artist_ok and max_per_artist is not None and int(max_per_artist) > 0:
            pieces = []
            for artist, ag in g.groupby("artist", sort=False):
                idx = ag.index.to_numpy()
                rng.shuffle(idx)
                pieces.append(ag.loc[idx[: min(len(idx), int(max_per_artist))]])
            pool = pd.concat(pieces, ignore_index=False)
        else:
            pool = g

        if len(pool) < target:
            print(
                f"WARNING {style}: artist cap leaves {len(pool)} < requested {target}; "
                "using all capped rows."
            )
            target = len(pool)
        idx = pool.index.to_numpy()
        rng.shuffle(idx)
        selected.append(pool.loc[idx[:target]])

    return pd.concat(selected, ignore_index=True)


def extract_features(
    path: Path,
    long_side: int,
    sigma_refs: tuple[float, ...],
    reference_long_side: int,
) -> dict[str, float]:
    _, I = preprocess(path, long_side=long_side)

    geom = relative_scale_curvature_features(
        I,
        long_side=long_side,
        sigma_refs=sigma_refs,
        reference_long_side=reference_long_side,
        return_maps=False,
    )
    orient_sigma = 2.0 * float(long_side) / float(reference_long_side)
    geom_orient = structure_tensor_features(I, sigma=orient_sigma)

    sigma_px = tuple(
        float(s) * float(long_side) / float(reference_long_side)
        for s in sigma_refs
    )
    baseline = {}
    baseline.update(multiscale_gradient_features(I, sigmas=sigma_px))
    baseline.update(orientation_histogram_features(I, sigma=orient_sigma))
    baseline.update(multidistance_glcm_features(I, distances=(1, 2, 4)))
    baseline.update(lbp_features(I))

    out = {f"geom__curv__{k}": v for k, v in geom.items()}
    out.update({f"geom__orient__{k}": v for k, v in geom_orient.items()})
    out.update({f"base__{k}": v for k, v in baseline.items()})
    return out


def main(
    manifest_path: Path,
    output_path: Path,
    train_per_style: int | None,
    test_per_style: int | None,
    max_train_per_artist: int | None,
    max_test_per_artist: int | None,
    long_side: int,
    sigma_refs: list[float],
    reference_long_side: int,
    seed: int,
):
    manifest = pd.read_csv(manifest_path)
    required = {"split", "style", "filename", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {sorted(missing)}")

    train = balanced_sample(
        manifest[manifest["split"].astype(str) == "train"],
        train_per_style,
        max_train_per_artist,
        seed,
    )
    test = balanced_sample(
        manifest[manifest["split"].astype(str) == "test"],
        test_per_style,
        max_test_per_artist,
        seed + 1,
    )
    sample = pd.concat([train, test], ignore_index=True)
    sample["pilot_selected"] = True

    print("Selected sample:")
    print(sample.groupby(["split", "style"]).size().unstack(fill_value=0).to_string())
    if "artist" in sample.columns:
        print("\nUnique artists per style:")
        print(
            sample[sample["artist"].fillna("").astype(str).str.strip().ne("")]
            .groupby("style")["artist"].nunique()
            .sort_values()
            .to_string()
        )

    rows = []
    failures = []
    sigma_refs_t = tuple(float(x) for x in sigma_refs)
    for rec in tqdm(sample.itertuples(index=False), total=len(sample), desc="ArtBench features", dynamic_ncols=True):
        meta = {
            "split": rec.split,
            "style": rec.style,
            "artist": getattr(rec, "artist", ""),
            "source": getattr(rec, "source", ""),
            "filename": rec.filename,
            "path": rec.path,
            "long_side": int(long_side),
        }
        try:
            feats = extract_features(
                Path(rec.path),
                long_side=int(long_side),
                sigma_refs=sigma_refs_t,
                reference_long_side=int(reference_long_side),
            )
            meta.update(feats)
            rows.append(meta)
        except Exception as exc:
            fail = meta.copy()
            fail["error"] = repr(exc)
            failures.append(fail)

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            output_path.with_name(output_path.stem + "_failures.csv"), index=False
        )

    print("\nFeatures ->", output_path)
    print("Shape:", out.shape)
    print("Failures:", len(failures))
    print("Baseline features:", sum(c.startswith("base__") for c in out.columns))
    print("Geometry features:", sum(c.startswith("geom__") for c in out.columns))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Phase-IV ArtBench style features.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/artbench_pilot_features.csv"))
    p.add_argument("--train-per-style", type=int, default=300)
    p.add_argument("--test-per-style", type=int, default=100)
    p.add_argument("--max-train-per-artist", type=int, default=40)
    p.add_argument("--max-test-per-artist", type=int, default=20)
    p.add_argument("--long-side", type=int, default=256)
    p.add_argument("--sigma-refs", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--reference-long-side", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(
        args.manifest,
        args.output,
        args.train_per_style,
        args.test_per_style,
        args.max_train_per_artist,
        args.max_test_per_artist,
        args.long_side,
        args.sigma_refs,
        args.reference_long_side,
        args.seed,
    )
