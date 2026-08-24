# Phase IIIb — final confirmation layer

Phase IIIb is a short confirmation layer after the main Phase I–III experiments. It does not rerun the expensive corpus-wide curvature extraction.

Open in Google Colab:

https://colab.research.google.com/github/ardominguezm/painting-geometry/blob/multiscale-corpus-analysis/notebooks/04_phase3b_starry_validation_colab.ipynb

## Goals

1. Recompute the Phase-III scale-ablation metrics after applying the Phase-II validation leakage exclusions.
2. Audit whether the uploaded *The Starry Night* reproduction, or a perceptually near-identical version, is already present in the Van Gogh reference corpus.
3. Reposition *The Starry Night* after reference cleaning using a covariance-aware multivariate distance.

## Starry-specific duplicate audit

`scripts/audit_starry_reference.py` compares the uploaded image against every Van Gogh training and validation image using SHA1, pHash, and dHash.

The permissive screen is

```text
exact OR pHash <= 10 OR dHash <= 10
```

and is used only to generate candidates and a visual contact sheet.

The automatic exclusion rule is deliberately stricter:

```text
exact OR (pHash <= 4 AND dHash <= 4)
```

Only strict exclusions are removed from the Van Gogh reference used for the final positioning analysis.

## Covariance-aware position

`scripts/position_starry_night_covaware.py` first reproduces the Phase-III robust RMS distance on the cleaned reference. It then robustly standardizes the 44 geometry descriptors, fits PCA using the Van Gogh reference only, and retains the smallest number of components explaining at least 90% of the variance, capped at 20 components.

Two covariance-aware distances are reported:

- Minimum Covariance Determinant (MCD) Mahalanobis distance — primary robust result;
- Ledoit-Wolf Mahalanobis distance — regularized sensitivity analysis.

The conclusion is based on the within-reference percentile of the uploaded image under each distance definition. Exact numerical agreement is not expected; qualitative agreement about whether the painting is central/moderate versus extreme is the robustness criterion.

## Leakage-clean scale metrics

`scripts/recompute_clean_scale_metrics.py` reuses the stored Phase-III predictions and removes the Phase-II excluded validation images before recalculating Macro-F1 and paired bootstrap intervals. It also evaluates the incremental contribution of the coarsest scale through

```text
S1248 - S124
```

without retraining the Phase-III models.

## Expected output

The Colab packages the final confirmation files as

```text
painting_geometry_phase3b_results.zip
```

This package should be archived alongside the Phase I, II, and III result bundles before freezing the manuscript claims.
