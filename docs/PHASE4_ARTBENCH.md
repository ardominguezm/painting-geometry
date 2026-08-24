# Phase IV — ArtBench-10 style generalization

## Scientific purpose

Phase IV tests whether the level-set geometry identified at the artist level also carries information about **artistic style** and, crucially, whether that information generalizes to **artists never seen during training**.

The primary question is:

> Does multiscale level-set geometry add style-discriminative information beyond conventional appearance descriptors under artist-disjoint evaluation?

ArtBench-10 contains 60,000 images across 10 balanced artistic styles (5,000 train and 1,000 test per style) and provides artwork metadata including artist attribution.

## Why the ordinary train/test split is not sufficient

If the same artist occurs in both training and test sets, a style classifier can partially succeed by learning artist identity. Therefore the central Phase-IV protocol is nested `StratifiedGroupKFold` with:

- label = artistic style;
- group = artist;
- outer folds = unseen-artist evaluation;
- inner folds = training-only SVM model selection.

The official ArtBench split is retained only as a secondary comparability benchmark.

## Feature families

At the native 256 px ArtBench resolution, Phase IV uses the Phase-III scale-normalized derivative-of-Gaussian representation with reference scales defined at 512 px:

\[
\sigma_{\mathrm{px}}(256)=\sigma_{\mathrm{ref}}\frac{256}{512},
\qquad
\sigma_{\mathrm{ref}}\in\{1,2,4,8\}.
\]

Models:

- `B_strong_full`: multiscale gradients, non-degenerate edge densities, global orientation histogram, multi-distance GLCM, and LBP;
- `G_geometry_full`: scale-normalized level-set curvature + structure tensor;
- `BG_combined_full`: baseline + geometry;
- matched-dimensionality `k=40` versions selected inside the training pipeline.

The main quantity is

\[
\Delta F_1 =
F_1(BG)-F_1(B).
\]

For artist-disjoint evaluation, uncertainty is estimated by bootstrapping **artist groups**, not individual paintings.

## Source-confound sensitivity

ArtBench documents three source databases. `Surrealism` and `Ukiyo-e` are single-style source databases, whereas the other eight selected styles are WikiArt-derived. Therefore all experiments are repeated on:

1. all ten ArtBench styles;
2. a WikiArt-derived eight-style subset excluding `surrealism` and `ukiyo_e`.

A result that survives the eight-style sensitivity analysis is less plausibly explained by acquisition/source differences.

## Pilot design

The first run uses:

- 300 train images per style;
- 100 test images per style;
- caps on images per artist during pilot sampling;
- 4,000 paintings total.

GO criterion:

- `BG_combined_full - B_strong_full` positive;
- 95% artist-group bootstrap CI entirely above zero in artist-disjoint nested CV;
- preferably reproduced in the WikiArt-derived eight-style subset;
- matched-`k=40` contrast used as a dimensionality control.

If the pilot is positive, the next stage can scale the same frozen protocol to the full ArtBench-10 corpus and add a style-level scale-anatomy analysis.

## Reproducibility

Colab:

`notebooks/05_phase4_artbench_style_generalization_colab.ipynb`

Scripts:

- `scripts/prepare_artbench_manifest.py`
- `scripts/extract_artbench_style_features.py`
- `scripts/run_artbench_style_generalization.py`

The notebook downloads only the 256×256 split rather than the much larger original-resolution archives.
