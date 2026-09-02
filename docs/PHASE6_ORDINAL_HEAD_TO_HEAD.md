# Phase VI — Tarozo ordinal patterns vs multiscale level-set geometry

## Scientific purpose

Phase VI compares the proposed multiscale level-set curvature representation with the closest interpretable prior representation identified in the literature: the tie-aware two-by-two ordinal patterns introduced by Tarozo et al. (PNAS Nexus, 2025).

The comparison is intentionally **representation-controlled**. Absolute accuracies are not compared with the published Tarozo experiment because the datasets, style labels, and validation protocols differ. Instead, all newly fitted representations are evaluated on the same ArtBench pilot using the same RBF-SVM probe, hyperparameter grid, and artist-disjoint nested cross-validation used in Phases IV/IVb.

## Ordinal representations

The extraction uses `ordpy>=1.2.0`, whose `two_by_two_patterns` implementation supports the 2025 two-by-two ordinal-pattern method.

For each 256x256 ArtBench image, RGB is converted to grayscale with the standard `skimage` luminance transformation. Overlapping two-by-two windows with unit delay are summarized as:

- `OP_HC`: normalized permutation entropy H and statistical complexity C;
- `OP11`: probabilities of the 11 tie-aware Tarozo pattern groups;
- `OP24`: probabilities of the standard 24 two-by-two ordinal patterns;
- `OP75`: probabilities of all 75 tie-aware ordinal patterns.

The 75-pattern representation explicitly preserves equal intensity ranks rather than resolving ties by position or random perturbation.

## Core level-set representation

The direct geometric comparator is `K40`: the 40 scale-normalized derivative-of-Gaussian level-set curvature summaries at reference scales sigma = {1,2,4,8}. The four auxiliary orientation summaries are excluded from the primary head-to-head so the comparison targets the core curvature contribution.

## Validation protocol

Primary validation is artist-disjoint nested CV:

- outer `StratifiedGroupKFold`: 5 folds, `group=artist`;
- inner grouped CV: 3 folds;
- RBF-SVM hyperparameters: `C={1,3,10}`, `gamma={scale,0.01,0.03}`;
- primary metric: Macro-F1;
- uncertainty: artist-level grouped bootstrap.

The exact Phase-IVb OOF predictions for `K40`, the 90-feature conventional baseline, and the baseline+curvature matched model are reused rather than refit.

## Prespecified head-to-head contrasts

The central tests are:

1. `OP75_K40 - OP75` — does curvature add information beyond the full tie-aware ordinal representation?
2. `OP75_K40_k40 - OP75_k40` — does the increment survive exact dimensionality matching at 40 features selected inside training folds?

The source-controlled WikiArt-8 subset is treated as the most conservative style-level test. There, two additional contextual comparisons are run:

3. `B90_OP75_K40 - B90_OP75` — curvature after conventional appearance plus ordinal patterns;
4. `B90_OP75_K40_k90 - B90_OP75_k90` — the same comparison at exact baseline dimensionality.

BH-FDR is reported across the prespecified geometry-increment tests within each dataset. Additional comparisons (e.g. OP75 vs OP11, OP24, and H-C) are secondary and used to place the Tarozo representation ladder in context.

## Interpretation rules

A positive result supports **complementarity**, not superiority of differential geometry in general. A null result means that the ordinal representation already captures most of the predictive information available to this probe. A negative result is also scientifically informative and should not be hidden.

No claim is made that the experiment reproduces the published ~28% Tarozo accuracy, because that work uses a different WikiArt corpus, 20 styles, stratified image splits, and XGBoost. The purpose here is a controlled comparison of representations under unseen-artist generalization.

## Colab

`notebooks/08_phase6_ordinal_geometry_head_to_head_colab.ipynb`

Inputs:

- `painting_geometry_phase4_artbench_pilot.zip`
- `painting_geometry_phase4b_scale_hierarchy.zip`

Output:

- `painting_geometry_phase6_ordinal_head_to_head.zip`
