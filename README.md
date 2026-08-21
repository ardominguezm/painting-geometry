# Multiscale Geometry of Paintings

This repository develops a corpus-scale, reproducible analysis of **luminance level-set geometry in paintings**. The current research branch moves beyond a single-image reading of *The Starry Night* and tests whether multiscale curvature provides artist-specific information beyond conventional edge and texture descriptors.

## Scientific question

Does multiscale level-set curvature provide independent, interpretable information about artistic style once conventional edge and texture structure are controlled for?

The central hypotheses are:

1. curvature statistics differ systematically across artists;
2. curvature adds information beyond conventional appearance descriptors;
3. geometric descriptors are stable across image resolution and smoothing scale;
4. *The Starry Night* can be located quantitatively within the distribution of Van Gogh's oeuvre rather than treated as an isolated exemplar.

## Geometry

For luminance field `I(x,y)`, the signed curvature of its level sets is

```text
kappa = (Ixx Iy^2 - 2 Ix Iy Ixy + Iyy Ix^2) /
        (Ix^2 + Iy^2 + eps^2)^(3/2)
```

This is **level-set curvature of luminance contours**, not Gaussian curvature of the graph `z = I(x,y)`.

The principal corpus benchmark uses longest-side resolution 512 px and scales `sigma = {1, 2, 4, 8}`.

---

## Phase I — corpus ablation

Open:

```text
notebooks/01_corpus_ablation_colab.ipynb
```

Phase I establishes the initial signal by comparing edges, compact GLCM texture, multiscale curvature, structure-tensor geometry, and their combination on the fixed training/validation split.

| Experiment | Features |
|---|---|
| E1 | Gradient and edge descriptors |
| E2 | Compact GLCM texture |
| E3 | Multiscale curvature only |
| E4 | Curvature + structure-tensor geometry |
| E5 | Edge + texture baseline |
| E6 | Edge + texture + curvature + orientation |

The principal quantity is

```text
Delta macro-F1 = macro-F1(E6) - macro-F1(E5)
```

Macro-F1 uncertainty is estimated with a class-stratified bootstrap and the E6-E5 increment with a paired class-stratified bootstrap.

---

## Phase II — leakage audit and stronger controlled baseline

Open directly in Colab:

https://colab.research.google.com/github/ardominguezm/painting-geometry/blob/multiscale-corpus-analysis/notebooks/02_phase2_leakage_strong_baseline_colab.ipynb

Phase II addresses the main reviewer-facing controls after a positive Phase-I result. It **reuses the 512 px Phase-I curvature matrices** rather than recomputing them.

It performs four controls:

1. **Cross-split leakage audit.** SHA1, pHash, and dHash identify exact and perceptually similar train/validation images. Main clean evaluation excludes exact matches or pairs satisfying both conservative pHash and dHash criteria.
2. **Stronger conventional baseline.** The legacy image-wise 75th-percentile edge density is not used as the main Phase-II edge descriptor because it is nearly constant by construction. The replacement baseline includes multiscale gradient statistics, non-degenerate edge densities, HOG-like orientation summaries, multi-distance GLCM descriptors, and uniform LBP histograms.
3. **Training-only model selection.** RBF-SVM `C` and `gamma` are selected by stratified cross-validation using only training data.
4. **Controlled dimensionality.** Conventional and geometry-augmented models are compared after training-only `SelectKBest` with the same requested dimensionality (`k=40` by default).

Principal contrasts:

```text
BG_combined_full - B_strong_full
BG_combined_k40  - B_strong_k40
```

### Phase-II scripts

```text
scripts/audit_near_duplicates.py
scripts/extract_strong_baseline_features.py
scripts/run_phase2_experiments.py
```

---

## Phase III — geometry interpretation and robustness

Open directly in Colab:

https://colab.research.google.com/github/ardominguezm/painting-geometry/blob/multiscale-corpus-analysis/notebooks/03_phase3_geometry_interpretation_colab.ipynb

Phase III deliberately stops optimizing classification performance and asks what the geometric signal means computationally.

It contains four analyses.

### 1. Scale anatomy

The existing Phase-I 512 px feature matrices are reused to compare:

```text
sigma = 1
sigma = 2
sigma = 4
sigma = 8
{1,2}, {2,4}, {4,8}
{1,2,4}, {2,4,8}
{1,2,4,8}
{1,2,4,8} + orientation
```

Single-scale models have the same number of curvature variables. Hyperparameters are selected using training CV only. The best single scale is selected by **training CV**, not validation performance.

Script:

```text
scripts/run_scale_ablation.py
```

### 2. Resolution robustness with scale-normalized curvature

Resolution robustness is not evaluated by naively using the same smoothing radius in pixels. Phase III introduces a separate publication-grade implementation in

```text
src/curvature_v2.py
```

using true derivative-of-Gaussian derivatives. For a reference scale defined at 512 px,

```text
sigma_px(R) = sigma_ref * R / 512
```

and the dimensionless scale-normalized curvature is

```text
kappa_tilde = sigma_px * kappa
```

Thus 256, 512, and 1024 px images are compared at matched **relative spatial scales**. Stability is summarized using pairwise Spearman correlations, ICC(3,1), and robust median drift.

The Phase-I implementation is preserved unchanged for reproducibility. `curvature_v2` is used only for the explicit resolution-robustness layer so earlier results are not silently redefined.

Script:

```text
scripts/run_resolution_robustness.py
```

### 3. Artist-wise geometric structure

The clean corpus is analyzed using:

- Kruskal-Wallis tests;
- Benjamini-Hochberg FDR;
- Kruskal-Wallis epsilon-squared effect size;
- pairwise Mann-Whitney tests for the strongest descriptors;
- rank-biserial pairwise effect sizes.

The emphasis is on **effect sizes and geometric profiles**, not extremely small p-values alone.

Script:

```text
scripts/analyze_artist_geometry.py
```

### 4. Positioning *The Starry Night* within Van Gogh

A reliable reproduction of *The Starry Night* can be uploaded to the Phase-III notebook and processed with the same 512 px Phase-I geometry used by the corpus benchmark.

The analysis reports:

- robust multivariate distance from the Van Gogh median geometry;
- its percentile within the clean Van Gogh reference corpus;
- nearest corpus neighbors;
- per-feature percentiles and robust z-scores;
- a 2D PCA visualization.

This is a descriptive within-corpus analysis, **not** an authenticity, emotion, perception, or intention score.

Script:

```text
scripts/position_starry_night.py
```

---

## Corpus layout

The extraction scripts expect one folder per artist:

```text
training/
├── VanGogh/
├── Monet/
├── Cezanne/
└── ...

validation/
├── VanGogh/
├── Monet/
├── Cezanne/
└── ...
```

## Phase-I command-line extraction

```bash
python scripts/extract_corpus_features.py \
  --root /path/to/training \
  --output results/features_train_multiscale.csv \
  --long-side 512 \
  --sigmas 1 2 4 8

python scripts/extract_corpus_features.py \
  --root /path/to/validation \
  --output results/features_test_multiscale.csv \
  --long-side 512 \
  --sigmas 1 2 4 8
```

Each Phase-I row contains four explicitly namespaced feature families:

- `edge__`: legacy gradient/edge descriptors;
- `texture__`: compact GLCM descriptors;
- `orient__`: structure-tensor coherence/orientation descriptors;
- `curv__`: multiscale level-set curvature summaries.

## Repository structure

```text
painting-geometry/
├── painting_curvature_field.py
├── notebooks/
│   ├── 01_corpus_ablation_colab.ipynb
│   ├── 02_phase2_leakage_strong_baseline_colab.ipynb
│   └── 03_phase3_geometry_interpretation_colab.ipynb
├── scripts/
│   ├── extract_corpus_features.py
│   ├── run_ablation.py
│   ├── audit_near_duplicates.py
│   ├── extract_strong_baseline_features.py
│   ├── run_phase2_experiments.py
│   ├── run_scale_ablation.py
│   ├── run_resolution_robustness.py
│   ├── analyze_artist_geometry.py
│   └── position_starry_night.py
├── src/
│   ├── preprocessing.py
│   ├── curvature.py
│   ├── curvature_v2.py
│   ├── orientation.py
│   ├── baselines.py
│   └── statistics.py
├── figures/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Interpretation rule

A positive classification result is treated as evidence that the specified descriptors contain artist-discriminative information, not as evidence of emotion, intention, perception, authenticity, or causal artistic mechanisms. Psychological or affective claims are outside the scope of this computational analysis.

## Current scientific logic

The project now follows the sequence:

```text
Phase I   -> Is there geometric signal?
Phase II  -> Does it survive stronger baselines, leakage control, and matched dimensionality?
Phase III -> At which scales does it live, is it resolution-stable, how does it vary by artist,
             and where does The Starry Night sit within Van Gogh?
```

Only after Phase III should the main manuscript claims and final figure set be frozen.

## Legacy analysis

The original script `painting_curvature_field.py` is retained unchanged for provenance. It computes and visualizes curvature for *The Starry Night* using the divergence of the normalized luminance gradient.
