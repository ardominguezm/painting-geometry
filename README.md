# Multiscale Geometry of Paintings

This repository develops a corpus-scale, reproducible analysis of **luminance level-set geometry in paintings**. The current research branch moves beyond a single-image reading of *The Starry Night* and tests whether multiscale curvature provides artist-specific information beyond conventional edge and texture descriptors.

## Scientific question

Does multiscale level-set curvature provide independent, interpretable information about artistic style once conventional edge and texture structure are controlled for?

The central hypotheses are:

1. curvature statistics differ systematically across artists;
2. curvature adds information beyond edge and GLCM texture baselines;
3. geometric descriptors are stable across image resolution and smoothing scale;
4. *The Starry Night* can be located quantitatively within the distribution of Van Gogh's oeuvre rather than treated as an isolated exemplar.

## Geometry

For luminance field `I(x,y)`, the signed curvature of its level sets is computed as

```text
kappa = (Ixx Iy^2 - 2 Ix Iy Ixy + Iyy Ix^2) /
        (Ix^2 + Iy^2 + eps^2)^(3/2)
```

This is **level-set curvature of luminance contours**, not Gaussian curvature of the graph `z = I(x,y)`.

A Gaussian scale-space is used with initial scales `sigma = {1, 2, 4, 8}`. The main benchmark starts at longest-side resolution 512 px; robustness will then be checked at 256 and 1024 px.

## Corpus layout

The extraction script expects one folder per artist. The existing Kaggle corpus used in the exploratory notebook follows this pattern.

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

## Step 1 — extract features

From the repository root:

```bash
python scripts/extract_corpus_features.py \
  --root /path/to/training \
  --output results/features_train_geometry.csv \
  --long-side 512 \
  --sigmas 1 2 4 8

python scripts/extract_corpus_features.py \
  --root /path/to/validation \
  --output results/features_test_geometry.csv \
  --long-side 512 \
  --sigmas 1 2 4 8
```

Each output row corresponds to one painting and contains four feature families identified by prefixes:

- `edge__`: gradient magnitude and edge-density baselines;
- `texture__`: GLCM texture descriptors;
- `orient__`: structure-tensor coherence/orientation descriptors;
- `curv__`: multiscale level-set curvature summaries.

## Step 2 — ablation experiment

```bash
python scripts/run_ablation.py \
  --train results/features_train_geometry.csv \
  --test results/features_test_geometry.csv \
  --output results/ablation_results.csv
```

The first benchmark uses the same fixed train/validation partition as the exploratory notebook and an RBF-SVM with standardized features. Macro-F1 is accompanied by a nonparametric bootstrap 95% confidence interval.

## Ablation study

| Experiment | Features |
|---|---|
| E1 | Gradient and edge descriptors |
| E2 | GLCM texture |
| E3 | Multiscale curvature only |
| E4 | Curvature + structure-tensor geometry |
| E5 | Edge + texture baseline |
| E6 | Edge + texture + curvature + orientation |

The principal quantity is

```text
Delta macro-F1 = macro-F1(E6) - macro-F1(E5)
```

A positive and stable gain motivates the interpretation that geometry contributes information not already contained in conventional edge/texture descriptors. If the gain is negligible, the project moves toward intra-artist geometric characterization and outlier analysis rather than claiming improved artist classification.

## Repository structure

```text
painting-geometry/
├── painting_curvature_field.py      # original single-painting analysis (preserved)
├── scripts/
│   ├── extract_corpus_features.py   # corpus-wide feature extraction
│   └── run_ablation.py              # controlled feature-family comparison
├── src/
│   ├── preprocessing.py             # image loading, BT.601 luminance, resolutions
│   ├── curvature.py                 # multiscale level-set curvature
│   ├── orientation.py               # structure tensor and orientation coherence
│   ├── baselines.py                 # edge and GLCM baselines
│   └── statistics.py                # Kruskal-Wallis, FDR, bootstrap, stability
├── figures/
├── results/
└── README.md
```

## Immediate milestone

Produce and inspect the following table before drafting the manuscript:

```text
experiment | n_features | accuracy | macro_f1 | 95% CI
```

The go/no-go result is whether E6 improves meaningfully over E5. The next stages are resolution robustness, artist-wise inferential statistics, and finally positioning *The Starry Night* within Van Gogh's corpus.

## Legacy analysis

The original script `painting_curvature_field.py` is retained unchanged for provenance. It computes and visualizes curvature for *The Starry Night* using the divergence of the normalized luminance gradient.
