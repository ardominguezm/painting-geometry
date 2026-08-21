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

For luminance field `I(x,y)`, the signed curvature of its level sets is computed as

```text
kappa = (Ixx Iy^2 - 2 Ix Iy Ixy + Iyy Ix^2) /
        (Ix^2 + Iy^2 + eps^2)^(3/2)
```

This is **level-set curvature of luminance contours**, not Gaussian curvature of the graph `z = I(x,y)`.

A Gaussian scale-space is used with `sigma = {1, 2, 4, 8}`. The principal corpus benchmark uses longest-side resolution 512 px; resolution robustness is a later analysis.

## Phase I — corpus ablation

Open:

```text
notebooks/01_corpus_ablation_colab.ipynb
```

Phase I establishes the initial signal by comparing edges, compact GLCM texture, multiscale curvature, structure-tensor geometry, and their combination on the fixed training/validation split.

The first benchmark writes:

```text
results/ablation_results.csv
results/ablation_predictions.csv
results/ablation_delta.csv
```

The Phase-I experiments are:

| Experiment | Features |
|---|---|
| E1 | Gradient and edge descriptors |
| E2 | Compact GLCM texture |
| E3 | Multiscale curvature only |
| E4 | Curvature + structure-tensor geometry |
| E5 | Edge + texture baseline |
| E6 | Edge + texture + curvature + orientation |

The principal Phase-I quantity is

```text
Delta macro-F1 = macro-F1(E6) - macro-F1(E5)
```

Macro-F1 uncertainty is estimated with a class-stratified bootstrap and the E6-E5 increment with a paired class-stratified bootstrap.

## Phase II — leakage audit and stronger controlled baseline

Open directly in Colab:

https://colab.research.google.com/github/ardominguezm/painting-geometry/blob/multiscale-corpus-analysis/notebooks/02_phase2_leakage_strong_baseline_colab.ipynb

Phase II is designed to address the main methodological objections that remain after a positive Phase-I result. It **reuses the 512 px Phase-I curvature matrices** rather than recomputing them.

It performs four controls:

1. **Cross-split leakage audit.** Raw-byte SHA1, perceptual hash (pHash), and difference hash (dHash) are computed for all training and validation images. A permissive candidate list is generated for manual inspection, while the main clean evaluation excludes only exact matches or pairs satisfying both conservative pHash and dHash thresholds.
2. **Stronger conventional baseline.** The legacy image-wise 75th-percentile edge density is not used as the main Phase-II edge descriptor because it is nearly constant by construction. The replacement baseline contains multiscale gradient statistics, non-degenerate fixed-relative edge densities, HOG-like global orientation statistics, multi-distance GLCM descriptors, and uniform LBP histograms.
3. **Training-only model selection.** RBF-SVM `C` and `gamma` are selected by stratified cross-validation using only the training split. Validation images do not participate in hyperparameter selection.
4. **Controlled dimensionality.** In addition to full-feature models, the conventional baseline and geometry-augmented model are compared after `SelectKBest` inside the training pipeline with the same requested dimensionality (`k=40` by default).

The principal Phase-II comparisons are:

| Model | Meaning |
|---|---|
| `B_strong_full` | stronger conventional appearance baseline |
| `G_geometry_full` | curvature + structure-tensor geometry |
| `BG_combined_full` | strong baseline + geometry |
| `B_strong_k40` | conventional baseline after training-only selection to 40 features |
| `BG_combined_k40` | combined model after training-only selection to 40 features |

The two most important paired contrasts are:

```text
BG_combined_full - B_strong_full
BG_combined_k40  - B_strong_k40
```

Both are reported on the raw validation split and on the leakage-clean validation subset. The notebook also evaluates sensitivity to near-duplicate thresholds 0, 2, 4, and 6 without refitting the models.

### Phase-II scripts

```text
scripts/audit_near_duplicates.py
scripts/extract_strong_baseline_features.py
scripts/run_phase2_experiments.py
```

### Phase-II outputs

The notebook creates lightweight scientific outputs such as:

```text
results/phase2/phase2_results.csv
results/phase2/phase2_deltas.csv
results/phase2/phase2_metadata.csv
results/phase2/phase2_selected_features.csv
results/phase2/phase2_per_artist_f1.csv
results/phase2/leakage_threshold_sensitivity.csv
results/phase2/Figure_phase2_macroF1.png
results/phase2_leakage/leakage_audit_summary.csv
results/phase2_leakage/cross_split_near_duplicates.csv
results/phase2_leakage/near_duplicate_contact_sheet.jpg
```

Large regenerable matrices and hash caches are excluded from version control through `.gitignore`.

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
│   └── 02_phase2_leakage_strong_baseline_colab.ipynb
├── scripts/
│   ├── extract_corpus_features.py
│   ├── run_ablation.py
│   ├── audit_near_duplicates.py
│   ├── extract_strong_baseline_features.py
│   └── run_phase2_experiments.py
├── src/
│   ├── preprocessing.py
│   ├── curvature.py
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

## Next scientific stages

If the Phase-II gain remains positive after leakage cleaning, stronger baselines, hyperparameter control, and matched dimensionality, the project proceeds to resolution robustness, scale-specific ablation, artist-wise effect-size analysis, and finally positioning *The Starry Night* within Van Gogh's corpus.

## Legacy analysis

The original script `painting_curvature_field.py` is retained unchanged for provenance. It computes and visualizes curvature for *The Starry Night* using the divergence of the normalized luminance gradient.
