# Phase V — Multiscale geometric organization of artistic styles

Phase V reuses the 4,000-image ArtBench-10 pilot feature matrix generated in Phase IV. It does not re-download ArtBench and does not recompute image features.

## Scientific question

The objective is to move beyond classification accuracy and quantify how the level-set geometry is organized across:

- artistic style;
- artist;
- individual painting;
- spatial scale.

The phase is run for both:

1. `artbench10_all`;
2. `artbench10_wikiart8`, excluding `surrealism` and `ukiyo_e` as the source-confound sensitivity analysis introduced in Phase IV.

## Analyses

### 1. Multiscale geometric fingerprints

For every curvature feature, image values are robustly standardized over the current dataset using the global median and MAD (with IQR/std fallbacks).

For each style, scale and curvature summary, Phase V reports the median robust z-score. The resulting matrix is an interpretable style fingerprint across

`reference sigma = {1, 2, 4, 8}`.

These are descriptive geometric profiles, not aesthetic scores.

### 2. Style distances and dendrograms

At each scale, a style centroid is the median robust-z feature vector of all images carrying that style label.

Style-to-style distance is the RMS Euclidean distance between centroids:

`d(a,b) = ||z_a - z_b||_2 / sqrt(p)`,

where `p=10` curvature summaries per scale.

Ward dendrograms are generated from the same centroid vectors.

**Guardrail:** these dendrograms represent similarity in the selected luminance level-set geometry only. They must not be interpreted as historical descent, chronology, influence, or canonical art-historical taxonomy.

### 3. Reorganization of style geometry across scales

For each pair of scales, the upper triangles of the two style-distance matrices are compared with a Spearman correlation.

A Mantel-style permutation test permutes style labels in one matrix to obtain the null distribution. This tests whether the relative arrangement of styles is preserved or reorganized as scale changes.

### 4. Nested style → artist → painting variance decomposition

ArtBench contains some artists whose images are assigned to more than one style. A strict nested decomposition is therefore performed only on artists with **one observed style label** in the current subset.

For each curvature descriptor,

`SS_total = SS_style + SS_artist(style) + SS_painting`.

Phase V reports:

- `style_fraction`;
- `artist_within_style_fraction`;
- `painting_residual_fraction`;
- `style_share_of_between_artist_variation`.

This is a descriptive sum-of-squares decomposition and does not imply causality.

### 5. Equal-weight artist-centroid permutation test

The same single-style artist subset is used to create one median geometric centroid per artist at each scale.

A multivariate Euclidean pseudo-F statistic tests whether these artist centroids are organized by style. Style labels are permuted across artist centroids, so the inferential unit is the artist rather than the painting.

Default:

`4,999 permutations per scale`.

## Primary outputs

For each dataset:

- `style_geometric_fingerprints.csv`
- `style_centroids_by_scale.csv`
- `style_distance_matrix_sigma{1,2,4,8}.csv`
- `style_pair_distances_by_scale.csv`
- `within_style_artist_dispersion_by_scale.csv`
- `distance_matrix_scale_mantel_correlations.csv`
- `nested_single_style_variance_partition_features.csv`
- `nested_single_style_variance_partition_scale_summary.csv`
- `single_style_artist_centroid_permutation_tests.csv`

Main figures include:

- multiscale style fingerprint heatmap;
- one style-distance matrix per scale;
- one geometry-derived dendrogram per scale;
- scale-to-scale distance-matrix correlation heatmap;
- nested style/artist/painting variance partition;
- artist-centroid style effect by scale.

## Interpretation rule

Phase V is intended to answer whether style-level geometric organization changes with spatial scale.

A result should be described in terms of:

- separation of style centroids;
- stability/reorganization of the distance geometry;
- style-vs-artist variance allocation;
- artist-level permutation evidence.

Do not infer artistic influence, chronology, aesthetic quality, authenticity, intention, emotion, or perceptual response from these outputs.

## Colab

`notebooks/07_phase5_artbench_style_geometry_colab.ipynb`
