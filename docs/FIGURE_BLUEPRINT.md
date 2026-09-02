# Manuscript figure blueprint

This file freezes the main visual story before manuscript drafting. All quantitative panels should be regenerated from experiment outputs rather than hard-coded numbers.

## Global visual system

- Scale colors: sigma=1 teal, sigma=2 cobalt, sigma=4 ochre, sigma=8 terracotta.
- Strong conventional baseline `B`: charcoal.
- Ordinal patterns `OP`: olive.
- Level-set curvature `K`: cobalt.
- Combined representations: wine.
- Classification results: points + 95% confidence intervals, not bar charts.
- Variance decompositions: stacked bars are allowed because they represent parts of a total.
- White background, restrained axes, vector PDF/SVG masters plus 450 dpi PNG previews.

## Main figures

### Figure 1 — Multiscale level-set geometry
A. Original painting and luminance field.
B. Same crop at sigma=1,2,4,8.
C. Iso-luminance contours.
D. Scale-normalized level-set curvature maps.
E. Pipeline: image -> luminance -> smoothing -> curvature -> summary descriptors -> prediction/organization analyses.

### Figure 2 — Artist-level complementary information
A. Clean Phase-II macro-F1 for strong baseline, geometry, and combined models.
B. Paired delta macro-F1 for combined minus baseline, full and matched-k.
C. Per-artist improvement.
D. Compact leakage-audit inset.

Primary files: `phase2_results.csv`, `phase2_deltas.csv`, `phase2_per_artist_f1.csv`, `phase2_metadata.csv`.

### Figure 3 — Discrimination versus resolution robustness
A. Artist discrimination across single and multiscale curvature representations.
B. Multiscale and orientation increments.
C. ICC(3,1) distribution by reference scale.
D. Discrimination-versus-robustness map, one point per scale.

Primary files: `scale_ablation_results.csv`, `scale_ablation_deltas.csv`, `resolution_robustness_summary.csv`.

### Figure 4 — Style information generalizes to unseen artists
A. Artist-disjoint evaluation schematic.
B. ArtBench-10 baseline/geometry/combined performance.
C. WikiArt-8 source-controlled performance.
D. Single-scale and multiscale geometry hierarchy.
E. Scale-specific increments over the strong baseline.

Primary files: `artbench_artist_disjoint_results.csv`, `artbench_artist_disjoint_deltas.csv`, `phase4b_scale_hierarchy_results.csv`, `phase4b_scale_hierarchy_deltas.csv`.

### Figure 5 — Ordinal patterns versus differential geometry
A. Conceptual distinction between tie-aware local ordinal structure and multiscale level-set geometry.
B. HC, OP11, OP24, OP75, K40, and OP75+K40 in ArtBench-10 and WikiArt-8.
C. Pre-specified curvature increments, including matched-dimension controls.
D. Strict WikiArt-8 test after strong appearance descriptors plus OP75 are already present.

Primary files: `phase6_head_to_head_results.csv`, `phase6_head_to_head_deltas.csv`.

Generator: `scripts/figures/make_figure5_ordinal_vs_geometry.py`.

### Figure 6 — Style organization, heterogeneity, and source sensitivity
A. Nested style / artist-within-style / residual variance fractions across scale.
B. Style eta-squared among single-style artist centroids.
C. Mantel correlations among scale-specific style-distance matrices.
D. Source-sensitivity at sigma=2 for all10, drop Ukiyo-e, drop Surrealism, and WikiArt-8.
E. Reduced interpretable WikiArt-8 style fingerprint heatmap.

Primary files: `phase5_scale_summary.csv`, `single_style_artist_centroid_permutation_tests.csv`, `distance_matrix_scale_mantel_correlations.csv`, `phase5b_source_sensitivity_primary.csv`, `style_geometric_fingerprints.csv`.

## Supplementary figures

S1. Artist-level confusion matrices.
S2. Style-level confusion matrices.
S3. Per-artist and per-style heterogeneity.
S4. Descriptor-level effect and robustness diagnostics.
S5. Full style-distance matrices and dendrograms at sigma=1,2,4,8.
S6. Full source-sensitivity and leave-one-style-out diagnostics.
S7. The Starry Night within-Van-Gogh positioning.
S8. Ordinal-pattern sanity checks and tie-mass diagnostics.

## Production order

1. Figure 5 — direct novelty test against Tarozo-style ordinal patterns.
2. Figure 3 — scale-dependent discrimination/robustness trade-off.
3. Figure 6 — hierarchical organization and corpus sensitivity.
4. Figure 4 — style generalization.
5. Figure 2 — artist-level complementarity.
6. Figure 1 — final method schematic after the quantitative story is visually fixed.
