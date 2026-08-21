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

A Gaussian scale-space is used with initial scales `sigma = {1, 2, 4, 8}` and the analysis will be repeated at several image resolutions.

## Planned ablation study

| Experiment | Features |
|---|---|
| E0 | Random / majority baseline |
| E1 | Gradient and edge descriptors |
| E2 | GLCM texture |
| E3 | Multiscale curvature only |
| E4 | Curvature + structure-tensor geometry |
| E5 | Texture + edges + geometry |

The key quantity is the out-of-sample improvement in macro-F1 of E5 relative to the conventional baseline E1+E2.

## Repository structure

```text
painting-geometry/
├── painting_curvature_field.py      # original single-painting analysis (preserved)
├── src/
│   ├── preprocessing.py             # image loading, BT.601 luminance, resolutions
│   ├── curvature.py                 # multiscale level-set curvature
│   ├── orientation.py               # structure tensor and orientation coherence
│   ├── baselines.py                 # edge and GLCM baselines
│   └── statistics.py                # Kruskal-Wallis, FDR, bootstrap, stability
├── figures/
└── README.md
```

## Immediate milestone

Produce a reproducible comparison of:

```text
Edges | Texture | Curvature | Geometry | Texture + Geometry
```

using the same train/test split and reporting macro-F1 with uncertainty. If geometry adds stable information, the analysis proceeds to corpus-level inference and a dedicated *Starry Night* outlier analysis.

## Legacy analysis

The original script `painting_curvature_field.py` is retained unchanged for provenance. It computes and visualizes curvature for *The Starry Night* using the divergence of the normalized luminance gradient.
