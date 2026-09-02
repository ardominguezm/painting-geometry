from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def _sobel_fields(I: np.ndarray, sigma: float = 0.0):
    I = np.asarray(I, dtype=np.float64)
    if sigma > 0:
        ksize = max(3, int(2 * np.ceil(3 * sigma) + 1))
        I = cv2.GaussianBlur(I, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Ix * Ix + Iy * Iy)
    return Ix, Iy, mag


def edge_features(I: np.ndarray) -> dict[str, float]:
    """Legacy Phase-I edge descriptors.

    ``edge_density`` is retained only for reproducibility of Phase I. Because its
    threshold is the image-wise 75th percentile, it is almost constant by
    construction and should not be used as the principal edge-density baseline
    in Phase II.
    """
    _, _, mag = _sobel_fields(I)
    finite = mag[np.isfinite(mag)]
    if finite.size == 0:
        return {"grad_mean": np.nan, "grad_std": np.nan, "grad_p90": np.nan, "edge_density": np.nan}
    threshold = np.percentile(finite, 75)
    return {
        "grad_mean": float(np.mean(finite)),
        "grad_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "grad_p90": float(np.percentile(finite, 90)),
        "edge_density": float(np.mean(finite >= threshold)),
    }


def glcm_features(I: np.ndarray, levels: int = 32) -> dict[str, float]:
    """Legacy compact GLCM baseline averaged over four directions."""
    I = np.clip(np.asarray(I, dtype=np.float64), 0, 1)
    q = np.floor(I * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    return {f"glcm_{p}": float(np.nanmean(graycoprops(glcm, p))) for p in props}


def multiscale_gradient_features(
    I: np.ndarray,
    sigmas: Iterable[float] = (1.0, 2.0, 4.0, 8.0),
    relative_thresholds: Iterable[float] = (0.10, 0.20, 0.40),
) -> dict[str, float]:
    """Multiscale gradient summaries and non-degenerate edge densities.

    Edge densities use fixed thresholds after normalizing gradient magnitude by
    its image-wise 99th percentile. Unlike the legacy image-wise quantile rule,
    these densities are not fixed by construction.
    """
    out: dict[str, float] = {}
    thresholds = tuple(relative_thresholds)
    for sigma in sigmas:
        _, _, mag = _sobel_fields(I, sigma=float(sigma))
        finite = mag[np.isfinite(mag)]
        tag = str(float(sigma)).replace(".", "p")
        if finite.size == 0:
            for name in ["mean", "std", "median", "p75", "p90", "p95"]:
                out[f"grad_s{tag}_{name}"] = np.nan
            for tau in thresholds:
                ttag = str(float(tau)).replace(".", "p")
                out[f"edge_density_s{tag}_t{ttag}"] = np.nan
            continue

        out[f"grad_s{tag}_mean"] = float(np.mean(finite))
        out[f"grad_s{tag}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        out[f"grad_s{tag}_median"] = float(np.median(finite))
        out[f"grad_s{tag}_p75"] = float(np.percentile(finite, 75))
        out[f"grad_s{tag}_p90"] = float(np.percentile(finite, 90))
        out[f"grad_s{tag}_p95"] = float(np.percentile(finite, 95))

        scale = float(np.percentile(finite, 99)) + 1e-12
        normalized = np.clip(finite / scale, 0.0, 1.0)
        for tau in thresholds:
            ttag = str(float(tau)).replace(".", "p")
            out[f"edge_density_s{tag}_t{ttag}"] = float(np.mean(normalized >= tau))
    return out


def orientation_histogram_features(I: np.ndarray, sigma: float = 2.0, bins: int = 9) -> dict[str, float]:
    """Global HOG-like orientation distribution weighted by gradient magnitude."""
    Ix, Iy, mag = _sobel_fields(I, sigma=sigma)
    theta = np.mod(np.arctan2(Iy, Ix), np.pi)
    valid = np.isfinite(theta) & np.isfinite(mag)
    theta = theta[valid]
    weights = mag[valid]

    if theta.size == 0 or np.sum(weights) <= 0:
        out = {f"ori_bin_{i:02d}": np.nan for i in range(bins)}
        out["ori_entropy_norm"] = np.nan
        out["ori_resultant"] = np.nan
        return out

    hist, edges = np.histogram(theta, bins=bins, range=(0.0, np.pi), weights=weights)
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-12
    out = {f"ori_bin_{i:02d}": float(v) for i, v in enumerate(hist)}
    positive = hist[hist > 0]
    out["ori_entropy_norm"] = float(entropy(positive) / np.log(bins)) if positive.size else np.nan

    c = float(np.sum(weights * np.cos(2.0 * theta)))
    s = float(np.sum(weights * np.sin(2.0 * theta)))
    out["ori_resultant"] = float(np.sqrt(c * c + s * s) / (np.sum(weights) + 1e-12))
    return out


def multidistance_glcm_features(
    I: np.ndarray,
    levels: int = 32,
    distances: Iterable[int] = (1, 2, 4),
) -> dict[str, float]:
    """GLCM descriptors at several pixel distances, averaged over four angles."""
    I = np.clip(np.asarray(I, dtype=np.float64), 0, 1)
    q = np.floor(I * (levels - 1)).astype(np.uint8)
    distances = tuple(int(d) for d in distances)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        q,
        distances=list(distances),
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    out: dict[str, float] = {}
    for prop in props:
        values = graycoprops(glcm, prop)  # shape: n_distances x n_angles
        for i, d in enumerate(distances):
            out[f"glcm_d{d}_{prop}"] = float(np.nanmean(values[i]))
    return out


def lbp_features(
    I: np.ndarray,
    configs: Iterable[tuple[int, float]] = ((8, 1.0), (16, 2.0)),
) -> dict[str, float]:
    """Rotation-robust uniform LBP histograms at two local scales."""
    I = np.clip(np.asarray(I, dtype=np.float64), 0, 1)
    gray = np.round(I * 255).astype(np.uint8)
    out: dict[str, float] = {}
    for P, R in configs:
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), range=(0, n_bins))
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-12
        rtag = str(float(R)).replace(".", "p")
        for i, value in enumerate(hist):
            out[f"lbp_p{P}_r{rtag}_bin_{i:02d}"] = float(value)
    return out


def strong_baseline_features(I: np.ndarray) -> dict[str, float]:
    """Phase-II conventional appearance baseline.

    The descriptor intentionally excludes level-set curvature and structure-tensor
    coherence. It combines multiscale gradients/edge density, global orientation,
    multi-distance GLCM texture, and uniform LBP histograms.
    """
    out: dict[str, float] = {}
    out.update(multiscale_gradient_features(I))
    out.update(orientation_histogram_features(I))
    out.update(multidistance_glcm_features(I))
    out.update(lbp_features(I))
    return out
