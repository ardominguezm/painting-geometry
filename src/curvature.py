from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from scipy.stats import entropy, skew


@dataclass(frozen=True)
class CurvatureScaleResult:
    sigma: float
    curvature: np.ndarray
    grad_mag: np.ndarray
    valid_mask: np.ndarray


def _gaussian_derivatives(I: np.ndarray, sigma: float) -> tuple[np.ndarray, ...]:
    """Compute first and second Gaussian derivatives of a luminance field."""
    I = np.asarray(I, dtype=np.float64)
    ksize = max(3, int(2 * np.ceil(3 * sigma) + 1))
    smooth = cv2.GaussianBlur(I, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    Ix = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy, Ixx, Iyy, Ixy


def level_set_curvature(
    I: np.ndarray,
    sigma: float,
    eps: float = 1e-8,
    grad_quantile: float = 0.20,
) -> CurvatureScaleResult:
    """Compute signed curvature of luminance level sets at Gaussian scale sigma.

    kappa = (Ixx Iy^2 - 2 Ix Iy Ixy + Iyy Ix^2) /
            (Ix^2 + Iy^2 + eps^2)^(3/2)

    A gradient-based mask excludes numerically unstable nearly-flat regions.
    """
    Ix, Iy, Ixx, Iyy, Ixy = _gaussian_derivatives(I, sigma)
    grad2 = Ix * Ix + Iy * Iy
    grad_mag = np.sqrt(grad2)
    denom = np.power(grad2 + eps * eps, 1.5)
    kappa = (Ixx * Iy * Iy - 2.0 * Ix * Iy * Ixy + Iyy * Ix * Ix) / denom

    finite = np.isfinite(kappa) & np.isfinite(grad_mag)
    positive_grad = grad_mag[finite & (grad_mag > 0)]
    if positive_grad.size:
        threshold = np.quantile(positive_grad, grad_quantile)
        valid = finite & (grad_mag >= threshold)
    else:
        valid = finite

    return CurvatureScaleResult(sigma=sigma, curvature=kappa, grad_mag=grad_mag, valid_mask=valid)


def _hist_entropy(x: np.ndarray, bins: int = 128) -> float:
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=False)
    hist = hist[hist > 0]
    return float(entropy(hist)) if hist.size else np.nan


def summarize_curvature(result: CurvatureScaleResult) -> dict[str, float]:
    """Extract interpretable scalar descriptors from one curvature scale."""
    k = result.curvature[result.valid_mask]
    g = result.grad_mag[result.valid_mask]
    if k.size == 0:
        return {key: np.nan for key in [
            "median_abs", "p75_abs", "p90_abs", "p95_abs", "mad_abs",
            "entropy_signed", "skew_signed", "positive_fraction",
            "negative_fraction", "grad_weighted_abs",
        ]}

    abs_k = np.abs(k)
    med = np.median(abs_k)
    mad = np.median(np.abs(abs_k - med))
    weighted = np.sum(abs_k * g) / (np.sum(g) + 1e-12)

    return {
        "median_abs": float(med),
        "p75_abs": float(np.percentile(abs_k, 75)),
        "p90_abs": float(np.percentile(abs_k, 90)),
        "p95_abs": float(np.percentile(abs_k, 95)),
        "mad_abs": float(mad),
        "entropy_signed": _hist_entropy(k),
        "skew_signed": float(skew(k, bias=False, nan_policy="omit")) if k.size > 2 else np.nan,
        "positive_fraction": float(np.mean(k > 0)),
        "negative_fraction": float(np.mean(k < 0)),
        "grad_weighted_abs": float(weighted),
    }


def multiscale_curvature_features(
    I: np.ndarray,
    sigmas: Iterable[float] = (1.0, 2.0, 4.0, 8.0),
    eps: float = 1e-8,
    grad_quantile: float = 0.20,
) -> tuple[dict[str, float], dict[float, CurvatureScaleResult]]:
    """Compute curvature features over several Gaussian scales."""
    features: dict[str, float] = {}
    maps: dict[float, CurvatureScaleResult] = {}
    for sigma in sigmas:
        result = level_set_curvature(I, sigma=sigma, eps=eps, grad_quantile=grad_quantile)
        maps[float(sigma)] = result
        summary = summarize_curvature(result)
        tag = str(sigma).replace(".", "p")
        for name, value in summary.items():
            features[f"kappa_s{tag}_{name}"] = value
    return features, maps
