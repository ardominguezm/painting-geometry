from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy


def structure_tensor_features(I: np.ndarray, sigma: float = 2.0) -> dict[str, float]:
    """Compute interpretable structure-tensor coherence/orientation summaries."""
    I = np.asarray(I, dtype=np.float64)
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

    J11 = gaussian_filter(Ix * Ix, sigma)
    J22 = gaussian_filter(Iy * Iy, sigma)
    J12 = gaussian_filter(Ix * Iy, sigma)

    delta = np.sqrt((J11 - J22) ** 2 + 4.0 * J12 ** 2)
    lam1 = 0.5 * (J11 + J22 + delta)
    lam2 = 0.5 * (J11 + J22 - delta)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-12)

    theta = 0.5 * np.arctan2(2.0 * J12, J11 - J22)
    weights = np.sqrt(Ix * Ix + Iy * Iy)
    valid = np.isfinite(theta) & np.isfinite(coherence) & np.isfinite(weights)
    theta_v = theta[valid]
    coher_v = coherence[valid]
    weights_v = weights[valid]

    if theta_v.size == 0:
        return {
            "coherence_mean": np.nan,
            "coherence_std": np.nan,
            "coherence_p90": np.nan,
            "orientation_entropy": np.nan,
        }

    hist, _ = np.histogram(theta_v, bins=36, range=(-np.pi / 2, np.pi / 2), weights=weights_v)
    hist = hist[hist > 0]
    orient_entropy = float(entropy(hist)) if hist.size else np.nan

    return {
        "coherence_mean": float(np.mean(coher_v)),
        "coherence_std": float(np.std(coher_v, ddof=1)) if coher_v.size > 1 else 0.0,
        "coherence_p90": float(np.percentile(coher_v, 90)),
        "orientation_entropy": orient_entropy,
    }
