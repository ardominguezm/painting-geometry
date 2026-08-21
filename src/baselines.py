from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def edge_features(I: np.ndarray) -> dict[str, float]:
    """Simple edge/gradient baseline descriptors."""
    I = np.asarray(I, dtype=np.float64)
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Ix * Ix + Iy * Iy)
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
    """Compact GLCM texture baseline averaged over four directions."""
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
