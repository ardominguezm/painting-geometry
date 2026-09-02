from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy


@dataclass(frozen=True)
class ScaleNormalizedCurvatureResult:
    sigma_ref: float
    sigma_px: float
    curvature_px: np.ndarray
    curvature_scale_normalized: np.ndarray
    grad_mag: np.ndarray
    valid_mask: np.ndarray


def derivative_of_gaussian_fields(
    I: np.ndarray,
    sigma_px: float,
    truncate: float = 3.0,
) -> tuple[np.ndarray, ...]:
    """True first/second derivatives of a Gaussian-smoothed luminance field.

    Array axis 0 is y and axis 1 is x, hence the derivative orders below.
    Derivatives are expressed with respect to pixel coordinates.
    """
    I = np.asarray(I, dtype=np.float64)
    sigma_px = float(sigma_px)
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")

    kwargs = dict(sigma=sigma_px, mode="reflect", truncate=float(truncate))
    Ix = gaussian_filter(I, order=(0, 1), **kwargs)
    Iy = gaussian_filter(I, order=(1, 0), **kwargs)
    Ixx = gaussian_filter(I, order=(0, 2), **kwargs)
    Iyy = gaussian_filter(I, order=(2, 0), **kwargs)
    Ixy = gaussian_filter(I, order=(1, 1), **kwargs)
    return Ix, Iy, Ixx, Iyy, Ixy


def level_set_curvature_dog(
    I: np.ndarray,
    sigma_px: float,
    sigma_ref: float | None = None,
    eps: float = 1e-12,
    grad_quantile: float = 0.20,
    truncate: float = 3.0,
) -> ScaleNormalizedCurvatureResult:
    """Compute level-set curvature using derivative-of-Gaussian derivatives.

    The pixel-coordinate curvature is

        kappa = (Ixx Iy^2 - 2 Ix Iy Ixy + Iyy Ix^2)
                / (Ix^2 + Iy^2 + eps^2)^(3/2).

    For resolution comparisons, the returned scale-normalized quantity is

        kappa_tilde = sigma_px * kappa,

    which is dimensionless and is compared at matched relative smoothing scales.
    """
    Ix, Iy, Ixx, Iyy, Ixy = derivative_of_gaussian_fields(
        I, sigma_px=sigma_px, truncate=truncate
    )
    grad2 = Ix * Ix + Iy * Iy
    grad_mag = np.sqrt(grad2)
    denom = np.power(grad2 + eps * eps, 1.5)
    curvature_px = (
        Ixx * Iy * Iy
        - 2.0 * Ix * Iy * Ixy
        + Iyy * Ix * Ix
    ) / denom

    finite = np.isfinite(curvature_px) & np.isfinite(grad_mag)
    positive_grad = grad_mag[finite & (grad_mag > 0)]
    if positive_grad.size:
        threshold = np.quantile(positive_grad, float(grad_quantile))
        valid = finite & (grad_mag >= threshold)
    else:
        valid = finite

    sigma_ref = float(sigma_px if sigma_ref is None else sigma_ref)
    scale_normalized = float(sigma_px) * curvature_px
    return ScaleNormalizedCurvatureResult(
        sigma_ref=sigma_ref,
        sigma_px=float(sigma_px),
        curvature_px=curvature_px,
        curvature_scale_normalized=scale_normalized,
        grad_mag=grad_mag,
        valid_mask=valid,
    )


def _fixed_entropy_arctan(x: np.ndarray, bins: int = 128) -> float:
    """Entropy on a fixed, bounded transform so values are comparable across images."""
    if x.size == 0:
        return np.nan
    transformed = np.arctan(x)
    hist, _ = np.histogram(
        transformed,
        bins=int(bins),
        range=(-np.pi / 2.0, np.pi / 2.0),
        density=False,
    )
    hist = hist[hist > 0]
    return float(entropy(hist)) if hist.size else np.nan


def summarize_scale_normalized_curvature(
    result: ScaleNormalizedCurvatureResult,
) -> dict[str, float]:
    """Robust summaries of dimensionless scale-normalized curvature."""
    k = result.curvature_scale_normalized[result.valid_mask]
    g = result.grad_mag[result.valid_mask]
    if k.size == 0:
        return {
            key: np.nan
            for key in [
                "median_abs",
                "p75_abs",
                "p90_abs",
                "p95_abs",
                "mad_abs",
                "mean_abs",
                "median_signed",
                "positive_fraction",
                "grad_weighted_abs",
                "entropy_arctan_signed",
            ]
        }

    abs_k = np.abs(k)
    med_abs = np.median(abs_k)
    mad_abs = np.median(np.abs(abs_k - med_abs))
    weighted_abs = np.sum(abs_k * g) / (np.sum(g) + 1e-12)

    return {
        "median_abs": float(med_abs),
        "p75_abs": float(np.percentile(abs_k, 75)),
        "p90_abs": float(np.percentile(abs_k, 90)),
        "p95_abs": float(np.percentile(abs_k, 95)),
        "mad_abs": float(mad_abs),
        "mean_abs": float(np.mean(abs_k)),
        "median_signed": float(np.median(k)),
        "positive_fraction": float(np.mean(k > 0)),
        "grad_weighted_abs": float(weighted_abs),
        "entropy_arctan_signed": _fixed_entropy_arctan(k),
    }


def relative_scale_curvature_features(
    I: np.ndarray,
    long_side: int,
    sigma_refs: Iterable[float] = (1.0, 2.0, 4.0, 8.0),
    reference_long_side: int = 512,
    eps: float = 1e-12,
    grad_quantile: float = 0.20,
    truncate: float = 3.0,
    return_maps: bool = False,
):
    """Curvature features at matched *relative* scales across resolutions.

    sigma_ref is defined in pixels at reference_long_side (default 512).
    At another resolution R, sigma_px = sigma_ref * R / reference_long_side.
    """
    features: dict[str, float] = {}
    maps: dict[float, ScaleNormalizedCurvatureResult] = {}

    for sigma_ref in sigma_refs:
        sigma_ref = float(sigma_ref)
        sigma_px = sigma_ref * float(long_side) / float(reference_long_side)
        result = level_set_curvature_dog(
            I,
            sigma_px=sigma_px,
            sigma_ref=sigma_ref,
            eps=eps,
            grad_quantile=grad_quantile,
            truncate=truncate,
        )
        summary = summarize_scale_normalized_curvature(result)
        tag = str(sigma_ref).replace(".", "p")
        for name, value in summary.items():
            features[f"kappa_ref_s{tag}_{name}"] = value
        if return_maps:
            maps[sigma_ref] = result

    return (features, maps) if return_maps else features
