from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    """Resize while preserving aspect ratio so the longest side equals long_side."""
    h, w = img.shape[:2]
    scale = float(long_side) / float(max(h, w))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def luminance_bt601(img_rgb: np.ndarray) -> np.ndarray:
    """Compute normalized ITU-R BT.601 luminance in [0, 1]."""
    arr = img_rgb.astype(np.float64)
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    y -= y.min()
    denom = y.max() - y.min()
    if denom <= 0:
        return np.zeros_like(y, dtype=np.float64)
    return y / denom


def preprocess(path: str | Path, long_side: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB image and normalized luminance field."""
    rgb = load_rgb(path)
    if long_side is not None:
        rgb = resize_long_side(rgb, long_side)
    return rgb, luminance_bt601(rgb)


def multiresolution_luminance(
    path: str | Path,
    long_sides: Iterable[int] = (256, 512, 1024),
) -> dict[int, np.ndarray]:
    """Return normalized luminance fields at several spatial resolutions."""
    rgb = load_rgb(path)
    return {r: luminance_bt601(resize_long_side(rgb, r)) for r in long_sides}
