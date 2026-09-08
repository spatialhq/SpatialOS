"""Blurry-keyframe detection via Laplacian variance (design doc: 'Filter keyframes bị blur')."""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(image: np.ndarray) -> float:
    """Higher = sharper. `image` is a grayscale or BGR uint8 array."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    return laplacian_variance(image) < threshold


def filter_blurry_frames(rgb_paths: list[str], threshold: float = 100.0) -> list[bool]:
    """Returns a keep-mask (True = sharp enough to keep) aligned to `rgb_paths`."""
    keep = []
    for path in rgb_paths:
        img = cv2.imread(path)
        if img is None:
            keep.append(False)
            continue
        keep.append(not is_blurry(img, threshold))
    return keep
