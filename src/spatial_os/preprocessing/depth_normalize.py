"""Depth unit normalization and confidence/range masking."""

from __future__ import annotations

import numpy as np


def normalize_depth_to_meters(depth_raw: np.ndarray, unit: str) -> np.ndarray:
    """Converts a raw depth array to meters (float32).

    `unit` is the unit of `depth_raw` as declared in the session manifest
    ('mm' for 16-bit millimetre PNGs as used by ARKit/most depth sensors, or
    'm' if already in meters).
    """
    depth = depth_raw.astype(np.float32)
    if unit == "mm":
        return depth / 1000.0
    if unit == "m":
        return depth
    raise ValueError(f"unknown depth unit: {unit!r} (expected 'mm' or 'm')")


def mask_invalid_depth(
    depth_m: np.ndarray, depth_max: float = 5.0, depth_min: float = 0.05
) -> np.ndarray:
    """Zeroes out depth values outside a trusted range (sensor noise / no-return)."""
    masked = depth_m.copy()
    invalid = (masked < depth_min) | (masked > depth_max) | ~np.isfinite(masked)
    masked[invalid] = 0.0
    return masked
