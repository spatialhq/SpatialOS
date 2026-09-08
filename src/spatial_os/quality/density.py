"""Point Cloud Density Score (design doc weight: 20%): points/m^2 vs. threshold, penalized by outlier ratio."""

from __future__ import annotations

import numpy as np
import open3d as o3d

TARGET_POINTS_PER_M2 = 2000.0


def density_score(pcd: o3d.geometry.PointCloud, outlier_ratio: float) -> float:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 0.0

    xy = points[:, :2]
    extent = xy.max(axis=0) - xy.min(axis=0)
    area = max(float(extent[0] * extent[1]), 1e-6)

    density = len(points) / area
    density_component = min(density / TARGET_POINTS_PER_M2, 1.0) * 100.0

    outlier_penalty = max(1.0 - outlier_ratio / 0.2, 0.0)  # >=20% outliers -> zero credit
    return density_component * outlier_penalty
