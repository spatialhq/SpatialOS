"""Coverage Score (design doc weight: 30%): % of scanned area with no large holes.

Estimated from the reconstructed cloud's top-down (XY) footprint, since we
don't have the mobile app's live coverage-overlay signal: project points to
a 2D grid, and score = fraction of the bounding footprint's cells that are
occupied (a low fraction implies holes/gaps in the scan).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

CELL_SIZE_M = 0.1


def coverage_score(pcd: o3d.geometry.PointCloud, cell_size: float = CELL_SIZE_M) -> float:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 0.0

    xy = points[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    extent = maxs - mins
    if np.any(extent <= 0):
        return 0.0

    grid_shape = np.maximum(np.ceil(extent / cell_size).astype(int), 1)
    total_cells = int(grid_shape[0] * grid_shape[1])
    if total_cells == 0:
        return 0.0

    cell_idx = np.floor((xy - mins) / cell_size).astype(int)
    cell_idx = np.clip(cell_idx, 0, grid_shape - 1)
    occupied = len(set(map(tuple, cell_idx)))

    return min(occupied / total_cells, 1.0) * 100.0
