"""Splits semantically-labeled points into per-object instances.

There's no instance-segmentation model in the loop -- instances are derived
classically via per-semantic-class Euclidean clustering (DBSCAN), which is a
reasonable, dependency-light way to separate e.g. "chair" into individual
chairs once the AI model has already told us which points are chairs.
Structural classes (wall/floor/ceiling) are excluded since they form one
large contiguous surface, not discrete instances.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def cluster_instances(
    pcd: o3d.geometry.PointCloud,
    semantic_labels: np.ndarray,
    structural_class_ids: set[int],
    eps: float = 0.1,
    min_points: int = 20,
) -> np.ndarray:
    """Returns a per-point instance id array (-1 = unassigned/structural/noise),
    aligned to `pcd.points`."""
    points = np.asarray(pcd.points)
    instance_ids = np.full(len(points), -1, dtype=np.int32)

    next_instance_id = 0
    unique_classes = set(int(c) for c in np.unique(semantic_labels)) - structural_class_ids - {-1}

    for class_id in sorted(unique_classes):
        mask = semantic_labels == class_id
        if mask.sum() < min_points:
            continue

        class_pcd = o3d.geometry.PointCloud()
        class_pcd.points = o3d.utility.Vector3dVector(points[mask])
        labels = np.array(class_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        idx_in_full = np.where(mask)[0]
        for local_label in set(labels.tolist()) - {-1}:
            instance_ids[idx_in_full[labels == local_label]] = next_instance_id
            next_instance_id += 1

    return instance_ids
