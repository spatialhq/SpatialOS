"""Back-projects 2D per-pixel semantic labels into 3D using known depth+pose,
then transfers those labels onto the final reconstructed point cloud via
nearest-neighbor majority vote. This is what turns a 2D AI model's output
into a per-point 3D semantic label without needing a native 3D model.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import open3d as o3d

from spatial_os.preprocessing.unpack import PreparedFrame
from spatial_os.refine.segmentation_2d import Segmenter
from spatial_os.schema import SessionManifest


def _unproject_labeled_points(
    depth_m: np.ndarray,
    class_map: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics,
    depth_max: float,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Nx3 world-space points, N class ids), subsampled by `stride` pixels."""
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    depths = depth_m[ys, xs]
    labels = class_map[ys, xs]

    valid = (depths > 0) & (depths < depth_max)
    xs, ys, depths, labels = xs[valid], ys[valid], depths[valid], labels[valid]
    if len(depths) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32)

    x_cam = (xs - intrinsics.cx) * depths / intrinsics.fx
    y_cam = (ys - intrinsics.cy) * depths / intrinsics.fy
    z_cam = depths
    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)

    points_world = (pose_c2w @ points_cam.T).T[:, :3]
    return points_world, labels.astype(np.int32)


def label_point_cloud(
    manifest: SessionManifest,
    frames: list[PreparedFrame],
    pcd: o3d.geometry.PointCloud,
    segmenter: Segmenter,
    depth_max: float = 5.0,
    max_frames: int = 20,
    pixel_stride: int = 4,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, dict[int, str]]:
    """Returns (per-point class id array aligned to `pcd.points`, id2label)."""
    sampled = frames[:: max(1, len(frames) // max_frames)][:max_frames] if frames else []

    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    id2label: dict[int, str] = {}

    for pf in sampled:
        class_map, id2label = segmenter.segment(pf.rgb)
        pts, labels = _unproject_labeled_points(
            pf.depth_m, class_map, pf.frame.pose_matrix(), manifest.intrinsics, depth_max, stride=pixel_stride
        )
        if len(pts):
            all_points.append(pts)
            all_labels.append(labels)

    target_points = np.asarray(pcd.points)
    if not all_points or len(target_points) == 0:
        return np.full(len(target_points), -1, dtype=np.int32), id2label

    source_points = np.concatenate(all_points, axis=0)
    source_labels = np.concatenate(all_labels, axis=0)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    kdtree = o3d.geometry.KDTreeFlann(source_pcd)

    result_labels = np.full(len(target_points), -1, dtype=np.int32)
    for i, p in enumerate(target_points):
        k = min(k_neighbors, len(source_points))
        _, idx, _ = kdtree.search_knn_vector_3d(p, k)
        votes = Counter(source_labels[j] for j in idx)
        result_labels[i] = votes.most_common(1)[0][0]

    return result_labels, id2label
