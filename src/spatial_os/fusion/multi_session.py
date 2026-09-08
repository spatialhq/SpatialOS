"""Global Map Fusion (design doc stage 3, stretch): merges multiple already-
reconstructed session point clouds into one scene via global registration
(RANSAC feature matching) + point-to-plane ICP refinement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

VOXEL_SIZE = 0.05


def _preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float):
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return down, fpfh


def _register_pair(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel_size: float) -> np.ndarray:
    src_down, src_fpfh = _preprocess(source, voxel_size)
    tgt_down, tgt_fpfh = _preprocess(target, voxel_size)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        voxel_size * 0.4,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result_icp.transformation


def fuse_sessions(point_clouds: list[o3d.geometry.PointCloud], voxel_size: float = VOXEL_SIZE) -> o3d.geometry.PointCloud:
    """Merges point clouds pairwise onto the first cloud's frame."""
    if not point_clouds:
        raise ValueError("no point clouds provided")

    merged = o3d.geometry.PointCloud(point_clouds[0])
    for pcd in point_clouds[1:]:
        transform = _register_pair(pcd, merged, voxel_size)
        aligned = o3d.geometry.PointCloud(pcd).transform(transform)
        merged += aligned
        merged = merged.voxel_down_sample(voxel_size)

    return merged


def load_and_fuse(ply_paths: list[str | Path], voxel_size: float = VOXEL_SIZE) -> o3d.geometry.PointCloud:
    clouds = [o3d.io.read_point_cloud(str(p)) for p in ply_paths]
    return fuse_sessions(clouds, voxel_size=voxel_size)
