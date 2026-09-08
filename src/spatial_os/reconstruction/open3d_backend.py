"""Default reconstruction backend: pose-driven RGBD TSDF fusion.

LiDAR-equipped iPhones already provide camera pose via ARKit, so unlike a
from-scratch COLMAP pipeline we don't need Structure-from-Motion to recover
poses -- we only need dense fusion, which Open3D's TSDF integration does
directly from RGBD + known pose, with zero external binary dependency.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.unpack import PreparedFrame
from spatial_os.schema import SessionManifest
from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)


def _to_o3d_intrinsic(manifest: SessionManifest) -> o3d.camera.PinholeCameraIntrinsic:
    intr = manifest.intrinsics
    return o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)


def reconstruct(
    manifest: SessionManifest,
    frames: list[PreparedFrame],
    cfg: ProcessConfig,
) -> o3d.geometry.PointCloud:
    intrinsic = _to_o3d_intrinsic(manifest)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size,
        sdf_trunc=cfg.tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    for pf in frames:
        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(pf.rgb[:, :, ::-1]))  # BGR -> RGB
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(pf.depth_m))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,  # depth already normalized to meters upstream
            depth_trunc=cfg.depth_max,
            convert_rgb_to_intensity=False,
        )

        pose_c2w = pf.frame.pose_matrix()
        extrinsic_w2c = np.linalg.inv(pose_c2w)

        try:
            volume.integrate(rgbd, intrinsic, extrinsic_w2c)
            integrated += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to integrate frame %s: %s", pf.frame.frame_id, exc)

    if integrated == 0:
        raise ValueError("TSDF integration failed for all frames")

    logger.info("TSDF fusion: integrated %d/%d frames", integrated, len(frames))

    pcd = volume.extract_point_cloud()
    return pcd
