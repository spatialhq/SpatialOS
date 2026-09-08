"""Writes a COLMAP sparse-model text folder (cameras.txt/images.txt/points3D.txt)
plus the dense point cloud, from data we already have (ARKit poses + the
Open3D-reconstructed cloud) -- no COLMAP binary required for this export.
NeRF training tools (nerfstudio, instant-ngp) can consume this directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from spatial_os.schema import Frame, SessionManifest
from spatial_os.utils.geometry import rotation_matrix_to_quaternion
from spatial_os.utils.io_paths import ensure_dir

MAX_DENSE_POINTS_IN_TEXT = 200_000  # points3D.txt is uncompressed text; keep it bounded


def write_colmap_format(
    manifest: SessionManifest,
    frames: list[Frame],
    pcd: o3d.geometry.PointCloud,
    out_dir: str | Path,
) -> list[Path]:
    sparse_dir = ensure_dir(Path(out_dir) / "colmap" / "sparse")
    dense_dir = ensure_dir(Path(out_dir) / "colmap" / "dense")

    cameras_path = _write_cameras_txt(manifest, sparse_dir)
    images_path = _write_images_txt(frames, sparse_dir)
    points_path = _write_points3d_txt(pcd, sparse_dir)
    dense_ply_path = dense_dir / "fused.ply"
    o3d.io.write_point_cloud(str(dense_ply_path), pcd)

    return [cameras_path, images_path, points_path, dense_ply_path]


def _write_cameras_txt(manifest: SessionManifest, sparse_dir: Path) -> Path:
    intr = manifest.intrinsics
    path = sparse_dir / "cameras.txt"
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"1 PINHOLE {intr.width} {intr.height} {intr.fx} {intr.fy} {intr.cx} {intr.cy}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_images_txt(frames: list[Frame], sparse_dir: Path) -> Path:
    path = sparse_dir / "images.txt"
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for i, frame in enumerate(frames, start=1):
        pose_c2w = frame.pose_matrix()
        world_to_cam = np.linalg.inv(pose_c2w)
        r = world_to_cam[:3, :3]
        t = world_to_cam[:3, 3]
        qx, qy, qz, qw = rotation_matrix_to_quaternion(r)
        name = Path(frame.rgb_path).name
        lines.append(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}")
        lines.append("")  # no 2D-3D correspondences tracked
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_points3d_txt(pcd: o3d.geometry.PointCloud, sparse_dir: Path) -> Path:
    path = sparse_dir / "points3D.txt"
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    if len(points) > MAX_DENSE_POINTS_IN_TEXT:
        idx = np.random.default_rng(0).choice(len(points), size=MAX_DENSE_POINTS_IN_TEXT, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None

    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for i, p in enumerate(points, start=1):
        if colors is not None:
            r, g, b = (colors[i - 1] * 255).astype(int)
        else:
            r, g, b = 128, 128, 128
        lines.append(f"{i} {p[0]} {p[1]} {p[2]} {r} {g} {b} 0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
