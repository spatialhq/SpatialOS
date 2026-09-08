import numpy as np
import open3d as o3d

from spatial_os.export.colmap_format import write_colmap_format
from spatial_os.schema import DepthSource, DeviceInfo, Frame, Intrinsics, SessionManifest


def _manifest_and_frames():
    intrinsics = Intrinsics(width=64, height=48, fx=50, fy=50, cx=32, cy=24)
    frames = [
        Frame(
            frame_id=str(i),
            timestamp=float(i),
            rgb_path=f"frames/{i}.png",
            depth_path=f"frames/{i}_depth.png",
            pose=[[1, 0, 0, i * 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        for i in range(3)
    ]
    manifest = SessionManifest(
        session_id="s",
        device=DeviceInfo(model="test", depth_source=DepthSource.LIDAR),
        intrinsics=intrinsics,
        frames=frames,
    )
    return manifest, frames


def test_write_colmap_format_creates_expected_files(tmp_path):
    manifest, frames = _manifest_and_frames()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).random((30, 3)))

    written = write_colmap_format(manifest, frames, pcd, tmp_path)

    names = {p.name for p in written}
    assert names == {"cameras.txt", "images.txt", "points3D.txt", "fused.ply"}
    for p in written:
        assert p.exists()

    cameras_text = (tmp_path / "colmap" / "sparse" / "cameras.txt").read_text()
    assert "PINHOLE 64 48" in cameras_text

    images_text = (tmp_path / "colmap" / "sparse" / "images.txt").read_text()
    for frame in frames:
        expected_name = frame.rgb_path.split("/")[-1]
        assert expected_name in images_text

    points_text = (tmp_path / "colmap" / "sparse" / "points3D.txt").read_text()
    # header (2 lines) + one line per point
    assert points_text.strip().count("\n") + 1 - 2 == len(pcd.points)
