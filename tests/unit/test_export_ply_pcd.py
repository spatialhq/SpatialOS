import numpy as np
import open3d as o3d

from spatial_os.export.ply_pcd import write_pcd, write_ply


def _small_cloud() -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).random((50, 3)))
    return pcd


def test_write_ply_round_trip(tmp_path):
    pcd = _small_cloud()
    path = write_ply(pcd, tmp_path)
    assert path.exists()

    reread = o3d.io.read_point_cloud(str(path))
    assert len(reread.points) == len(pcd.points)


def test_write_pcd_round_trip(tmp_path):
    pcd = _small_cloud()
    path = write_pcd(pcd, tmp_path)
    assert path.exists()

    reread = o3d.io.read_point_cloud(str(path))
    assert len(reread.points) == len(pcd.points)
