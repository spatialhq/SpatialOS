import numpy as np
import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.postprocessing.downsample import voxel_downsample
from spatial_os.postprocessing.normals import estimate_normals
from spatial_os.postprocessing.outlier_removal import remove_outliers


def _grid_cloud(n_per_axis=10, spacing=0.05) -> o3d.geometry.PointCloud:
    xs = np.arange(n_per_axis) * spacing
    grid = np.array([[x, y, 0.0] for x in xs for y in xs])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid)
    return pcd


def test_voxel_downsample_reduces_point_count():
    pcd = _grid_cloud()
    down_small = voxel_downsample(pcd, 0.01)
    down_large = voxel_downsample(pcd, 0.2)
    assert len(down_large.points) <= len(down_small.points)
    assert len(down_large.points) <= len(pcd.points)


def test_outlier_removal_drops_injected_outliers():
    pcd = _grid_cloud()
    points = np.asarray(pcd.points)
    outliers = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
    pcd.points = o3d.utility.Vector3dVector(np.vstack([points, outliers]))

    cleaned, outlier_ratio = remove_outliers(pcd, ProcessConfig())

    assert len(cleaned.points) <= len(pcd.points)
    assert outlier_ratio > 0.0


def test_estimate_normals_produces_unit_length_normals():
    pcd = _grid_cloud()
    pcd = estimate_normals(pcd, radius=0.2, max_nn=10)
    normals = np.asarray(pcd.normals)
    assert normals.shape[0] == len(pcd.points)
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
