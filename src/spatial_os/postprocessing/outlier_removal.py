import open3d as o3d

from spatial_os.config import ProcessConfig


def remove_outliers(
    pcd: o3d.geometry.PointCloud, cfg: ProcessConfig
) -> tuple[o3d.geometry.PointCloud, float]:
    """Statistical outlier removal. Returns (cleaned_cloud, outlier_ratio)."""
    n_before = len(pcd.points)
    cleaned, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.outlier_nb_neighbors, std_ratio=cfg.outlier_std_ratio
    )
    n_after = len(cleaned.points)
    outlier_ratio = 0.0 if n_before == 0 else 1.0 - (n_after / n_before)
    return cleaned, outlier_ratio
