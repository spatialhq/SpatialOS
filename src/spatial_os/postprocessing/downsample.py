import open3d as o3d


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size)
