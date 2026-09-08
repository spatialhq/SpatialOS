from pathlib import Path

import open3d as o3d


def write_ply(pcd: o3d.geometry.PointCloud, out_dir: str | Path, name: str = "cloud") -> Path:
    path = Path(out_dir) / f"{name}.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def write_pcd(pcd: o3d.geometry.PointCloud, out_dir: str | Path, name: str = "cloud") -> Path:
    path = Path(out_dir) / f"{name}.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path
