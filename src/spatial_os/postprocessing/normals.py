import open3d as o3d

from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.1, max_nn: int = 30) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    except RuntimeError as exc:
        # Degenerate/near-planar local neighborhoods can make the tangent-plane
        # Delaunay step fail (e.g. a perfectly flat scanned wall); the raw
        # per-point normals from estimate_normals() are still usable, just
        # without consistent global orientation.
        logger.warning("normal orientation failed, keeping unoriented normals: %s", exc)
    return pcd
