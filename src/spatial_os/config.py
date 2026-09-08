from pydantic import BaseModel


class ProcessConfig(BaseModel):
    """Tunables for the `process` pipeline (preprocessing -> reconstruction -> postprocessing -> export)."""

    blur_threshold: float = 100.0
    """Laplacian variance below this is considered blurry and dropped."""

    voxel_size: float = 0.02
    """Voxel size (meters) for downsampling and TSDF integration."""

    tsdf_sdf_trunc: float = 0.04
    """TSDF truncation distance (meters)."""

    depth_max: float = 5.0
    """Max depth (meters) to trust; farther points are discarded as noise."""

    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    backend: str = "open3d"
    """'open3d' (default, no external binary) or 'colmap' (requires colmap on PATH)."""
