"""Orchestrates preprocessing -> reconstruction -> postprocessing -> export -> scoring
for a single canonical session directory. This is what `spatial-os process` calls."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.export.colmap_format import write_colmap_format
from spatial_os.export.metadata_json import write_metadata
from spatial_os.export.ply_pcd import write_pcd, write_ply
from spatial_os.export.rosbag import write_rosbag
from spatial_os.postprocessing.downsample import voxel_downsample
from spatial_os.postprocessing.normals import estimate_normals
from spatial_os.postprocessing.outlier_removal import remove_outliers
from spatial_os.preprocessing.unpack import prepare_frames
from spatial_os.quality.scoring import QualityReport, score_session
from spatial_os.schema import SessionManifest, SessionMetadataOverride
from spatial_os.utils.io_paths import ensure_dir, manifest_path, metadata_override_path
from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)

VALID_FORMATS = {"ply", "pcd", "rosbag", "colmap"}


class ProcessResult:
    def __init__(self, output_dir: Path, quality: QualityReport, written_files: list[Path]):
        self.output_dir = output_dir
        self.quality = quality
        self.written_files = written_files


def run_process(
    session_dir: str | Path,
    output_dir: str | Path,
    formats: list[str],
    cfg: ProcessConfig | None = None,
) -> ProcessResult:
    cfg = cfg or ProcessConfig()
    session_dir = Path(session_dir)
    output_dir = ensure_dir(output_dir)

    unknown = set(formats) - VALID_FORMATS
    if unknown:
        raise ValueError(f"unknown export format(s): {sorted(unknown)}; valid: {sorted(VALID_FORMATS)}")

    manifest = SessionManifest.load(manifest_path(session_dir))
    logger.info("loaded session %s (%d frames)", manifest.session_id, len(manifest.frames))

    prepared = prepare_frames(manifest, session_dir, cfg)

    if cfg.backend == "colmap":
        from spatial_os.reconstruction.colmap_backend import reconstruct as backend_reconstruct
    else:
        from spatial_os.reconstruction.open3d_backend import reconstruct as backend_reconstruct

    pcd = backend_reconstruct(manifest, prepared, cfg)
    logger.info("reconstruction produced %d points", len(pcd.points))

    pcd, outlier_ratio = remove_outliers(pcd, cfg)
    pcd = voxel_downsample(pcd, cfg.voxel_size)
    pcd = estimate_normals(pcd)
    logger.info("post-processing: %d points remain (outlier_ratio=%.3f)", len(pcd.points), outlier_ratio)

    override_path = metadata_override_path(session_dir)
    override = SessionMetadataOverride.load(override_path) if override_path.exists() else None

    frames = [pf.frame for pf in prepared]
    quality = score_session(manifest, frames, pcd, outlier_ratio, override=override)
    logger.info("quality score: %.1f (%s)", quality.total_score, quality.verdict)

    written: list[Path] = []
    if "ply" in formats:
        written.append(write_ply(pcd, output_dir))
    if "pcd" in formats:
        written.append(write_pcd(pcd, output_dir))
    if "rosbag" in formats:
        written.append(write_rosbag(pcd, frames, output_dir))
    if "colmap" in formats:
        written.extend(write_colmap_format(manifest, frames, pcd, output_dir))

    written.append(write_metadata(manifest, quality, output_dir))

    return ProcessResult(output_dir=output_dir, quality=quality, written_files=written)
