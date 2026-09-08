"""Optional reconstruction backend using the COLMAP CLI (Structure-from-Motion
+ Multi-View Stereo). Requires the `colmap` binary on PATH -- not supported on
bare Windows; use the `Dockerfile.colmap` image instead. See docs/colmap-tradeoffs.md.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.unpack import PreparedFrame
from spatial_os.schema import SessionManifest
from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)


class ColmapNotAvailableError(RuntimeError):
    pass


def _require_colmap() -> str:
    binary = shutil.which("colmap")
    if binary is None:
        raise ColmapNotAvailableError(
            "colmap binary not found on PATH. The colmap backend requires Docker "
            "(see Dockerfile.colmap) or a native COLMAP install; the default "
            "'open3d' backend has no external dependency -- use --backend open3d instead. "
            "See docs/colmap-tradeoffs.md."
        )
    return binary


def reconstruct(
    manifest: SessionManifest,
    frames: list[PreparedFrame],
    cfg: ProcessConfig,
) -> o3d.geometry.PointCloud:
    colmap_bin = _require_colmap()

    with tempfile.TemporaryDirectory(prefix="spatial-os-colmap-") as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        db_path = tmp_path / "database.db"
        sparse_dir = tmp_path / "sparse"
        sparse_dir.mkdir()
        dense_dir = tmp_path / "dense"
        dense_dir.mkdir()

        import cv2

        for pf in frames:
            cv2.imwrite(str(image_dir / f"{pf.frame.frame_id}.png"), pf.rgb)

        _run(colmap_bin, ["feature_extractor", "--database_path", str(db_path), "--image_path", str(image_dir)])
        _run(colmap_bin, ["exhaustive_matcher", "--database_path", str(db_path)])
        _run(
            colmap_bin,
            [
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse_dir),
            ],
        )
        _run(
            colmap_bin,
            [
                "image_undistorter",
                "--image_path",
                str(image_dir),
                "--input_path",
                str(sparse_dir / "0"),
                "--output_path",
                str(dense_dir),
            ],
        )
        _run(colmap_bin, ["patch_match_stereo", "--workspace_path", str(dense_dir)])
        fused_ply = dense_dir / "fused.ply"
        _run(
            colmap_bin,
            [
                "stereo_fusion",
                "--workspace_path",
                str(dense_dir),
                "--output_path",
                str(fused_ply),
            ],
        )

        return o3d.io.read_point_cloud(str(fused_ply))


def _run(colmap_bin: str, args: list[str]) -> None:
    cmd = [colmap_bin, *args]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
