"""Loads a canonical session manifest and produces a filtered, normalized frame list ready for reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.blur_filter import filter_blurry_frames
from spatial_os.preprocessing.depth_normalize import mask_invalid_depth, normalize_depth_to_meters
from spatial_os.schema import Frame, SessionManifest
from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedFrame:
    frame: Frame
    rgb: np.ndarray
    depth_m: np.ndarray


def prepare_frames(
    manifest: SessionManifest, session_dir: str | Path, cfg: ProcessConfig
) -> list[PreparedFrame]:
    """Stage 1: unpack keyframes + depth maps, drop blurry frames, normalize depth units."""
    session_dir = Path(session_dir)
    rgb_paths = [str(session_dir / f.rgb_path) for f in manifest.frames]
    keep_mask = filter_blurry_frames(rgb_paths, threshold=cfg.blur_threshold)

    kept = sum(keep_mask)
    dropped = len(keep_mask) - kept
    logger.info("blur filter: kept %d/%d frames (dropped %d blurry)", kept, len(keep_mask), dropped)

    prepared: list[PreparedFrame] = []
    for frame, keep in zip(manifest.frames, keep_mask):
        if not keep:
            continue
        rgb = cv2.imread(str(session_dir / frame.rgb_path))
        if rgb is None:
            logger.warning("could not read RGB frame %s, skipping", frame.frame_id)
            continue

        depth_raw = cv2.imread(str(session_dir / frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            logger.warning("could not read depth frame %s, skipping", frame.frame_id)
            continue

        depth_m = normalize_depth_to_meters(depth_raw, manifest.depth_unit)
        depth_m = mask_invalid_depth(depth_m, depth_max=cfg.depth_max)

        prepared.append(PreparedFrame(frame=frame, rgb=rgb, depth_m=depth_m))

    if not prepared:
        raise ValueError("no usable frames remained after preprocessing")

    return prepared
