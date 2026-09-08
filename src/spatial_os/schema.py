"""Canonical on-disk session schema.

Raw exports from scanning apps (3D Scanner App, Record3D, ARKitScenes, ...)
each use their own layout. Every ``ingest.*`` adapter converts a raw export
into this canonical schema so the rest of the pipeline never has to know
where the data originally came from.

On disk, a canonical session directory looks like::

    session_dir/
        manifest.json          # SessionManifest, serialized
        frames/
            000000.png         # RGB
            000000_depth.png   # depth, 16-bit PNG, millimetres
            ...
        session_metadata.json  # optional, see SessionMetadataOverride
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class DepthSource(str, Enum):
    LIDAR = "lidar"
    TOF = "tof"
    ML_ESTIMATED = "ml_estimated"


class Intrinsics(BaseModel):
    """Pinhole camera intrinsics, shared by all frames in a session."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class DeviceInfo(BaseModel):
    model: str = "unknown"
    os_version: str | None = None
    app: str | None = None
    depth_source: DepthSource = DepthSource.ML_ESTIMATED


class Frame(BaseModel):
    """One keyframe: RGB image, depth map, and camera pose."""

    frame_id: str
    timestamp: float
    rgb_path: str
    depth_path: str
    pose: list[list[float]] = Field(
        description="4x4 camera-to-world matrix, row-major (list of 4 rows of 4 floats)"
    )
    confidence_path: str | None = None

    def pose_matrix(self):
        import numpy as np

        return np.array(self.pose, dtype=np.float64)


class SessionManifest(BaseModel):
    """The canonical description of one scanning session."""

    session_id: str
    device: DeviceInfo
    intrinsics: Intrinsics
    depth_unit: str = Field(default="mm", description="'mm' or 'm' — unit of raw depth values")
    frames: list[Frame] = Field(default_factory=list)
    source_format: str | None = Field(
        default=None, description="Name of the ingest adapter that produced this manifest"
    )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SessionManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class SessionMetadataOverride(BaseModel):
    """Optional sidecar with authoritative values the mobile app would supply.

    When present, quality scoring prefers these over pipeline-derived
    estimates. All fields are optional — provide only what you know.
    """

    coverage_pct: float | None = Field(default=None, ge=0, le=100)
    depth_source: DepthSource | None = None
    has_loop_closure: bool | None = None
    notes: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "SessionMetadataOverride":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
