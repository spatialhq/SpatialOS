from __future__ import annotations

from typing import Protocol

import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.unpack import PreparedFrame
from spatial_os.schema import SessionManifest


class ReconstructionBackend(Protocol):
    def reconstruct(
        self,
        manifest: SessionManifest,
        frames: list[PreparedFrame],
        cfg: ProcessConfig,
    ) -> o3d.geometry.PointCloud: ...
