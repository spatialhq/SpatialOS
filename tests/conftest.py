from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from spatial_os.schema import DepthSource, DeviceInfo, Frame, Intrinsics, SessionManifest

WIDTH, HEIGHT = 64, 48
FX = FY = 50.0
CX, CY = WIDTH / 2, HEIGHT / 2
PLANE_DEPTH_MM = 1500
N_FRAMES = 6


def _make_checkerboard(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    block = 8
    for y in range(0, height, block):
        for x in range(0, width, block):
            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            img[y : y + block, x : x + block] = color
    return img


def _make_depth(width: int, height: int) -> np.ndarray:
    # A flat plane at PLANE_DEPTH_MM with tiny per-pixel jitter so the cloud
    # isn't perfectly degenerate; well within depth_max after /1000.
    depth = np.full((height, width), PLANE_DEPTH_MM, dtype=np.uint16)
    jitter = (np.indices((height, width)).sum(axis=0) % 5).astype(np.uint16)
    return depth + jitter


def build_synthetic_session(session_dir: Path, n_frames: int = N_FRAMES) -> SessionManifest:
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(n_frames):
        frame_id = f"{i:06d}"
        rgb = _make_checkerboard(WIDTH, HEIGHT, seed=i)
        depth = _make_depth(WIDTH, HEIGHT)

        rgb_path = frames_dir / f"{frame_id}.png"
        depth_path = frames_dir / f"{frame_id}_depth.png"
        cv2.imwrite(str(rgb_path), rgb)
        cv2.imwrite(str(depth_path), depth)

        # Camera-to-world: identity rotation, translate slightly along x per frame.
        pose = [
            [1.0, 0.0, 0.0, i * 0.05],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        frames.append(
            Frame(
                frame_id=frame_id,
                timestamp=float(i),
                rgb_path=f"frames/{frame_id}.png",
                depth_path=f"frames/{frame_id}_depth.png",
                pose=pose,
            )
        )

    manifest = SessionManifest(
        session_id="synthetic-session",
        device=DeviceInfo(model="synthetic", app="pytest", depth_source=DepthSource.LIDAR),
        intrinsics=Intrinsics(width=WIDTH, height=HEIGHT, fx=FX, fy=FY, cx=CX, cy=CY),
        depth_unit="mm",
        frames=frames,
        source_format="synthetic",
    )
    manifest.save(session_dir / "manifest.json")
    return manifest


@pytest.fixture
def synthetic_session_dir(tmp_path: Path) -> Path:
    session_dir = tmp_path / "session"
    build_synthetic_session(session_dir)
    return session_dir
