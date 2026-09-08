import json

import cv2
import numpy as np

from spatial_os.ingest.generic_folder import ingest as generic_ingest
from spatial_os.ingest.threed_scanner_app import ingest as threed_scanner_app_ingest


def test_generic_ingest_copies_canonical_session(synthetic_session_dir, tmp_path):
    dest = tmp_path / "copied_session"

    manifest = generic_ingest(synthetic_session_dir, dest)

    assert manifest.source_format == "synthetic"  # already set by conftest's manifest
    assert (dest / "manifest.json").exists()
    assert len(list((dest / "frames").glob("*.png"))) > 0


def test_generic_ingest_raises_without_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    try:
        generic_ingest(raw_dir, tmp_path / "out")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def _write_3dscannerapp_raw(raw_dir, rgb_size=(192, 144), depth_size=(64, 48)):
    """Mimics a real 3D Scanner App export where RGB is full camera resolution
    and depth is the (much lower) native LiDAR sensor resolution -- verified
    against a real download (laanlabs/3dScannerApp_samples' chair_scan:
    1920x1440 RGB vs 256x192 depth, both exactly 7.5x apart)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rw, rh = rgb_size
    dw, dh = depth_size

    rgb = np.zeros((rh, rw, 3), dtype=np.uint8)
    cv2.imwrite(str(raw_dir / "frame_00000.jpg"), rgb)
    depth = np.full((dh, dw), 1500, dtype=np.uint16)
    cv2.imwrite(str(raw_dir / "depth_00000.png"), depth)

    fx, fy, cx, cy = 100.0, 100.0, rw / 2, rh / 2
    meta = {
        "cameraPoseARFrame": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
        "time": 0.0,
    }
    (raw_dir / "frame_00000.json").write_text(json.dumps(meta), encoding="utf-8")
    return fx, fy, cx, cy, rw, rh, dw, dh


def test_threed_scanner_app_ingest_scales_intrinsics_for_depth_resolution(tmp_path):
    raw_dir = tmp_path / "raw"
    fx, fy, cx, cy, rw, rh, dw, dh = _write_3dscannerapp_raw(raw_dir)

    manifest = threed_scanner_app_ingest(raw_dir, tmp_path / "session")

    assert manifest.intrinsics.width == dw
    assert manifest.intrinsics.height == dh
    sx, sy = dw / rw, dh / rh
    assert manifest.intrinsics.fx == fx * sx
    assert manifest.intrinsics.cx == cx * sx
    assert manifest.intrinsics.fy == fy * sy
    assert manifest.intrinsics.cy == cy * sy

    rgb_out = cv2.imread(str(tmp_path / "session" / "frames" / "00000.png"))
    assert rgb_out.shape[:2] == (dh, dw)  # resized to match depth, not left at RGB resolution
