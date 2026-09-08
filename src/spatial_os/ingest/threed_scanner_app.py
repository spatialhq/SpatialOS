"""Adapter for the "3D Scanner App" (Laan Labs) 'All Data' export -- the app
referenced by intent.md's protocols.io tutorial and by the sample exports at
https://github.com/laanlabs/3dScannerApp_samples.

Expected raw_dir layout (per-frame triplet + per-frame pose/intrinsics JSON):

    frame_00000.jpg
    depth_00000.png        # 16-bit, millimetres
    conf_00000.png         # optional confidence map, ignored here
    frame_00000.json       # {"cameraPoseARFrame": [16 floats, row-major camera-to-world],
                            #  "intrinsics": [9 floats, row-major 3x3], "time": float}
    info.json               # optional: {"deviceModel": ..., "iosVersion": ...}

Verified against a real export (laanlabs/3dScannerApp_samples' `chair_scan`):
RGB is full camera resolution (e.g. 1920x1440) while depth is the LiDAR
sensor's native, much lower resolution (e.g. 256x192) -- this adapter
downscales RGB to depth resolution and scales intrinsics to match, since
downstream RGBD integration needs both at the same size. Only every few
frames has an RGB `.jpg` (frames without one are skipped); depth/pose exist
for every frame.

If your actual downloaded sample differs further (field names/casing vary
across app versions), adjust the parsing below accordingly.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2

from spatial_os.schema import DepthSource, DeviceInfo, Frame, Intrinsics, SessionManifest
from spatial_os.utils.io_paths import ensure_dir, frames_dir

FRAME_JSON_RE = re.compile(r"frame_(\d+)\.json$")


def ingest(raw_dir: Path, session_dir: Path) -> SessionManifest:
    raw_dir = Path(raw_dir)
    out_frames_dir = ensure_dir(frames_dir(session_dir))

    frame_jsons = sorted(raw_dir.glob("frame_*.json"), key=lambda p: p.name)
    if not frame_jsons:
        raise FileNotFoundError(f"no frame_*.json files found under {raw_dir}")

    info = {}
    info_path = raw_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))

    frames: list[Frame] = []
    intrinsics: Intrinsics | None = None

    for frame_json_path in frame_jsons:
        match = FRAME_JSON_RE.search(frame_json_path.name)
        if not match:
            continue
        idx = match.group(1)

        rgb_src = raw_dir / f"frame_{idx}.jpg"
        depth_src = raw_dir / f"depth_{idx}.png"
        if not rgb_src.exists() or not depth_src.exists():
            continue

        meta = json.loads(frame_json_path.read_text(encoding="utf-8"))
        pose_flat = meta["cameraPoseARFrame"]
        pose = [pose_flat[i * 4 : i * 4 + 4] for i in range(4)]

        # Real ARKit LiDAR exports have RGB at full camera resolution (e.g.
        # 1920x1440) but depth at a much lower sensor resolution (e.g.
        # 256x192); the two must match for RGBD integration downstream, so
        # RGB is downscaled to depth resolution and intrinsics (given at RGB
        # resolution) are scaled to match.
        rgb_img = cv2.imread(str(rgb_src))
        depth_img = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        rh, rw = rgb_img.shape[:2]
        dh, dw = depth_img.shape[:2]
        if (rw, rh) != (dw, dh):
            rgb_img = cv2.resize(rgb_img, (dw, dh), interpolation=cv2.INTER_AREA)

        if intrinsics is None:
            k = meta["intrinsics"]
            sx, sy = dw / rw, dh / rh
            intrinsics = Intrinsics(width=dw, height=dh, fx=k[0] * sx, fy=k[4] * sy, cx=k[2] * sx, cy=k[5] * sy)

        rgb_dst_name = f"{idx}.png"
        depth_dst_name = f"{idx}_depth.png"
        cv2.imwrite(str(out_frames_dir / rgb_dst_name), rgb_img)
        shutil.copyfile(depth_src, out_frames_dir / depth_dst_name)

        frames.append(
            Frame(
                frame_id=idx,
                timestamp=float(meta.get("time", 0.0)),
                rgb_path=f"frames/{rgb_dst_name}",
                depth_path=f"frames/{depth_dst_name}",
                pose=pose,
            )
        )

    if intrinsics is None or not frames:
        raise ValueError(f"no valid frames ingested from {raw_dir}")

    manifest = SessionManifest(
        session_id=session_dir.name,
        device=DeviceInfo(
            model=info.get("deviceModel", "unknown"),
            os_version=info.get("iosVersion"),
            app="3D Scanner App",
            depth_source=DepthSource.LIDAR,
        ),
        intrinsics=intrinsics,
        depth_unit="mm",
        frames=frames,
        source_format="3dscannerapp",
    )
    manifest.save(session_dir / "manifest.json")
    return manifest
