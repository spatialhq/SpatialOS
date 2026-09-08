import json
from pathlib import Path

from spatial_os.quality.scoring import QualityReport
from spatial_os.schema import SessionManifest


def write_metadata(
    manifest: SessionManifest, quality: QualityReport, out_dir: str | Path, name: str = "metadata"
) -> Path:
    path = Path(out_dir) / f"{name}.json"
    payload = {
        "session_id": manifest.session_id,
        "device": manifest.device.model_dump(),
        "source_format": manifest.source_format,
        "frame_count": len(manifest.frames),
        "quality": quality.model_dump(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
