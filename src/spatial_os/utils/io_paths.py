from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
METADATA_OVERRIDE_FILENAME = "session_metadata.json"
FRAMES_DIRNAME = "frames"


def manifest_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / MANIFEST_FILENAME


def metadata_override_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / METADATA_OVERRIDE_FILENAME


def frames_dir(session_dir: str | Path) -> Path:
    return Path(session_dir) / FRAMES_DIRNAME


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
