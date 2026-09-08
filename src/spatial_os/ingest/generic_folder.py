"""Fallback adapter: assumes `raw_dir` already matches the canonical schema
(a manifest.json + frames/ directory), and simply copies it into place.
Use this when you've hand-authored a session in spatial-os's own format.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from spatial_os.schema import SessionManifest
from spatial_os.utils.io_paths import manifest_path


def ingest(raw_dir: Path, session_dir: Path) -> SessionManifest:
    src_manifest = manifest_path(raw_dir)
    if not src_manifest.exists():
        raise FileNotFoundError(
            f"{raw_dir} has no manifest.json -- 'generic' ingest expects data already "
            "in spatial-os's canonical schema. Use a format-specific adapter instead, "
            "or author a manifest.json (see spatial_os.schema.SessionManifest)."
        )

    if session_dir.resolve() != raw_dir.resolve():
        session_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(raw_dir, session_dir, dirs_exist_ok=True)

    manifest = SessionManifest.load(manifest_path(session_dir))
    manifest.source_format = manifest.source_format or "generic"
    return manifest
