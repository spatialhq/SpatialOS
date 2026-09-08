from pathlib import Path
from typing import Protocol

from spatial_os.schema import SessionManifest


class IngestAdapter(Protocol):
    """Converts one raw scanning-app export into a canonical session directory."""

    def ingest(self, raw_dir: Path, session_dir: Path) -> SessionManifest: ...
