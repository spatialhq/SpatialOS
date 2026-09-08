"""Non-UI orchestration for the web app: raw upload -> ingest -> process ->
refine -> a display-ready point cloud path + JSON reports. Kept separate from
`app.py` so it's testable without gradio installed.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from spatial_os.config import ProcessConfig
from spatial_os.pipeline import run_process
from spatial_os.utils.io_paths import manifest_path
from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineRunOutput:
    display_cloud_path: Path
    quality_report: dict
    labels_report: dict | None
    workdir: Path


def _find_raw_root(extracted_dir: Path) -> Path:
    """If the zip contained one wrapping folder, descend into it."""
    entries = [p for p in extracted_dir.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted_dir


def run_full_pipeline(
    raw_zip_path: str | Path,
    ingest_format: str = "3dscannerapp",
    do_refine: bool = False,
    do_mesh_completion: bool = False,
    max_frames: int = 20,
    cfg: ProcessConfig | None = None,
) -> PipelineRunOutput:
    """Upload -> ingest -> process -> (optional) refine, in one call for the web UI."""
    cfg = cfg or ProcessConfig()
    workdir = Path(tempfile.mkdtemp(prefix="spatial-os-ui-"))

    extracted_dir = workdir / "raw_extracted"
    with zipfile.ZipFile(raw_zip_path) as zf:
        zf.extractall(extracted_dir)
    raw_dir = _find_raw_root(extracted_dir)

    session_dir = workdir / "session"
    if ingest_format == "3dscannerapp":
        from spatial_os.ingest.threed_scanner_app import ingest as adapter_ingest
    elif ingest_format == "generic":
        from spatial_os.ingest.generic_folder import ingest as adapter_ingest
    else:
        raise ValueError(f"unknown ingest format: {ingest_format!r}")

    logger.info("ingesting %s (format=%s)", raw_dir, ingest_format)
    adapter_ingest(raw_dir, session_dir)
    assert manifest_path(session_dir).exists()

    out_dir = workdir / "out"
    logger.info("processing session -> %s", out_dir)
    result = run_process(session_dir, out_dir, formats=["ply"], cfg=cfg)

    display_path = out_dir / "cloud.ply"
    labels_report: dict | None = None

    if do_refine:
        from spatial_os.refine.pipeline import run_refine

        refine_dir = workdir / "refined"
        try:
            logger.info("running AI refinement -> %s", refine_dir)
            run_refine(
                session_dir,
                display_path,
                refine_dir,
                max_frames=max_frames,
                do_mesh_completion=do_mesh_completion,
                cfg=cfg,
            )
            display_path = refine_dir / "labeled_cloud.ply"
            labels_report = json.loads((refine_dir / "labels.json").read_text(encoding="utf-8"))
        except ImportError as exc:
            labels_report = {"error": str(exc)}

    return PipelineRunOutput(
        display_cloud_path=display_path,
        quality_report=result.quality.model_dump(),
        labels_report=labels_report,
        workdir=workdir,
    )
