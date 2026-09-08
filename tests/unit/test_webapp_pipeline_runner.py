import zipfile
from pathlib import Path

from spatial_os.webapp.pipeline_runner import run_full_pipeline


def _zip_session(session_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in session_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(session_dir))
    return zip_path


def test_run_full_pipeline_without_refine(synthetic_session_dir, tmp_path):
    zip_path = _zip_session(synthetic_session_dir, tmp_path / "raw.zip")

    output = run_full_pipeline(zip_path, ingest_format="generic", do_refine=False)

    assert output.display_cloud_path.exists()
    assert output.display_cloud_path.name == "cloud.ply"
    assert 0.0 <= output.quality_report["total_score"] <= 100.0
    assert output.labels_report is None


def test_run_full_pipeline_with_refine_reports_missing_ai_extra_gracefully(synthetic_session_dir, tmp_path):
    zip_path = _zip_session(synthetic_session_dir, tmp_path / "raw.zip")

    output = run_full_pipeline(zip_path, ingest_format="generic", do_refine=True)

    # Without the 'ai' extra installed, refinement should fail gracefully and
    # still return a usable (pre-refine) point cloud rather than crashing.
    assert output.display_cloud_path.exists()
    if output.labels_report is not None:
        assert "error" in output.labels_report
