from typer.testing import CliRunner

from spatial_os.cli import app

runner = CliRunner()


def test_process_cli_end_to_end(synthetic_session_dir, tmp_path):
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "process",
            str(synthetic_session_dir),
            "--output",
            str(output_dir),
            "--format",
            "ply,pcd,rosbag,colmap",
            "--blur-threshold",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "cloud.ply").exists()
    assert (output_dir / "cloud.pcd").exists()
    assert (output_dir / "session_rosbag").exists()
    assert (output_dir / "colmap" / "sparse" / "cameras.txt").exists()
    assert (output_dir / "colmap" / "dense" / "fused.ply").exists()
    assert (output_dir / "metadata.json").exists()
