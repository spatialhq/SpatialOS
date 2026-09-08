from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from spatial_os.__about__ import __version__
from spatial_os.config import ProcessConfig

app = typer.Typer(add_completion=False, help="spatial-os: process raw LiDAR point cloud sessions.")
console = Console()


def _version_callback(value: bool):
    if value:
        console.print(f"spatial-os {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
):
    pass


@app.command()
def ingest(
    raw_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Raw export directory from a scanning app."),
    output: Path = typer.Option(..., "--output", "-o", help="Canonical session directory to create."),
    format: str = typer.Option("3dscannerapp", "--format", "-f", help="Raw format adapter: 3dscannerapp | generic."),
):
    """Convert a raw scanning-app export into spatial-os's canonical session schema."""
    if format == "3dscannerapp":
        from spatial_os.ingest.threed_scanner_app import ingest as adapter_ingest
    elif format == "generic":
        from spatial_os.ingest.generic_folder import ingest as adapter_ingest
    else:
        console.print(f"[red]Unknown format:[/red] {format} (expected 3dscannerapp|generic)")
        raise typer.Exit(code=1)

    manifest = adapter_ingest(raw_dir, output)
    console.print(f"[green]Ingested[/green] {len(manifest.frames)} frames -> {output}")


@app.command()
def process(
    session_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Canonical session directory."),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to write pipeline outputs into."),
    format: str = typer.Option("ply", "--format", "-f", help="Comma-separated: ply,pcd,rosbag,colmap"),
    backend: str = typer.Option("open3d", "--backend", help="open3d (default, no external deps) or colmap."),
    voxel_size: float = typer.Option(0.02, "--voxel-size", help="Voxel size in meters."),
    blur_threshold: float = typer.Option(100.0, "--blur-threshold", help="Laplacian variance blur cutoff."),
):
    """Run preprocessing -> reconstruction -> postprocessing -> export -> scoring on one session."""
    from spatial_os.pipeline import run_process

    formats = [f.strip() for f in format.split(",") if f.strip()]
    cfg = ProcessConfig(backend=backend, voxel_size=voxel_size, blur_threshold=blur_threshold)

    result = run_process(session_dir, output, formats, cfg)

    console.print(f"[green]Quality score:[/green] {result.quality.total_score:.1f} ({result.quality.verdict})")
    for path in result.written_files:
        console.print(f"  wrote {path}")


@app.command()
def score(
    session_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    ply: Path = typer.Option(..., "--ply", help="Reconstructed point cloud to score (e.g. from `process`)."),
    metadata: Path | None = typer.Option(None, "--metadata", help="Optional session_metadata.json override."),
):
    """Score an already-reconstructed point cloud without re-running the full pipeline."""
    import open3d as o3d

    from spatial_os.postprocessing.outlier_removal import remove_outliers
    from spatial_os.quality.scoring import score_session
    from spatial_os.schema import SessionManifest, SessionMetadataOverride
    from spatial_os.utils.io_paths import manifest_path

    manifest = SessionManifest.load(manifest_path(session_dir))
    pcd = o3d.io.read_point_cloud(str(ply))
    _, outlier_ratio = remove_outliers(pcd, ProcessConfig())
    override = SessionMetadataOverride.load(metadata) if metadata else None

    report = score_session(manifest, manifest.frames, pcd, outlier_ratio, override=override)
    console.print_json(json.dumps(report.model_dump(), indent=2))


@app.command()
def fuse(
    session_plys: list[Path] = typer.Argument(..., help="Two or more reconstructed .ply files to merge."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for the fused scene."),
):
    """Merge multiple sessions' point clouds into one scene (global registration + ICP)."""
    from spatial_os.export.ply_pcd import write_ply
    from spatial_os.fusion.multi_session import load_and_fuse
    from spatial_os.utils.io_paths import ensure_dir

    if len(session_plys) < 2:
        console.print("[red]Need at least two point clouds to fuse.[/red]")
        raise typer.Exit(code=1)

    merged = load_and_fuse(session_plys)
    out_dir = ensure_dir(output)
    path = write_ply(merged, out_dir, name="fused_scene")
    console.print(f"[green]Fused[/green] {len(session_plys)} sessions -> {path} ({len(merged.points)} points)")


@app.command()
def refine(
    session_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Canonical session directory (for RGB/depth/pose keyframes)."),
    cloud: Path = typer.Option(..., "--cloud", help="Reconstructed point cloud to refine (e.g. output of `process`)."),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to write refinement outputs into."),
    max_frames: int = typer.Option(20, "--max-frames", help="Max keyframes to run semantic segmentation on."),
    mesh_completion: bool = typer.Option(False, "--mesh-completion", help="Also run Poisson mesh completion (classical, not AI)."),
):
    """AI-assisted object/layout refinement: semantic segmentation of keyframes,
    back-projected to 3D, plus per-object instance clustering. Requires the
    'ai' extra: pip install "spatial-os[ai]"."""
    from spatial_os.refine.pipeline import run_refine

    try:
        result = run_refine(session_dir, cloud, output, max_frames=max_frames, do_mesh_completion=mesh_completion)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Refined[/green] -> {result.output_dir}")
    for path in result.written_files:
        console.print(f"  wrote {path}")


@app.command()
def ui(
    port: int = typer.Option(7860, "--port", help="Port to serve the Gradio UI on."),
):
    """Launch the Gradio web UI: upload -> process -> refine -> visualize.
    Requires the 'app' extra: pip install "spatial-os[app]"."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        console.print('[red]The web UI requires the \'app\' extra: pip install "spatial-os[app]"[/red]')
        raise typer.Exit(code=1)

    from spatial_os.webapp.app import build_app

    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=port)


@app.command()
def inspect(
    session_dir: Path = typer.Argument(..., exists=True, file_okay=False),
):
    """Print quick diagnostics for a canonical session directory."""
    from spatial_os.schema import SessionManifest
    from spatial_os.utils.io_paths import manifest_path

    manifest = SessionManifest.load(manifest_path(session_dir))
    console.print(f"session_id: {manifest.session_id}")
    console.print(f"source_format: {manifest.source_format}")
    console.print(f"device: {manifest.device.model} (depth_source={manifest.device.depth_source})")
    console.print(f"frames: {len(manifest.frames)}")
    if manifest.frames:
        console.print(f"intrinsics: {manifest.intrinsics}")


if __name__ == "__main__":
    app()
