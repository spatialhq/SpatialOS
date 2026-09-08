"""Gradio UI: upload a raw scan -> process -> (optional) AI layout/object
refinement -> visualize the resulting point cloud in-browser.

Requires the 'app' extra: pip install "spatial-os[app]"
Run with: spatial-os ui   (or: python -m spatial_os.webapp.app)
"""

from __future__ import annotations

from spatial_os.config import ProcessConfig
from spatial_os.webapp.pipeline_runner import run_full_pipeline


def _run(raw_zip, ingest_format, do_refine, do_mesh_completion, max_frames, voxel_size, blur_threshold):
    if raw_zip is None:
        return None, {"error": "Upload a .zip of your raw scan export first."}, None, "No file uploaded."

    try:
        cfg = ProcessConfig(voxel_size=voxel_size, blur_threshold=blur_threshold)
        output = run_full_pipeline(
            raw_zip.name if hasattr(raw_zip, "name") else raw_zip,
            ingest_format=ingest_format,
            do_refine=do_refine,
            do_mesh_completion=do_mesh_completion,
            max_frames=int(max_frames),
            cfg=cfg,
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the UI, don't crash the server
        return None, {"error": str(exc)}, None, f"Failed: {exc}"

    status = f"Done. Quality: {output.quality_report['total_score']:.1f} ({output.quality_report['verdict']})"
    return str(output.display_cloud_path), output.quality_report, output.labels_report, status


def build_app():
    import gradio as gr

    with gr.Blocks(title="spatial-os") as demo:
        gr.Markdown(
            "# spatial-os\n"
            "Upload a raw LiDAR scan export (zipped) → process → optional AI "
            "layout/object refinement → visualize the result."
        )

        with gr.Row():
            with gr.Column(scale=1):
                raw_zip = gr.File(label="Raw scan export (.zip)", file_types=[".zip"])
                ingest_format = gr.Radio(
                    ["3dscannerapp", "generic"], value="3dscannerapp", label="Raw format"
                )
                with gr.Accordion("Processing options", open=False):
                    voxel_size = gr.Slider(0.005, 0.1, value=0.02, step=0.005, label="Voxel size (m)")
                    blur_threshold = gr.Slider(0.0, 300.0, value=100.0, step=10.0, label="Blur filter threshold")
                with gr.Accordion("AI layout/object refinement", open=True):
                    do_refine = gr.Checkbox(value=False, label="Run AI refinement (semantic segmentation + instances)")
                    do_mesh_completion = gr.Checkbox(value=False, label="Also run mesh completion (hole filling)")
                    max_frames = gr.Slider(1, 60, value=20, step=1, label="Max keyframes for segmentation")
                run_btn = gr.Button("Run pipeline", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                viewer = gr.Model3D(label="Reconstructed point cloud", clear_color=(0.05, 0.05, 0.05, 1.0))
                quality_json = gr.JSON(label="Quality report")
                labels_json = gr.JSON(label="Refinement report (labels/instances)")

        run_btn.click(
            _run,
            inputs=[raw_zip, ingest_format, do_refine, do_mesh_completion, max_frames, voxel_size, blur_threshold],
            outputs=[viewer, quality_json, labels_json, status],
        )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
