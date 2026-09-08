# Web UI (Gradio)

`spatial-os ui` launches a browser UI for the full flow: upload a raw scan →
process → optional AI layout/object refinement → visualize the resulting
point cloud in-browser.

## Local

```bash
pip install -e ".[app,ai]"   # 'ai' optional, only needed for refinement
spatial-os ui --port 7860
```

Open http://localhost:7860.

## Docker (recommended — this image is large, ~ a few GB, due to torch/transformers)

```bash
docker build -f Dockerfile.app -t spatial-os-app .
docker run --rm -p 7860:7860 spatial-os-app
```

or `docker compose up spatial-os-ui`.

## Usage

1. Zip your raw scan export (the same folder you'd pass to
   `spatial-os ingest`) and upload it.
2. Pick the raw format (`3dscannerapp` for 3D Scanner App exports,
   `generic` if the folder is already in spatial-os's canonical
   `manifest.json` + `frames/` schema).
3. Optionally enable AI refinement (semantic segmentation + instance
   clustering) and/or mesh completion — see [refine.md](refine.md) for what
   these do and their tradeoffs.
4. Click **Run pipeline**. The reconstructed (or refined) point cloud
   renders in the 3D viewer; the quality score and refinement report show
   alongside it.

The UI is a thin wrapper (`src/spatial_os/webapp/`) around the same
`spatial_os.pipeline.run_process` and `spatial_os.refine.pipeline.run_refine`
functions the CLI uses — nothing web-specific lives in the core pipeline.
