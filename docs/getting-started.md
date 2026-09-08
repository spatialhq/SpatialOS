# Getting Started

## 1. Capture a scan

`spatial-os` processes data — it doesn't capture it. Use the iPhone LiDAR
scanning app described in the [protocols.io tutorial](https://www.protocols.io/view/iphone-lidar-tutorial-yxmvm21w9g3p/v1)
to record a scan and export its raw data. See [data-capture.md](data-capture.md).

If you don't have your own scan yet, fetch a public sample:

```bash
python scripts/download_sample_data.py --dataset 3dscannerapp_samples --output data/samples
```

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
spatial-os --version
```

Or skip local install entirely and use Docker — see [docker.md](docker.md).

## 3. Ingest the raw export

Raw exports don't share a single format, so the first step converts them into
spatial-os's canonical session layout (`manifest.json` + `frames/`):

```bash
spatial-os ingest data/samples/<scan_name> --format 3dscannerapp --output data/sessions/<scan_name>
```

`--format generic` is available if your data is already in the canonical
schema (see `src/spatial_os/schema.py`).

Sanity-check what was ingested:

```bash
spatial-os inspect data/sessions/<scan_name>
```

## 4. Process

```bash
spatial-os process data/sessions/<scan_name> \
  --output data/out/<scan_name> \
  --format ply,pcd,rosbag,colmap
```

This runs preprocessing (blur filtering, depth normalization), reconstruction
(Open3D pose-driven TSDF fusion by default), postprocessing (outlier removal,
voxel downsampling, normal estimation), and writes every requested export
format plus `metadata.json` (device info + quality score).

## 5. Interpret the output

`data/out/<scan_name>/`:

- `cloud.ply`, `cloud.pcd` — open in Open3D, CloudCompare, MeshLab, etc.
- `session_rosbag/` — a rosbag2 bag with `/spatial_os/points`
  (`sensor_msgs/PointCloud2`) and `/spatial_os/path` (`nav_msgs/Path`).
- `colmap/sparse/{cameras,images,points3D}.txt` + `colmap/dense/fused.ply` —
  feed directly into `nerfstudio`/`instant-ngp` for NeRF training.
- `metadata.json` — device info and the quality report (see
  [quality-scoring.md](quality-scoring.md)); score `< 40` means the session
  should probably be re-scanned.

To re-score without re-running the whole pipeline (e.g. after supplying a
`session_metadata.json` override):

```bash
spatial-os score data/sessions/<scan_name> --ply data/out/<scan_name>/cloud.ply
```

## 6. (Optional) Fuse multiple sessions of the same space

```bash
spatial-os fuse data/out/session_a/cloud.ply data/out/session_b/cloud.ply --output data/out/merged
```
