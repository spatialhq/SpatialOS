# Changelog

## 0.1.0

- Initial pipeline: ingest (3D Scanner App / generic) -> preprocess -> Open3D
  RGBD/TSDF reconstruction -> postprocess -> export (PLY/PCD/ROS bag/COLMAP) ->
  quality scoring.
- Optional COLMAP backend (Docker-only, see docs/colmap-tradeoffs.md).
- Optional multi-session fusion (`spatial-os fuse`).
