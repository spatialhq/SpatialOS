# Running in Docker

## Default image (Open3D only, recommended)

```bash
docker build -t spatial-os .

docker run --rm -v "${PWD}/data:/data" spatial-os \
  ingest /data/raw/<scan_name> --format 3dscannerapp --output /data/sessions/<scan_name>

docker run --rm -v "${PWD}/data:/data" spatial-os \
  process /data/sessions/<scan_name> --output /data/out/<scan_name> --format ply,rosbag,colmap
```

Or with `docker-compose` if you prefer:

```bash
docker compose run --rm spatial-os process /data/sessions/<scan_name> --output /data/out/<scan_name> --format ply
```

## Running tests in the container

```bash
docker build -t spatial-os-test .
docker run --rm --entrypoint pytest spatial-os-test -q
```

## Optional: COLMAP-backed image

Only needed if you want `--backend colmap` (native SfM+MVS) instead of the
default pose-driven Open3D reconstruction. See
[colmap-tradeoffs.md](colmap-tradeoffs.md) for why this isn't the default.

```bash
docker build -f Dockerfile.colmap -t spatial-os-colmap .
docker run --rm -v "${PWD}/data:/data" spatial-os-colmap \
  process /data/sessions/<scan_name> --output /data/out/<scan_name> --backend colmap --format colmap
```

GPU acceleration for COLMAP inside Docker requires `--gpus all` and the
NVIDIA Container Toolkit; on Windows this means Docker Desktop with WSL2 GPU
passthrough enabled.
