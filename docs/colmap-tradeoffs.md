# Why COLMAP is optional and Docker-only

COLMAP is a C++ binary. It isn't pip-installable, and building it from source
on native Windows requires VCPKG, CUDA toolchain setup, and CMake — a heavy
lift for a backend processing tool.

We don't actually need COLMAP's core value-add (Structure-from-Motion pose
recovery) here: LiDAR-equipped iPhones already provide accurate camera pose
per frame via ARKit. What COLMAP would normally contribute on top of that —
dense multi-view stereo reconstruction — is covered by Open3D's pose-driven
RGBD/TSDF fusion (`reconstruction/open3d_backend.py`), which has no external
binary dependency and runs the same way on Windows, macOS, and Linux.

So:

- **Default backend (`--backend open3d`)**: zero external dependencies,
  works out of the box, including on native Windows.
- **COLMAP-format export** (`--format colmap`): ships regardless of backend —
  it's just writing `cameras.txt`/`images.txt`/`points3D.txt` from data we
  already have (ARKit poses + the Open3D-reconstructed cloud), so you get
  NeRF-ready output without installing COLMAP at all.
- **COLMAP backend (`--backend colmap`)**: only for when you specifically
  want COLMAP's own SfM+MVS pipeline (e.g. to compare against, or for
  scenes without reliable pose data). Requires the `colmap` binary on PATH;
  the documented, supported way to get that is `Dockerfile.colmap`, not a
  native Windows install.

If your workflow requires COLMAP on bare Windows, the recommended path is a
WSL2 dev environment or the Docker image above, rather than fighting a native
Windows build.
