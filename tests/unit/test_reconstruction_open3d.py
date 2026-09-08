from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.unpack import prepare_frames
from spatial_os.reconstruction.open3d_backend import reconstruct
from spatial_os.schema import SessionManifest


def test_open3d_backend_produces_nonempty_cloud(synthetic_session_dir):
    manifest = SessionManifest.load(synthetic_session_dir / "manifest.json")
    cfg = ProcessConfig(blur_threshold=0.0, voxel_size=0.02, depth_max=5.0)
    prepared = prepare_frames(manifest, synthetic_session_dir, cfg)

    pcd = reconstruct(manifest, prepared, cfg)

    assert len(pcd.points) > 0
    bbox = pcd.get_axis_aligned_bounding_box()
    # Synthetic scene is a plane ~1.5m in front of the camera; z should stay in a sane range.
    assert bbox.min_bound[2] > 0.0
    assert bbox.max_bound[2] < 5.0
