import numpy as np
import open3d as o3d

from spatial_os.quality.scoring import score_session
from spatial_os.schema import DepthSource, DeviceInfo, Frame, Intrinsics, SessionManifest, SessionMetadataOverride


def _manifest_and_frames(depth_source=DepthSource.LIDAR):
    intrinsics = Intrinsics(width=64, height=48, fx=50, fy=50, cx=32, cy=24)
    frames = [
        Frame(
            frame_id=str(i),
            timestamp=float(i),
            rgb_path=f"frames/{i}.png",
            depth_path=f"frames/{i}_depth.png",
            pose=[[1, 0, 0, i * 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        for i in range(6)
    ]
    manifest = SessionManifest(
        session_id="s", device=DeviceInfo(model="test", depth_source=depth_source), intrinsics=intrinsics, frames=frames
    )
    return manifest, frames


def _dense_cloud(n=5000) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).random((n, 3)))
    return pcd


def test_score_is_weighted_average_in_0_100_range():
    manifest, frames = _manifest_and_frames()
    pcd = _dense_cloud()

    report = score_session(manifest, frames, pcd, outlier_ratio=0.0)

    assert 0.0 <= report.total_score <= 100.0
    assert sum(c.weight for c in report.components.values()) == 1.0


def test_lidar_scores_higher_depth_quality_than_ml_estimated():
    manifest_lidar, frames = _manifest_and_frames(DepthSource.LIDAR)
    manifest_ml, _ = _manifest_and_frames(DepthSource.ML_ESTIMATED)
    pcd = _dense_cloud()

    report_lidar = score_session(manifest_lidar, frames, pcd, outlier_ratio=0.0)
    report_ml = score_session(manifest_ml, frames, pcd, outlier_ratio=0.0)

    assert report_lidar.components["depth_quality"].score > report_ml.components["depth_quality"].score


def test_metadata_override_marks_components_not_estimated():
    manifest, frames = _manifest_and_frames()
    pcd = _dense_cloud()
    override = SessionMetadataOverride(coverage_pct=80.0, has_loop_closure=True)

    report = score_session(manifest, frames, pcd, outlier_ratio=0.0, override=override)

    assert report.components["coverage"].estimated is False
    assert report.components["coverage"].score == 80.0
    assert report.components["trajectory"].estimated is False


def test_empty_cloud_yields_low_score_and_reject_verdict():
    # Zero-motion trajectory (no coverage/density either) -> worst case, should reject.
    manifest, frames = _manifest_and_frames()
    for frame in frames:
        frame.pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    empty = o3d.geometry.PointCloud()

    report = score_session(manifest, frames, empty, outlier_ratio=0.0)

    assert report.verdict == "reject"
    assert report.total_score < 40.0
