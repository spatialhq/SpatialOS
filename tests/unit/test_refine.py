import numpy as np
import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.unpack import prepare_frames
from spatial_os.reconstruction.open3d_backend import reconstruct
from spatial_os.refine.backproject import label_point_cloud
from spatial_os.refine.instance_clustering import cluster_instances
from spatial_os.refine.mesh_completion import complete_mesh
from spatial_os.schema import SessionManifest

WALL_ID, FLOOR_ID, CHAIR_ID = 0, 1, 2
ID2LABEL = {WALL_ID: "wall", FLOOR_ID: "floor", CHAIR_ID: "chair"}


class FakeSegmenter:
    """Deterministic stand-in for SegformerADE20KSegmenter -- avoids the `ai` extra in tests."""

    def segment(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        class_map = np.full((h, w), WALL_ID, dtype=np.int32)
        class_map[h // 2 :, :] = FLOOR_ID
        return class_map, ID2LABEL


def test_label_point_cloud_assigns_known_labels(synthetic_session_dir):
    manifest = SessionManifest.load(synthetic_session_dir / "manifest.json")
    cfg = ProcessConfig(blur_threshold=0.0)
    prepared = prepare_frames(manifest, synthetic_session_dir, cfg)
    pcd = reconstruct(manifest, prepared, cfg)

    labels, id2label = label_point_cloud(manifest, prepared, pcd, FakeSegmenter(), max_frames=5)

    assert id2label == ID2LABEL
    assert len(labels) == len(pcd.points)
    assert set(np.unique(labels).tolist()) <= {WALL_ID, FLOOR_ID, -1}


def test_cluster_instances_finds_separated_groups_and_excludes_structural():
    points = np.vstack(
        [
            np.random.default_rng(0).normal(loc=[0, 0, 0], scale=0.01, size=(50, 3)),
            np.random.default_rng(1).normal(loc=[5, 5, 5], scale=0.01, size=(50, 3)),
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.full(100, CHAIR_ID, dtype=np.int32)
    labels[:10] = WALL_ID  # a few structural points mixed in, should never get an instance id

    instances = cluster_instances(pcd, labels, structural_class_ids={WALL_ID}, eps=0.5, min_points=5)

    assert instances[:10].tolist() == [-1] * 10  # structural excluded
    found_instances = set(instances[10:].tolist()) - {-1}
    assert len(found_instances) == 2  # the two well-separated chair clusters


def test_complete_mesh_produces_nonempty_mesh():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    pcd = sphere.sample_points_poisson_disk(500)
    pcd.estimate_normals()

    mesh = complete_mesh(pcd, depth=6)

    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
