"""Orchestrates the optional AI refinement stage: semantic segmentation of
keyframes -> 3D label back-projection -> instance clustering -> optional
mesh completion. Runs after `spatial-os process` (operates on its output
point cloud + the source session's frames), not inside the core pipeline,
since it depends on heavier optional dependencies (see pyproject `[ai]` extra).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.unpack import prepare_frames
from spatial_os.refine.backproject import label_point_cloud
from spatial_os.refine.instance_clustering import cluster_instances
from spatial_os.refine.mesh_completion import complete_mesh
from spatial_os.refine.segmentation_2d import Segmenter, structural_label_ids
from spatial_os.schema import SessionManifest
from spatial_os.utils.io_paths import ensure_dir, manifest_path
from spatial_os.utils.logging import get_logger

logger = get_logger(__name__)

# Distinct, stable-ish colors per semantic class id for quick visual inspection.
_PALETTE_SEED = 42


class RefineResult:
    def __init__(self, output_dir: Path, written_files: list[Path], id2label: dict[int, str]):
        self.output_dir = output_dir
        self.written_files = written_files
        self.id2label = id2label


def run_refine(
    session_dir: str | Path,
    cloud_path: str | Path,
    output_dir: str | Path,
    segmenter: Segmenter | None = None,
    max_frames: int = 20,
    do_mesh_completion: bool = False,
    cfg: ProcessConfig | None = None,
) -> RefineResult:
    cfg = cfg or ProcessConfig()
    session_dir = Path(session_dir)
    output_dir = ensure_dir(output_dir)

    if segmenter is None:
        from spatial_os.refine.segmentation_2d import SegformerADE20KSegmenter

        segmenter = SegformerADE20KSegmenter()

    manifest = SessionManifest.load(manifest_path(session_dir))
    prepared = prepare_frames(manifest, session_dir, cfg)
    pcd = o3d.io.read_point_cloud(str(cloud_path))

    logger.info("running semantic segmentation on up to %d keyframes", max_frames)
    labels, id2label = label_point_cloud(manifest, prepared, pcd, segmenter, depth_max=cfg.depth_max, max_frames=max_frames)

    structural = {v for v in structural_label_ids(id2label).values() if v is not None}
    logger.info("structural class ids: %s", structural)

    instances = cluster_instances(pcd, labels, structural_class_ids=structural)
    n_instances = len(set(instances.tolist()) - {-1})
    logger.info("found %d object instances", n_instances)

    written: list[Path] = []

    np.save(output_dir / "semantic_labels.npy", labels)
    np.save(output_dir / "instance_ids.npy", instances)
    written += [output_dir / "semantic_labels.npy", output_dir / "instance_ids.npy"]

    labels_json_path = output_dir / "labels.json"
    labels_json_path.write_text(
        json.dumps({"id2label": id2label, "structural_class_ids": sorted(structural), "num_instances": n_instances}, indent=2),
        encoding="utf-8",
    )
    written.append(labels_json_path)

    colored = _color_by_label(pcd, labels)
    colored_path = output_dir / "labeled_cloud.ply"
    o3d.io.write_point_cloud(str(colored_path), colored)
    written.append(colored_path)

    if do_mesh_completion:
        mesh = complete_mesh(pcd)
        mesh_path = output_dir / "mesh_completed.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        written.append(mesh_path)

    return RefineResult(output_dir=output_dir, written_files=written, id2label=id2label)


def _color_by_label(pcd: o3d.geometry.PointCloud, labels: np.ndarray) -> o3d.geometry.PointCloud:
    colored = o3d.geometry.PointCloud(pcd)
    rng = np.random.default_rng(_PALETTE_SEED)
    unique = sorted(set(labels.tolist()))
    palette = {cls: rng.random(3) for cls in unique}
    colors = np.array([palette[c] for c in labels])
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored
