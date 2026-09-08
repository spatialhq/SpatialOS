"""Geometry cleanup & hole-filling via Poisson surface reconstruction.

Note: this is classical geometry processing (Open3D's Poisson reconstruction),
not a learned/AI completion model. Deep-learning shape-completion models
(e.g. shape priors, diffusion-based completion) need GPU inference and
task-specific pretrained weights that aren't a reliable drop-in dependency;
Poisson reconstruction is the well-understood, dependency-light way to turn a
noisy/incomplete point cloud (with normals, which postprocessing already
estimates) into a watertight-ish mesh that fills small gaps. Swap this
implementation for a learned completion model later if a suitable pretrained
checkpoint is identified.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def complete_mesh(
    pcd: o3d.geometry.PointCloud, depth: int = 9, density_trim_quantile: float = 0.02
) -> o3d.geometry.TriangleMesh:
    if not pcd.has_normals():
        pcd.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_trim_quantile)
    low_density_vertices = densities < threshold
    mesh.remove_vertices_by_mask(low_density_vertices)

    return mesh
