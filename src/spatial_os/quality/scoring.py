"""Weighted quality score orchestrator, per design doc weights: Coverage 30%,
Depth Quality 25%, Trajectory 25%, Density 20%. Score < 40 -> reject,
40-70 -> accept as low-quality, >70 -> accept fully.
"""

from __future__ import annotations

import open3d as o3d
from pydantic import BaseModel

from spatial_os.quality.coverage import coverage_score
from spatial_os.quality.density import density_score
from spatial_os.quality.depth_quality import depth_quality_score
from spatial_os.quality.trajectory import trajectory_score
from spatial_os.schema import Frame, SessionMetadataOverride, SessionManifest

WEIGHTS = {"coverage": 0.30, "depth_quality": 0.25, "trajectory": 0.25, "density": 0.20}

REJECT_THRESHOLD = 40.0
LOW_QUALITY_THRESHOLD = 70.0


class ScoreComponent(BaseModel):
    score: float
    weight: float
    estimated: bool


class QualityReport(BaseModel):
    total_score: float
    verdict: str  # "reject" | "accept_low_quality" | "accept"
    components: dict[str, ScoreComponent]


def score_session(
    manifest: SessionManifest,
    frames: list[Frame],
    pcd: o3d.geometry.PointCloud,
    outlier_ratio: float,
    override: SessionMetadataOverride | None = None,
) -> QualityReport:
    depth_source = (override.depth_source if override and override.depth_source else manifest.device.depth_source)

    if override and override.coverage_pct is not None:
        cov_score, cov_estimated = override.coverage_pct, False
    else:
        cov_score, cov_estimated = coverage_score(pcd), True

    traj_score, traj_estimated = trajectory_score(
        frames, loop_closure_override=override.has_loop_closure if override else None
    )

    components = {
        "coverage": ScoreComponent(score=cov_score, weight=WEIGHTS["coverage"], estimated=cov_estimated),
        "depth_quality": ScoreComponent(
            score=depth_quality_score(depth_source), weight=WEIGHTS["depth_quality"], estimated=False
        ),
        "trajectory": ScoreComponent(score=traj_score, weight=WEIGHTS["trajectory"], estimated=traj_estimated),
        "density": ScoreComponent(
            score=density_score(pcd, outlier_ratio), weight=WEIGHTS["density"], estimated=True
        ),
    }

    total = sum(c.score * c.weight for c in components.values())

    if total < REJECT_THRESHOLD:
        verdict = "reject"
    elif total < LOW_QUALITY_THRESHOLD:
        verdict = "accept_low_quality"
    else:
        verdict = "accept"

    return QualityReport(total_score=total, verdict=verdict, components=components)
