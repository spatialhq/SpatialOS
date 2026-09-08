"""Trajectory Score (design doc weight: 25%): loop closure + path length >5m.

Derived post-hoc from the pose stream since we don't have live SLAM
telemetry. This is an estimate, not the app's real-time signal.
"""

from __future__ import annotations

import numpy as np

from spatial_os.schema import Frame

TARGET_PATH_LENGTH_M = 5.0
LOOP_CLOSURE_DISTANCE_M = 1.0


def path_length_m(frames: list[Frame]) -> float:
    if len(frames) < 2:
        return 0.0
    positions = np.array([f.pose_matrix()[:3, 3] for f in frames])
    deltas = np.diff(positions, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def has_loop_closure(frames: list[Frame], distance_threshold: float = LOOP_CLOSURE_DISTANCE_M) -> bool:
    if len(frames) < 2:
        return False
    start = frames[0].pose_matrix()[:3, 3]
    end = frames[-1].pose_matrix()[:3, 3]
    return bool(np.linalg.norm(end - start) < distance_threshold)


def trajectory_score(frames: list[Frame], loop_closure_override: bool | None = None) -> tuple[float, bool]:
    """Returns (score_0_100, estimated). `estimated` is False when loop_closure_override is provided."""
    length = path_length_m(frames)
    length_score = min(length / TARGET_PATH_LENGTH_M, 1.0) * 60.0

    estimated = loop_closure_override is None
    loop_closed = has_loop_closure(frames) if loop_closure_override is None else loop_closure_override
    loop_score = 40.0 if loop_closed else 0.0

    return length_score + loop_score, estimated
