"""Depth Quality Score (design doc weight: 25%). lidar=high, tof=medium, ml_estimated=low."""

from spatial_os.schema import DepthSource

_SCORE_BY_SOURCE = {
    DepthSource.LIDAR: 100.0,
    DepthSource.TOF: 65.0,
    DepthSource.ML_ESTIMATED: 35.0,
}


def depth_quality_score(depth_source: DepthSource) -> float:
    return _SCORE_BY_SOURCE[depth_source]
