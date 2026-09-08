import cv2
import numpy as np

from spatial_os.config import ProcessConfig
from spatial_os.preprocessing.blur_filter import is_blurry, laplacian_variance
from spatial_os.preprocessing.depth_normalize import mask_invalid_depth, normalize_depth_to_meters
from spatial_os.preprocessing.unpack import prepare_frames
from spatial_os.schema import SessionManifest


def test_sharp_image_has_higher_variance_than_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (25, 25), 10)

    assert laplacian_variance(sharp) > laplacian_variance(blurred)
    assert not is_blurry(sharp, threshold=1.0)
    assert is_blurry(blurred, threshold=laplacian_variance(sharp))


def test_normalize_depth_mm_round_trip():
    raw = np.array([[1000, 2000]], dtype=np.uint16)
    meters = normalize_depth_to_meters(raw, "mm")
    assert np.allclose(meters, [[1.0, 2.0]])


def test_normalize_depth_unknown_unit_raises():
    raw = np.zeros((2, 2), dtype=np.uint16)
    try:
        normalize_depth_to_meters(raw, "furlongs")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mask_invalid_depth_zeroes_out_of_range():
    depth = np.array([[0.01, 1.0, 10.0]], dtype=np.float32)
    masked = mask_invalid_depth(depth, depth_max=5.0, depth_min=0.05)
    assert masked[0, 0] == 0.0  # too close
    assert masked[0, 1] == 1.0  # valid
    assert masked[0, 2] == 0.0  # too far


def test_prepare_frames_drops_blurry_and_normalizes(synthetic_session_dir):
    manifest = SessionManifest.load(synthetic_session_dir / "manifest.json")
    cfg = ProcessConfig(blur_threshold=0.0)  # keep everything, checkerboard frames are sharp

    prepared = prepare_frames(manifest, synthetic_session_dir, cfg)

    assert len(prepared) == len(manifest.frames)
    for pf in prepared:
        assert pf.depth_m.max() <= cfg.depth_max
        assert pf.rgb.shape[:2] == (manifest.intrinsics.height, manifest.intrinsics.width)
