"""GPU-free unit tests for the depth-of-field optics math."""
import math

import numpy as np
import pytest

import app as studio


def test_focal35_is_rotation_invariant():
    """35mm-equivalent must use the sensor diagonal, not the width.

    A 50mm full-frame lens on a 3:2 sensor reports 50mm whether the frame
    is landscape (6000x4000) or portrait (4000x6000).
    """
    f_px = 50.0 * math.hypot(6000, 4000) / math.hypot(36, 24)
    assert studio.focal35_from_fpx(f_px, 6000, 4000) == pytest.approx(50.0, abs=0.01)
    assert studio.focal35_from_fpx(f_px, 4000, 6000) == pytest.approx(50.0, abs=0.01)


def test_disparity_to_metres():
    """Inverse depth converts back to metres; sky (0) becomes the stats cap."""
    inv = np.array([[0.5, 0.0], [0.1, 0.0]], dtype=np.float32)  # 2 m, sky, 10 m, sky
    depth_m = studio.disparity_to_metres(inv)
    assert depth_m[0, 0] == pytest.approx(2.0)
    assert depth_m[1, 0] == pytest.approx(10.0)
    assert depth_m[0, 1] == pytest.approx(2000.0)


def test_true_disparity_removes_network_normalization():
    """The network outputs f_px/width * 1/z; the shaders need true 1/z (1/m)."""
    iw, f_px = 1280, 1044.9
    z = np.array([2.0, 10.0, 100.0], dtype=np.float32)
    normalized = (f_px / iw) / z
    d = studio.true_disparity(normalized, iw, f_px)
    np.testing.assert_allclose(d, 1.0 / z, rtol=1e-5)


def test_upsample_radius_scales_with_factor():
    """Guided-filter radius must grow with the coarse->full upsample factor."""
    small = studio.upsample_radius(256, 256, 16, 16)   # 16x upsample
    mid = studio.upsample_radius(1440, 1920, 960, 1280)  # 1.5x
    assert 1 <= mid <= 2, "a 1.5x upsample must use a tight window (no halo)"
    assert small > mid, "a 16x upsample needs a wider window"
    assert small <= 8

