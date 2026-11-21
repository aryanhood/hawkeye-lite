"""Tests for CourtReference, Homography DLT, CourtMapper compatibility, and PinholeCamera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from camera.calibration import (
    build_lookat_camera,
    interpolate_replay_camera,
    pose_from_court_homography,
)
from geometry.court_reference import (
    ITF_COURT_LENGTH_M,
    ITF_DOUBLES_WIDTH_M,
    CourtReference,
)
from geometry.homography import (
    CourtMapper,
    apply_homography,
    estimate_homography_dlt,
    homography_jacobian,
    line_intersection,
    sort_intersection_points,
)
from hypothesis import given, settings
from hypothesis import strategies as st


def test_court_reference_masks_and_pixel_meter_roundtrip(tmp_path: Path) -> None:
    ref = CourtReference()
    assert len(ref.court_conf) == 12
    assert len(ref.get_important_lines()) == 20
    assert len(ref.get_extra_parts()) == 2
    assert len(ref.get_line_segments()) == 9

    saved_png = tmp_path / "court_ref.png"
    built = ref.build_court_reference(save_path=saved_png)
    assert saved_png.exists()
    assert built.shape == (ref.court_total_height, ref.court_total_width)

    m0 = ref.get_court_mask(0)
    m1 = ref.get_court_mask(1)
    m2 = ref.get_court_mask(2)
    m3 = ref.get_court_mask(3)
    assert int(np.sum(m0)) > int(np.sum(m1)) > 0
    assert int(np.sum(m0)) > int(np.sum(m2)) > 0
    assert int(np.sum(m0)) > int(np.sum(m3)) > 0

    for xm, ym in ((0.0, 0.0), (11.885, 5.485), (23.77, 10.97), (5.485, 1.37)):
        u_px, v_px = ref.meters_to_pixel(xm, ym)
        xr, yr = ref.pixel_to_meters(u_px, v_px)
        assert abs(xr - xm) < 1e-6
        assert abs(yr - ym) < 1e-6


@given(
    xm=st.floats(min_value=-2.0, max_value=26.0),
    ym=st.floats(min_value=-2.0, max_value=13.0),
)
@settings(max_examples=25)
def test_property_meters_pixel_invertible(xm: float, ym: float) -> None:
    ref = CourtReference()
    u, v = ref.meters_to_pixel(xm, ym)
    x2, y2 = ref.pixel_to_meters(u, v)
    assert abs(x2 - xm) < 1e-5
    assert abs(y2 - ym) < 1e-5


def test_line_intersection_and_homography_dlt() -> None:
    ix, iy = line_intersection(((0.0, 0.0), (10.0, 10.0)), ((0.0, 10.0), (10.0, 0.0)))
    assert abs(ix - 5.0) < 1e-6
    assert abs(iy - 5.0) < 1e-6

    # Parallel line fallback
    px, py = line_intersection(((0.0, 2.0), (10.0, 2.0)), ((0.0, 4.0), (10.0, 4.0)))
    assert px == 0.0
    assert py == 3.0

    unsorted_pts = [(100.0, 200.0), (20.0, 200.0), (100.0, 10.0), (20.0, 10.0)]
    sorted_pts = sort_intersection_points(unsorted_pts)
    assert sorted_pts == [(20.0, 10.0), (100.0, 10.0), (20.0, 200.0), (100.0, 200.0)]

    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]], dtype=np.float64)
    dst = np.array([[120.0, 80.0], [520.0, 85.0], [640.0, 610.0], [60.0, 605.0]], dtype=np.float64)
    h_mat = estimate_homography_dlt(src, dst)
    proj = apply_homography(h_mat, src)
    assert np.allclose(proj, dst, atol=1e-4)

    # Single point apply_homography and finite-difference Jacobian verification
    pt0 = apply_homography(h_mat, np.array([5.0, 10.0], dtype=np.float64))
    j_analytic = homography_jacobian(h_mat, 5.0, 10.0)
    eps = 1e-5
    dx_du = (apply_homography(h_mat, np.array([5.0 + eps, 10.0])) - pt0) / eps
    dx_dv = (apply_homography(h_mat, np.array([5.0, 10.0 + eps])) - pt0) / eps
    assert np.allclose(j_analytic[:, 0], dx_du, atol=1e-4)
    assert np.allclose(j_analytic[:, 1], dx_dv, atol=1e-4)

    with pytest.raises(ValueError, match="Expected matching"):
        estimate_homography_dlt(np.zeros((3, 2)), np.zeros((3, 2)))


def test_court_mapper_compatibility_and_camera_pose() -> None:
    mapper = CourtMapper()
    assert mapper.map_point((100.0, 100.0)) is None
    assert mapper.map_point_to_meters((100.0, 100.0)) is None
    assert not mapper.is_in_singles_court((100.0, 100.0))

    with pytest.raises(ValueError, match="Expected"):
        mapper.calibrate_from_corners(np.zeros((3, 2), dtype=np.float32))

    corners = [(320.0, 180.0), (960.0, 180.0), (1120.0, 640.0), (160.0, 640.0)]
    mapper.calibrate_from_corners(corners)
    center_mapped = mapper.map_point((640.0, 410.0))
    assert center_mapped is not None
    assert mapper.is_in_singles_court((640.0, 410.0))
    assert not mapper.is_in_singles_court((10.0, 10.0))

    xm_s, ym_s = mapper.map_point_to_meters((640.0, 410.0), is_doubles_corners=False) or (0.0, 0.0)
    _xm_d, ym_d = mapper.map_point_to_meters((640.0, 410.0), is_doubles_corners=True) or (0.0, 0.0)
    assert 0.0 < xm_s < ITF_COURT_LENGTH_M
    assert 0.0 < ym_s < ITF_DOUBLES_WIDTH_M
    assert abs(ym_d - 5.485) < 1.0

    # Test pose decomposition from court homography and replay camera interpolation
    ref = CourtReference()
    true_cam = build_lookat_camera(eye_world=(30.5, 5.485, 8.8), target_world=(11.885, 5.485, 0.0))
    src_ref = np.array(ref.court_conf[1], dtype=np.float64)
    dst_3d = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.97, 0.0], [23.77, 0.0, 0.0], [23.77, 10.97, 0.0]],
        dtype=np.float64,
    )
    dst_img = true_cam.project_3d_to_2d(dst_3d)
    h_ref_to_img = estimate_homography_dlt(src_ref, dst_img)

    recovered_cam = pose_from_court_homography(h_ref_to_img, width=1280, height=720, court_ref=ref)
    assert float(recovered_cam.camera_center_world[2]) > 1.0

    mid_cam = interpolate_replay_camera(recovered_cam, bounce_xy_m=(5.485, 9.60), alpha=0.5)
    assert 1.5 < float(mid_cam.camera_center_world[2]) < float(recovered_cam.camera_center_world[2])
