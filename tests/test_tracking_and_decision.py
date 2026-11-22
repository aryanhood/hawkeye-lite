"""Tests for SORT, KalmanFilter, Interpolation, SEU 3D KNN Reconstruction, and Line Calling."""

from __future__ import annotations

import numpy as np
import pytest
from camera.calibration import build_lookat_camera
from decision.line_call import (
    compute_skid_footprint,
    distance_to_nearest_court_line_mm,
    evaluate_line_call,
)
from geometry.court_reference import CourtReference
from hypothesis import given, settings
from hypothesis import strategies as st
from tracking.interpolation import (
    diff_xy,
    interpolation,
    remove_outliers,
    smooth_trajectory_savgol,
)
from tracking.reconstruction import (
    detect_bounces_artlabss,
    find_bounce,
    find_hit,
    find_service,
    fit_curve_with_constraints,
    knn_filter_outliers,
    model_curve_function,
    monocular_parallax_bound_m,
    project_2d_trajectory_to_court_plane,
    reconstruct_3d_trajectory_seu,
    reconstruct_multiview_bounce,
    remove_outlier,
    simulate_aerodynamic_flight_rk4,
    simulate_aerodynamic_rally_exact,
)
from tracking.sort import (
    BallKalmanTracker,
    Sort,
    convert_bbox_to_z,
    convert_x_to_bbox,
    iou_batch,
)


def test_sort_and_ball_kalman_tracker() -> None:
    b1 = np.array([[10.0, 10.0, 50.0, 80.0]], dtype=np.float64)
    b2 = np.array([[12.0, 12.0, 52.0, 82.0]], dtype=np.float64)
    iou = iou_batch(b1, b2)
    assert 0.7 < float(iou[0, 0]) < 1.0

    z = convert_bbox_to_z(b1[0])
    box_back = convert_x_to_bbox(z)
    assert np.allclose(box_back[0], b1[0], atol=1e-4)
    box_scored = convert_x_to_bbox(z, score=0.95)
    assert box_scored.shape == (1, 5)

    mot = Sort(max_age=5, min_hits=1, iou_threshold=0.1)
    out1 = mot.update([b1[0]], [0.9])
    assert len(out1) == 1
    out2 = mot.update([b2[0]], [0.92])
    assert len(out2) == 1
    out_empty = mot.update(None, None)
    assert out_empty.shape[1] == 5

    b_trk = BallKalmanTracker(max_coast_frames=3, gate_distance_px=100.0)
    assert b_trk.step(None) is None
    p0 = b_trk.step((100.0, 200.0))
    assert p0 == (100.0, 200.0)
    p1 = b_trk.step((112.0, 215.0))
    assert p1 is not None
    # Coast across missed frame
    p_coast = b_trk.step(None)
    assert p_coast is not None
    b_trk.reset()
    assert not b_trk.initialized


def test_interpolation_and_outlier_removal() -> None:
    coords: list[tuple[float, float] | list[float] | None] = [
        [100.0, 100.0],
        [110.0, 110.0],
        [450.0, 480.0],  # outlier spike
        [130.0, 130.0],
        None,
        [150.0, 150.0],
    ]
    xd, yd = diff_xy(coords)
    remove_outliers(xd, yd, coords, max_step_px=50.0)
    assert coords[2] is None

    interp = interpolation(coords)
    assert len(interp) == 6
    assert abs(interp[2][0] - 120.0) < 1e-4
    assert abs(interp[4][0] - 140.0) < 1e-4

    arr = np.column_stack([np.linspace(0.0, 10.0, 15), np.linspace(5.0, 25.0, 15)])
    sm = smooth_trajectory_savgol(arr, window_length=7, polyorder=2)
    assert np.allclose(sm, arr, atol=1e-4)


def test_seu_3d_reconstruction_knn_and_aerodynamics() -> None:
    # For a ball traveling in -X (Vx = -25 m/s), topspin is omega_y < 0 (omega x v points in -Z),
    # which dips and bounces earlier than backspin (omega_y > 0, omega x v points in +Z).
    traj_top, b_top = simulate_aerodynamic_flight_rk4(
        pos0_m=(22.0, 5.0, 1.2),
        vel0_m_s=(-25.0, 0.0, 2.5),
        spin_rad_s=(0.0, -280.0, 0.0),
        num_steps=80,
    )
    _traj_back, b_back = simulate_aerodynamic_flight_rk4(
        pos0_m=(22.0, 5.0, 1.2),
        vel0_m_s=(-25.0, 0.0, 2.5),
        spin_rad_s=(0.0, 280.0, 0.0),
        num_steps=80,
    )
    assert b_top is not None
    assert b_back is not None
    assert b_top < b_back

    # Inject 3D outlier and verify remove_outlier + knn_filter_outliers clean it
    noisy = traj_top.copy()
    noisy[10] = np.array([18.0, 11.5, 4.5], dtype=np.float64)
    cleaned1 = remove_outlier(noisy, iteration=2, max_dist_m=2.0)
    assert np.allclose(cleaned1[10], [0.0, 0.0, 0.0])

    noisy2 = traj_top.copy()
    noisy2[15] = np.array([15.0, 10.5, 3.8], dtype=np.float64)
    cleaned_xy = knn_filter_outliers(noisy2, n_neighbors=4, deviation=0.5, plane="xy")
    cleaned_yz = knn_filter_outliers(cleaned_xy, n_neighbors=4, deviation=0.5, plane="yz")
    cleaned_xz = knn_filter_outliers(cleaned_yz, n_neighbors=4, deviation=0.5, plane="xz")
    assert np.allclose(cleaned_xz[15], [0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Invalid plane"):
        knn_filter_outliers(noisy2, plane="invalid")

    bounces = find_bounce(traj_top)
    assert len(bounces) >= 1

    # Construct synthetic X-reversal for find_hit and acceleration for find_service
    hit_seq = np.array(
        [[10.0, 5.0, 1.0], [8.0, 5.0, 1.0], [6.0, 5.0, 1.0], [8.0, 5.0, 1.0], [10.0, 5.0, 1.0], [12.0, 5.0, 1.0]],
        dtype=np.float64,
    )
    assert find_hit(hit_seq) == [2]

    srv_seq = np.array(
        [[23.0, 5.0, 2.5], [22.99, 5.0, 2.5], [22.98, 5.0, 2.5], [22.2, 5.0, 2.3], [21.1, 5.0, 2.0]],
        dtype=np.float64,
    )
    assert find_service(srv_seq) == [2]

    # Test fit_curve_with_constraints endpoint preservation
    x_d = np.linspace(0.0, 1.0, 12)
    y_d = 1.2 + 2.0 * x_d - 3.0 * (x_d**2)
    _, _, y_fit = fit_curve_with_constraints(x_d, y_d, model_curve_function, [1.0, 1.0, -1.0], insert_num=20)
    assert abs(float(y_fit[0]) - float(y_d[0])) < 1e-2
    assert abs(float(y_fit[-1]) - float(y_d[-1])) < 1e-2

    recon_3d, b_fit, _ = reconstruct_3d_trajectory_seu(noisy2)
    assert len(recon_3d) >= 10
    assert len(b_fit) >= 1

    # Test 2D v-cusp bounce detector, court-plane Z=0 projection, and multiview DLT reconstruction
    coords_2d = [(400.0 + 5.0 * k, 200.0 + 12.0 * k if k <= 15 else 380.0 - 10.0 * (k - 15)) for k in range(32)]
    b_frames, lag_mat = detect_bounces_artlabss(coords_2d, fps=30.0)
    assert 15 in b_frames
    assert lag_mat.shape == (32, 60)

    ref = CourtReference()
    proj_2d = project_2d_trajectory_to_court_plane(
        coords_2d=coords_2d,
        game_warp_matrices=[np.eye(3, dtype=np.float64)],
        pixel_to_meters_fn=ref.pixel_to_meters,
    )
    assert proj_2d.shape == (32, 2)
    assert monocular_parallax_bound_m(0.5, 18.0, 9.0) > 1.0

    # Verify exact sub-stepped RK4 + 2-cam multiview DLT bounce localization
    traj_ex, _, _, pos_gt, vel_gt = simulate_aerodynamic_rally_exact(
        pos0_m=(22.0, 4.0, 1.2),
        vel0_m_s=(-25.0, 5.0, 2.3),
        spin_rad_s=(0.0, -240.0, 0.0),
        num_steps=64,
    )
    assert pos_gt is not None
    assert vel_gt is not None
    c1 = build_lookat_camera(eye_world=(31.0, 5.485, 9.0), target_world=(11.5, 5.485, 0.0))
    c2 = build_lookat_camera(eye_world=(-5.5, 11.5, 8.5), target_world=(10.5, 5.5, 0.0))
    mv = reconstruct_multiview_bounce(
        [c1.p_mat, c2.p_mat],
        [c1.project_3d_to_2d(traj_ex), c2.project_3d_to_2d(traj_ex)],
        fps=60.0,
    )
    assert np.hypot(mv.bounce_xy_m[0] - pos_gt[0], mv.bounce_xy_m[1] - pos_gt[1]) * 1000.0 < 35.0


def test_decision_line_call_and_skid_footprint() -> None:
    # Ball center is 12 mm outside the right singles sideline (Y = 9.60 m -> Y = 9.612 m),
    # so without skid footprint it is OUT, but with a ~25 mm skid radius it clips the line -> IN!
    call_with_skid = evaluate_line_call(
        bounce_x_m=6.0,
        bounce_y_m=9.612,
        vel_incident_m_s=(-25.0, 8.0, -7.5),
        singles=True,
        use_skid_footprint=True,
    )
    call_no_skid = evaluate_line_call(
        bounce_x_m=6.0,
        bounce_y_m=9.612,
        vel_incident_m_s=(-25.0, 8.0, -7.5),
        singles=True,
        use_skid_footprint=False,
    )
    assert call_with_skid.is_in
    assert call_with_skid.verdict == "IN"
    assert not call_no_skid.is_in
    assert call_no_skid.verdict == "OUT"

    # Clearly OUT ball (Y = 9.75 m) with homography Jacobian uncertainty propagation
    call_out = evaluate_line_call(
        bounce_x_m=6.0,
        bounce_y_m=9.75,
        vel_incident_m_s=(-25.0, 8.0, -7.5),
        singles=True,
        use_skid_footprint=True,
        homography_img_to_m=np.diag([0.01, 0.01, 1.0]),
        bounce_uv_px=(500.0, 300.0),
    )
    assert not call_out.is_in
    assert call_out.verdict == "OUT"
    assert call_out.margin_mm < -50.0
    assert call_out.confidence > 0.95

    line_name, dist_mm = distance_to_nearest_court_line_mm(6.0, 9.575, singles=True)
    assert line_name == "right_sideline"
    assert dist_mm < 5.0


@given(
    vx=st.floats(min_value=-40.0, max_value=40.0),
    vy=st.floats(min_value=-20.0, max_value=20.0),
    vz=st.floats(min_value=-20.0, max_value=-1.0),
)
@settings(max_examples=25)
def test_property_skid_footprint_major_ge_minor(vx: float, vy: float, vz: float) -> None:
    fp = compute_skid_footprint(10.0, 5.0, (vx, vy, vz))
    assert fp.semi_major_m >= fp.semi_minor_m > 0.0
