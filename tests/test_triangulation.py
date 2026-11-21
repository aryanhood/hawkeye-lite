"""Tests for 3D projection matrix estimation, multi-view DLT triangulation, and plane ray lifting."""

from __future__ import annotations

import numpy as np
import pytest
from camera.calibration import build_lookat_camera
from geometry.court_reference import WORLD_COORDINATES_3D
from geometry.triangulation import (
    calc_projection_matrix,
    inside_range,
    intersect_ray_with_horizontal_plane,
    pflat,
    triangulate_dlt,
    triangulate_pair_seu,
)


def test_pflat_and_inside_range() -> None:
    arr = np.array([[4.0, 6.0], [8.0, 9.0], [2.0, 3.0]], dtype=np.float64)
    flat = pflat(arr)
    assert np.allclose(flat[:, 0], [2.0, 4.0, 1.0])
    assert np.allclose(flat[:, 1], [2.0, 3.0, 1.0])

    assert inside_range(np.array([11.885, 5.485, 1.2], dtype=np.float64))
    assert not inside_range(np.array([-5.0, 5.485, 1.2], dtype=np.float64))


def test_calc_projection_matrix_and_multi_view_triangulation() -> None:
    cam_left = build_lookat_camera(eye_world=(29.0, 1.5, 8.5), target_world=(11.885, 5.485, 0.5))
    cam_right = build_lookat_camera(eye_world=(29.0, 9.5, 8.5), target_world=(11.885, 5.485, 0.5))

    # Project all 26 3D world keypoints (including Z=1.2m net keypoints) into both cameras
    pts_2d_left = cam_left.project_3d_to_2d(WORLD_COORDINATES_3D)
    pts_2d_right = cam_right.project_3d_to_2d(WORLD_COORDINATES_3D)

    p_est_left = calc_projection_matrix(WORLD_COORDINATES_3D, pts_2d_left)
    p_est_right = calc_projection_matrix(WORLD_COORDINATES_3D, pts_2d_right)

    # Verify reprojection error of estimated P matrix is near zero (< 1e-4 px)
    pts_h = np.column_stack([WORLD_COORDINATES_3D, np.ones(len(WORLD_COORDINATES_3D))])
    reproj_l = (p_est_left @ pts_h.T).T
    reproj_l = reproj_l[:, :2] / reproj_l[:, 2:3]
    assert float(np.max(np.linalg.norm(reproj_l - pts_2d_left, axis=1))) < 1e-3

    # Triangulate an aerial 3D ball point from the two estimated projection matrices
    true_ball_3d = np.array([6.25, 8.45, 1.35], dtype=np.float64)
    uv_l = cam_left.project_3d_to_2d(true_ball_3d)
    uv_r = cam_right.project_3d_to_2d(true_ball_3d)

    recon_3d = triangulate_dlt(
        [p_est_left, p_est_right], [(float(uv_l[0]), float(uv_l[1])), (float(uv_r[0]), float(uv_r[1]))]
    )
    assert np.allclose(recon_3d, true_ball_3d, atol=1e-4)

    # Also test SEU pairwise cv2.triangulatePoints wrapper with valid and missing (0, 0) entries
    batch_l = np.array([[uv_l[0], uv_l[1]], [0.0, 0.0]], dtype=np.float64)
    batch_r = np.array([[uv_r[0], uv_r[1]], [100.0, 100.0]], dtype=np.float64)
    seu_out = triangulate_pair_seu(p_est_left, p_est_right, batch_l, batch_r)
    assert np.allclose(seu_out[0], true_ball_3d, atol=1e-4)
    assert np.allclose(seu_out[1], [0.0, 0.0, 0.0])


def test_intersect_ray_with_horizontal_plane_and_errors() -> None:
    cam = build_lookat_camera(eye_world=(30.0, 5.485, 9.0), target_world=(11.885, 5.485, 0.0))
    pt_3d = np.array([5.485, 9.60, 0.0335], dtype=np.float64)
    uv = cam.project_3d_to_2d(pt_3d)
    lifted = intersect_ray_with_horizontal_plane(cam.p_mat, float(uv[0]), float(uv[1]), z_plane=0.0335)
    assert np.allclose(lifted, pt_3d, atol=1e-5)

    with pytest.raises(ValueError, match="Expected"):
        calc_projection_matrix(np.zeros((4, 3)), np.zeros((4, 2)))

    with pytest.raises(ValueError, match="At least 2"):
        triangulate_dlt([cam.p_mat], [(100.0, 200.0)])

    with pytest.raises(ValueError, match="Expected matching"):
        triangulate_pair_seu(cam.p_mat, cam.p_mat, np.zeros((3, 2)), np.zeros((4, 2)))
