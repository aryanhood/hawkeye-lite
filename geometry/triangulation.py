"""3D projection matrix estimation, multi-view DLT triangulation, and monocular plane-ray lifting.

Implements:
1. ``calc_projection_matrix`` (from ``SEU-Robot-Vision-Project-tennis-Hawkeye-system``
   ``least_square_solution_net.py``, solved analytically via Hartley-normalized SVD
   rather than a 10,000-epoch PyTorch loop).
2. ``triangulate_dlt`` and ``triangulate_opencv`` (wrapping ``cv2.triangulatePoints``
   with SEU's ``inside_range`` court volume validation).
3. ``intersect_ray_with_plane`` for monocular 2D pixel -> 3D court-plane / known-height lifting.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from geometry.homography import normalize_points_2d


def pflat(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Divide homogeneous coordinates ``(D, N)`` by their last row (SEU ``pflat`` helper)."""
    y = np.array(x, dtype=np.float64, copy=True)
    denom = np.where(np.abs(y[-1:, :]) < 1e-12, 1e-12, y[-1:, :])
    return y / denom


def inside_range(point: NDArray[np.float64]) -> bool:
    """SEU court volume sanity check: ``X in (-1, 25)``, ``Y in (-1, 13)``, ``Z in (-0.1, 100)``."""
    return bool(-1.0 < float(point[0]) < 25.0 and -1.0 < float(point[1]) < 13.0 and -0.1 < float(point[2]) < 100.0)


def calc_projection_matrix(
    points_3d: NDArray[np.float64],
    points_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Estimate ``3x4`` camera projection matrix ``P`` from 3D-2D correspondences (``N >= 6``).

    Replaces the 10,000-epoch PyTorch ``LinearSolver`` in SEU's ``least_square_solution_net.py``
    with an exact, deterministic Hartley-normalized Direct Linear Transform (SVD).
    """
    p3 = np.asarray(points_3d, dtype=np.float64)
    p2 = np.asarray(points_2d, dtype=np.float64)
    if p3.shape[0] < 6 or p3.shape[1] != 3 or p2.shape != (p3.shape[0], 2):
        msg = f"Expected (N>=6, 3) and (N>=6, 2), got {p3.shape} and {p2.shape}"
        raise ValueError(msg)

    # Normalize 2D points
    p2_norm, t_2d = normalize_points_2d(p2)

    # Normalize 3D points
    centroid_3d = np.mean(p3, axis=0)
    shifted_3d = p3 - centroid_3d
    mean_dist_3d = float(np.mean(np.linalg.norm(shifted_3d, axis=1)))
    scale_3d = np.sqrt(3.0) / max(mean_dist_3d, 1e-9)
    t_3d = np.array(
        [
            [scale_3d, 0.0, 0.0, -scale_3d * float(centroid_3d[0])],
            [0.0, scale_3d, 0.0, -scale_3d * float(centroid_3d[1])],
            [0.0, 0.0, scale_3d, -scale_3d * float(centroid_3d[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p3_h = np.column_stack([p3, np.ones(len(p3), dtype=np.float64)])
    p3_norm = (t_3d @ p3_h.T).T

    n = len(p3)
    a_mat = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        row_x = p3_norm[i]
        u, v = float(p2_norm[i, 0]), float(p2_norm[i, 1])
        a_mat[2 * i, 0:4] = row_x
        a_mat[2 * i, 8:12] = -u * row_x
        a_mat[2 * i + 1, 4:8] = row_x
        a_mat[2 * i + 1, 8:12] = -v * row_x

    _, _, vh = np.linalg.svd(a_mat)
    p_norm = vh[-1].reshape(3, 4)
    p_mat = np.linalg.inv(t_2d) @ p_norm @ t_3d

    # Enforce positive depth sign in front of camera (SEU testsign check)
    test_proj = p_mat @ p3_h[0]
    if float(test_proj[2]) < 0.0:
        p_mat = -p_mat

    norm_f = float(np.linalg.norm(p_mat[2, :3]))
    if norm_f > 1e-12:
        p_mat = p_mat / norm_f
    return np.asarray(p_mat, dtype=np.float64)


def triangulate_dlt(
    proj_matrices: list[NDArray[np.float64]],
    image_points: list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Triangulate a 3D point ``(X, Y, Z)`` from ``M >= 2`` camera projection matrices and 2D pixels."""
    if len(proj_matrices) < 2 or len(proj_matrices) != len(image_points):
        msg = "At least 2 matching projection matrices and 2D points are required"
        raise ValueError(msg)

    a_rows: list[NDArray[np.float64]] = []
    for p_mat, (u, v) in zip(proj_matrices, image_points, strict=True):
        a_rows.append(u * p_mat[2] - p_mat[0])
        a_rows.append(v * p_mat[2] - p_mat[1])

    a_mat = np.asarray(a_rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(a_mat)
    x_h = vh[-1]
    w = float(x_h[3]) if abs(float(x_h[3])) > 1e-12 else 1e-12
    return np.asarray(x_h[:3] / w, dtype=np.float64)


def triangulate_pair_seu(
    p_left: NDArray[np.float64],
    p_right: NDArray[np.float64],
    pts_left: NDArray[np.float64],
    pts_right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """SEU ``RECONSTRUCTION.py`` pairwise triangulation using ``cv2.triangulatePoints``.

    Missing observations ``(0, 0)`` or out-of-range 3D points are set to ``(0, 0, 0)``.
    """
    arr_l = np.asarray(pts_left, dtype=np.float64)
    arr_r = np.asarray(pts_right, dtype=np.float64)
    if arr_l.shape != arr_r.shape or arr_l.ndim != 2 or arr_l.shape[1] != 2:
        msg = f"Expected matching (N, 2) arrays, got {arr_l.shape} and {arr_r.shape}"
        raise ValueError(msg)

    n = arr_l.shape[0]
    p3d = np.zeros((n, 3), dtype=np.float64)
    for j in range(n):
        if (arr_l[j, 0] == 0.0 and arr_l[j, 1] == 0.0) or (arr_r[j, 0] == 0.0 and arr_r[j, 1] == 0.0):
            continue
        pt4d = cv2.triangulatePoints(
            p_left.astype(np.float64),
            p_right.astype(np.float64),
            arr_l[j].reshape(2, 1),
            arr_r[j].reshape(2, 1),
        )
        w = float(pt4d[3, 0]) if abs(float(pt4d[3, 0])) > 1e-12 else 1e-12
        pt3 = (pt4d[:3, 0] / w).astype(np.float64)
        if inside_range(pt3):
            p3d[j] = pt3
    return p3d


def intersect_ray_with_horizontal_plane(
    p_mat: NDArray[np.float64],
    u: float,
    v: float,
    z_plane: float = 0.0,
) -> NDArray[np.float64]:
    """Lift a monocular 2D pixel ``(u, v)`` to 3D ``(X, Y, z_plane)`` via projection matrix ``P``."""
    # Solve [p00 - u*p20, p01 - u*p21] [X] = [u*(p22*Z + p23) - (p02*Z + p03)]
    #       [p10 - v*p20, p11 - v*p21] [Y]   [v*(p22*Z + p23) - (p12*Z + p13)]
    a_mat = np.array(
        [
            [p_mat[0, 0] - u * p_mat[2, 0], p_mat[0, 1] - u * p_mat[2, 1]],
            [p_mat[1, 0] - v * p_mat[2, 0], p_mat[1, 1] - v * p_mat[2, 1]],
        ],
        dtype=np.float64,
    )
    b_vec = np.array(
        [
            u * (p_mat[2, 2] * z_plane + p_mat[2, 3]) - (p_mat[0, 2] * z_plane + p_mat[0, 3]),
            v * (p_mat[2, 2] * z_plane + p_mat[2, 3]) - (p_mat[1, 2] * z_plane + p_mat[1, 3]),
        ],
        dtype=np.float64,
    )
    xy, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    return np.array([float(xy[0]), float(xy[1]), float(z_plane)], dtype=np.float64)
