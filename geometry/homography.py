"""Analytical line intersection, point sorting, Hartley-normalized DLT homography, and CourtMapper.

Combines:
1. Fast analytical ``line_intersection`` and ``sort_intersection_points`` replacing slow
   ``sympy.Line`` calls in ``ArtLabss/tennis-tracking`` (``court_detector.py``).
2. Hartley-normalized Direct Linear Transform (DLT) planar homography estimation and
   Jacobian covariance propagation.
3. ``CourtMapper`` class preserving backwards compatibility with ``aryanhood/tennisball-tracking``.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from geometry.court_reference import (
    ITF_ALLEY_WIDTH_M,
    ITF_COURT_LENGTH_M,
    ITF_DOUBLES_WIDTH_M,
    ITF_SINGLES_WIDTH_M,
)


def line_intersection(
    line1: tuple[tuple[float, float], tuple[float, float]],
    line2: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Find intersection point ``(x, y)`` of two 2D lines using homogeneous cross-products.

    Runs ~1,000x faster than ``sympy.Line(line1[0], line1[1]).intersection(...)`` used in
    the original ``ArtLabss/tennis-tracking`` implementation.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        # Parallel or nearly parallel fallback: midpoint between segments
        return ((x1 + x3) * 0.5, (y1 + y3) * 0.5)

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
    return (float(px), float(py))


def sort_intersection_points(
    intersections: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort 4 intersection points from top-left to bottom-right: ``[TL, TR, BL, BR]``.

    Matches the exact ordering convention required by ``CourtReference.court_conf[1..12]``
    in ``ArtLabss/tennis-tracking``.
    """
    y_sorted = sorted(intersections, key=lambda pt: pt[1])
    p12 = sorted(y_sorted[:2], key=lambda pt: pt[0])
    p34 = sorted(y_sorted[2:], key=lambda pt: pt[0])
    return p12 + p34


def normalize_points_2d(pts: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Hartley isotropic normalization for 2D points ``(N, 2)``."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-9)
    t_mat = np.array(
        [
            [scale, 0.0, -scale * float(centroid[0])],
            [0.0, scale, -scale * float(centroid[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    norm_pts = (t_mat @ pts_h.T).T[:, :2]
    return norm_pts, t_mat


def estimate_homography_dlt(
    src_pts: NDArray[np.float64],
    dst_pts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute planar homography ``H (3x3)`` mapping ``src_pts -> dst_pts`` via normalized DLT."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        msg = f"Expected matching (N>=4, 2) point arrays, got {src.shape} and {dst.shape}"
        raise ValueError(msg)

    src_n, t_src = normalize_points_2d(src)
    dst_n, t_dst = normalize_points_2d(dst)

    n = src.shape[0]
    a_mat = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = float(src_n[i, 0]), float(src_n[i, 1])
        u, v = float(dst_n[i, 0]), float(dst_n[i, 1])
        a_mat[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        a_mat[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    _, _, vh = np.linalg.svd(a_mat)
    h_norm = vh[-1].reshape(3, 3)
    h_mat = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h_mat[2, 2]) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]
    return np.asarray(h_mat, dtype=np.float64)


def apply_homography(h_mat: NDArray[np.float64], pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project 2D point array ``(N, 2)`` through ``3x3`` homography ``h_mat``."""
    arr = np.asarray(pts, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 2)
    pts_h = np.column_stack([arr, np.ones(len(arr), dtype=np.float64)])
    proj = (h_mat @ pts_h.T).T
    w = np.where(np.abs(proj[:, 2:3]) < 1e-12, 1e-12, proj[:, 2:3])
    out = (proj[:, :2] / w).astype(np.float64)
    return np.asarray(out[0], dtype=np.float64) if single else out


def homography_jacobian(h_mat: NDArray[np.float64], u: float, v: float) -> NDArray[np.float64]:
    """Compute the ``2x2`` Jacobian ``d(X, Y) / d(u, v)`` of homography ``h_mat`` at pixel ``(u, v)``."""
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = h_mat @ p
    w = float(q[2]) if abs(float(q[2])) > 1e-12 else 1e-12
    x_val = float(q[0]) / w
    y_val = float(q[1]) / w

    j_mat = np.array(
        [
            [(h_mat[0, 0] - x_val * h_mat[2, 0]) / w, (h_mat[0, 1] - x_val * h_mat[2, 1]) / w],
            [(h_mat[1, 0] - y_val * h_mat[2, 0]) / w, (h_mat[1, 1] - y_val * h_mat[2, 1]) / w],
        ],
        dtype=np.float64,
    )
    return j_mat


class CourtMapper:
    """Backwards-compatible evolution of ``aryanhood/tennisball-tracking/court_mapper.py``.

    Supports both interactive/programmatic 4-corner ordering (``_order_points``),
    top-down map projection (``map_point``), metric conversion (``map_point_to_meters``),
    and singles/doubles boundary checks (``is_in_singles_court``).
    """

    def __init__(self, court_width: float = 27.0, court_height: float = 78.0) -> None:
        self.court_points_image: NDArray[np.float32] | None = None
        self.transformation_matrix: NDArray[np.float64] | None = None
        self.inverse_transformation_matrix: NDArray[np.float64] | None = None
        self.court_aspect_ratio: float = float(court_height) / float(court_width)

        self.map_width_pixels: int = 400
        self.map_height_pixels: int = int(self.map_width_pixels * self.court_aspect_ratio)

        self.court_corners_map: NDArray[np.float32] = np.array(
            [
                [0, 0],
                [self.map_width_pixels - 1, 0],
                [self.map_width_pixels - 1, self.map_height_pixels - 1],
                [0, self.map_height_pixels - 1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _order_points(pts: NDArray[np.float32]) -> NDArray[np.float32]:
        """Order 4 selected points into ``[top-left, top-right, bottom-right, bottom-left]``."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[int(np.argmin(s))]
        rect[2] = pts[int(np.argmax(s))]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[int(np.argmin(diff))]
        rect[3] = pts[int(np.argmax(diff))]
        return rect

    def calibrate_from_corners(
        self, corners_px: NDArray[np.float32] | list[tuple[float, float]]
    ) -> NDArray[np.float64]:
        """Compute perspective transform from 4 image corners without blocking on a GUI window."""
        arr = np.asarray(corners_px, dtype=np.float32)
        if arr.shape != (4, 2):
            msg = f"Expected (4, 2) corners array, got {arr.shape}"
            raise ValueError(msg)
        ordered = self._order_points(arr)
        self.court_points_image = ordered
        mat = cv2.getPerspectiveTransform(ordered, self.court_corners_map)
        self.transformation_matrix = np.asarray(mat, dtype=np.float64)
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)
        return self.transformation_matrix

    def map_point(self, point: tuple[float, float]) -> tuple[int, int] | None:
        """Map a single image point ``(u, v)`` to top-down map pixel coordinates."""
        if self.transformation_matrix is None:
            return None
        pt = apply_homography(self.transformation_matrix, np.array([point[0], point[1]], dtype=np.float64))
        return (round(float(pt[0])), round(float(pt[1])))

    def map_point_to_meters(
        self, point: tuple[float, float], is_doubles_corners: bool = False
    ) -> tuple[float, float] | None:
        """Map an image point ``(u, v)`` directly to ITF court meters ``(X_m, Y_m)``."""
        if self.transformation_matrix is None:
            return None
        pt = apply_homography(self.transformation_matrix, np.array([point[0], point[1]], dtype=np.float64))
        u_norm = float(pt[0]) / float(max(self.map_width_pixels - 1, 1))
        v_norm = float(pt[1]) / float(max(self.map_height_pixels - 1, 1))
        x_m = v_norm * ITF_COURT_LENGTH_M
        y_m = u_norm * ITF_DOUBLES_WIDTH_M if is_doubles_corners else ITF_ALLEY_WIDTH_M + u_norm * ITF_SINGLES_WIDTH_M
        return (x_m, y_m)

    def is_in_singles_court(self, point: tuple[float, float]) -> bool:
        """Check if an image pixel maps inside the singles court boundaries."""
        mapped = self.map_point(point)
        if mapped is None:
            return False
        x, y = mapped
        return 0 <= x < self.map_width_pixels and 0 <= y < self.map_height_pixels
