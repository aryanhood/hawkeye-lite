"""PinholeCamera model and homography-to-3D-pose decomposition.

Combines:
1. Intrinsic/Extrinsic camera projection ``P = K [R | t]`` and court homography
   decomposition ``H = K [r1, r2, t]`` so monocular broadcast cameras recover full 3D pose.
2. Virtual 3D orbit/zoom camera generator for rendering the Australian Open / SEU
   3D Hawk-Eye replay cut-in at the end of processed videos.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from geometry.court_reference import ITF_COURT_LENGTH_M, ITF_DOUBLES_WIDTH_M, CourtReference
from geometry.homography import apply_homography, estimate_homography_dlt
from numpy.typing import NDArray


@dataclass(frozen=True)
class PinholeCamera:
    """Calibrated 3D pinhole camera ``P = K [R | t]``."""

    k_mat: NDArray[np.float64]  # (3, 3)
    r_mat: NDArray[np.float64]  # (3, 3)
    t_vec: NDArray[np.float64]  # (3,)
    width: int = 1280
    height: int = 720

    @property
    def p_mat(self) -> NDArray[np.float64]:
        """Return the ``3x4`` projection matrix ``P = K [R | t]``."""
        rt = np.column_stack([self.r_mat, self.t_vec.reshape(3, 1)])
        return (self.k_mat @ rt).astype(np.float64)

    @property
    def camera_center_world(self) -> NDArray[np.float64]:
        """Return 3D camera center ``C = -R^T t`` in world coordinates (meters)."""
        return (-self.r_mat.T @ self.t_vec).astype(np.float64)

    def project_3d_to_2d(self, points_3d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project 3D world point(s) ``(N, 3)`` or ``(3,)`` to 2D image pixels ``(u, v)``."""
        arr = np.asarray(points_3d, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, 3)
        pts_h = np.column_stack([arr, np.ones(len(arr), dtype=np.float64)])
        proj = (self.p_mat @ pts_h.T).T
        z = np.where(np.abs(proj[:, 2:3]) < 1e-9, 1e-9, proj[:, 2:3])
        uv = proj[:, :2] / z
        return uv[0] if single else uv

    def ground_homography_meters_to_image(self) -> NDArray[np.float64]:
        """Return ``3x3`` homography mapping metric ground plane ``(X_m, Y_m, 0)`` to image ``(u, v)``."""
        h_cols = np.column_stack([self.r_mat[:, 0], self.r_mat[:, 1], self.t_vec])
        h_mat = self.k_mat @ h_cols
        if abs(h_mat[2, 2]) > 1e-12:
            h_mat = h_mat / h_mat[2, 2]
        return h_mat.astype(np.float64)


def build_lookat_camera(
    eye_world: tuple[float, float, float],
    target_world: tuple[float, float, float],
    width: int = 1280,
    height: int = 720,
    fov_deg: float = 42.0,
) -> PinholeCamera:
    """Construct a virtual ``PinholeCamera`` positioned at ``eye_world`` looking at ``target_world``."""
    eye = np.asarray(eye_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = target - eye
    f_norm = float(np.linalg.norm(forward))
    forward = forward / max(f_norm, 1e-9)

    # Right vector in OpenCV camera convention (+x right, +y down, +z forward)
    right = np.cross(forward, up_world)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        right = right / float(np.linalg.norm(right))

    down = np.cross(forward, right)
    down = down / max(float(np.linalg.norm(down)), 1e-9)

    r_mat = np.vstack([right, down, forward]).astype(np.float64)
    t_vec = (-r_mat @ eye).astype(np.float64)

    f_px = (0.5 * float(width)) / float(np.tan(np.deg2rad(fov_deg * 0.5)))
    k_mat = np.array(
        [
            [f_px, 0.0, width * 0.5],
            [0.0, f_px, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return PinholeCamera(k_mat=k_mat, r_mat=r_mat, t_vec=t_vec, width=width, height=height)


def pose_from_court_homography(
    court_warp_matrix: NDArray[np.float64],
    width: int = 1280,
    height: int = 720,
    court_ref: CourtReference | None = None,
) -> PinholeCamera:
    """Recover a full 3D ``PinholeCamera`` from ``ArtLabss`` ``court_warp_matrix`` (ref canvas -> frame).

    Why SEU's README noted naive ``cv2.solvePnP`` fails without orthogonality constraints:
    An unconstrained least-squares ``3x4`` matrix on coplanar ``Z=0`` points has a zero column
    for ``Z``. By decomposing the metric planar homography ``H_metric = K [r1, r2, t]`` and
    enforcing ``r3 = r1 x r2`` via SVD orthogonalization, we guarantee an orthonormal
    3D rotation matrix ``R in SO(3)`` that accurately projects both ``Z=0`` court lines and
    ``Z>0`` net/ball heights!
    """
    ref = court_ref if court_ref is not None else CourtReference()

    # Map the 4 outer doubles corners from metric (X_m, Y_m) -> ref pixel -> image pixel
    corners_m = np.array(
        [
            [0.0, 0.0],
            [0.0, ITF_DOUBLES_WIDTH_M],
            [ITF_COURT_LENGTH_M, 0.0],
            [ITF_COURT_LENGTH_M, ITF_DOUBLES_WIDTH_M],
        ],
        dtype=np.float64,
    )
    corners_ref = np.array([ref.meters_to_pixel(xm, ym) for xm, ym in corners_m], dtype=np.float64)
    corners_img = apply_homography(court_warp_matrix, corners_ref)

    h_m_to_img = estimate_homography_dlt(corners_m, corners_img)

    f_px = float(width) * 1.15
    k_mat = np.array(
        [
            [f_px, 0.0, width * 0.5],
            [0.0, f_px, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    k_inv = np.linalg.inv(k_mat)
    b_mat = k_inv @ h_m_to_img

    # Ensure positive depth for court center (11.885, 5.485)
    center_proj = h_m_to_img @ np.array([11.885, 5.485, 1.0], dtype=np.float64)
    if float(center_proj[2]) < 0.0:
        b_mat = -b_mat

    norm1 = float(np.linalg.norm(b_mat[:, 0]))
    norm2 = float(np.linalg.norm(b_mat[:, 1]))
    scale = 2.0 / max(norm1 + norm2, 1e-9)

    r1 = b_mat[:, 0] * scale
    r2 = b_mat[:, 1] * scale
    r3 = np.cross(r1, r2)
    t_vec = (b_mat[:, 2] * scale).astype(np.float64)

    # Project [r1, r2, r3] onto SO(3) via SVD
    r_approx = np.column_stack([r1, r2, r3])
    u_svd, _, vh_svd = np.linalg.svd(r_approx)
    r_mat = (u_svd @ vh_svd).astype(np.float64)
    if float(np.linalg.det(r_mat)) < 0.0:
        u_svd[:, -1] *= -1.0
        r_mat = (u_svd @ vh_svd).astype(np.float64)

    # Refine with cv2.solvePnP using the 4 corners on Z=0
    obj_pts = np.column_stack([corners_m, np.zeros(4, dtype=np.float64)]).astype(np.float64)
    img_pts = corners_img.astype(np.float64)
    rvec_init, _ = cv2.Rodrigues(r_mat)
    tvec_init = t_vec.reshape(3, 1).copy()
    ok, rvec_opt, tvec_opt = cv2.solvePnP(
        obj_pts,
        img_pts,
        k_mat,
        np.zeros(4, dtype=np.float64),
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        r_opt, _ = cv2.Rodrigues(rvec_opt)
        cam = PinholeCamera(
            k_mat=k_mat,
            r_mat=np.asarray(r_opt, dtype=np.float64),
            t_vec=np.asarray(tvec_opt, dtype=np.float64).reshape(3),
            width=width,
            height=height,
        )
        # Ensure camera center is above the court plane Z > 0
        if float(cam.camera_center_world[2]) < 0.0:
            # Flip planar ambiguity to upper hemisphere
            cam = build_lookat_camera(
                eye_world=(29.5, 5.485, 8.2),
                target_world=(11.885, 5.485, 0.0),
                width=width,
                height=height,
            )
        return cam

    return PinholeCamera(k_mat=k_mat, r_mat=r_mat, t_vec=t_vec, width=width, height=height)


def interpolate_replay_camera(
    start_cam: PinholeCamera,
    bounce_xy_m: tuple[float, float],
    alpha: float,
) -> PinholeCamera:
    """Interpolate a smooth 3D virtual camera zoom from broadcast view to Hawk-Eye line close-up."""
    t_clamped = float(np.clip(alpha, 0.0, 1.0))
    # Smoothstep easing
    s = t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)

    start_eye = start_cam.camera_center_world
    bx, by = bounce_xy_m
    # End camera positioned 2.6m above and slightly behind the bounce point
    end_eye = np.array([bx + 1.8, by + 0.9, 2.8], dtype=np.float64)
    eye_now = (1.0 - s) * start_eye + s * end_eye

    start_target = np.array([11.885, 5.485, 0.0], dtype=np.float64)
    end_target = np.array([bx, by, 0.0], dtype=np.float64)
    target_now = (1.0 - s) * start_target + s * end_target

    fov_now = 42.0 * (1.0 - s) + 26.0 * s
    return build_lookat_camera(
        eye_world=(float(eye_now[0]), float(eye_now[1]), float(eye_now[2])),
        target_world=(float(target_now[0]), float(target_now[1]), float(target_now[2])),
        width=start_cam.width,
        height=start_cam.height,
        fov_deg=fov_now,
    )
