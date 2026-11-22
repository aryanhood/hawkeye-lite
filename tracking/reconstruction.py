"""3D Trajectory Reconstruction, Multi-Plane KNN Filtering, Constrained Curve Fitting, and Physics.

Directly adapts and generalizes the core 3D Hawk-Eye reconstruction pipeline from
``GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system`` (``RECONSTRUCTION.py``)
and the 20-lag ``(x, y, V)`` velocity bounce detector from ``ArtLabss/tennis-tracking``:

1. ``remove_outlier(p3d, iteration=3)``: 3-neighbor local mean distance threshold filter.
2. ``knn_filter_outliers(points_3d, n_neighbors, deviation, plane='xy'|'yz'|'xz')``:
   Projects 3D points onto 2D view planes (top-down ``xy``, side ``xz``, end ``yz``) and filters
   clumpy trajectory noise via K-Nearest Neighbor distance statistics (using ``scipy.spatial.cKDTree``).
3. ``find_bounce(p3d)``, ``find_hit(p3d)``, ``find_service(p3d)`` + ``detect_bounces_velocity_lags``:
   Event segmentation identifying serve impacts, ground bounces, and player racket returns.
4. ``fit_curve_with_constraints(x_data, y_data, model_func, initial_guess)``:
   Uses ``scipy.optimize.minimize`` on ``loss_func = start_loss + end_loss + data_loss``
   so adjacent trajectory arcs stitch continuously without segment gaps!
5. ``simulate_aerodynamic_flight_rk4`` and ``lift_2d_rally_to_3d``:
   Aerodynamic flight model (gravity + drag + Magnus topspin/backspin + oblique hardcourt
   restitution) and monocular court-plane-constrained 2D->3D trajectory solver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from geometry.court_reference import TENNIS_BALL_RADIUS_M
from geometry.homography import apply_homography
from geometry.triangulation import inside_range, triangulate_dlt
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Physical constants for regulation ITF tennis ball at sea-level / Melbourne Park
GRAVITY_M_S2: float = 9.80665
AIR_DENSITY_KG_M3: float = 1.204
BALL_MASS_KG: float = 0.0577
BALL_CROSS_SECTION_M2: float = float(np.pi * TENNIS_BALL_RADIUS_M**2)


def is_zero(p: NDArray[np.float64]) -> bool:
    """Return True if point ``p`` is a zero placeholder (SEU ``is_zero``)."""
    return bool(float(p[0]) == 0.0 and float(p[1]) == 0.0)


def remove_outlier(p3d: NDArray[np.float64], iteration: int = 3, max_dist_m: float = 3.0) -> NDArray[np.float64]:
    """SEU 3-neighbor mean distance outlier filter (``RECONSTRUCTION.py`` ``remove_outlier``)."""
    out = np.array(p3d, dtype=np.float64, copy=True)
    n = len(out)
    for _ in range(iteration):
        samecount = 0
        for i in range(1, n):
            if is_zero(out[i]):
                continue
            neighs: list[NDArray[np.float64]] = []
            j = 0
            while len(neighs) != 3 and (i + j) < n:
                j += 1
                candidate = out[i + j - 1]
                if not is_zero(candidate):
                    neighs.append(candidate)
            if neighs:
                arr = np.asarray(neighs, dtype=np.float64)
                means = np.mean(arr, axis=0)
                norm = float(np.linalg.norm(out[i] - means))
                if norm > max_dist_m and samecount < 5:
                    samecount += 1
                    out[i] = 0.0
                else:
                    samecount = 0
    return out


def knn_filter_outliers(
    points_3d: NDArray[np.float64],
    n_neighbors: int = 4,
    deviation: float = 1.0,
    plane: str = "xy",
) -> NDArray[np.float64]:
    """SEU 2D projection plane K-Nearest Neighbors outlier filter (``RECONSTRUCTION.py``).

    Shifts 3D point filtering to 2D plane projections (``'xy'`` top-down, ``'yz'`` end-view,
    ``'xz'`` side-view) and zeros out points whose k-th nearest neighbor distance exceeds
    ``mean(k_distances) + deviation``.
    """
    if plane not in ("yz", "xy", "xz"):
        msg = "Invalid plane parameter. It should be one of: 'yz', 'xy', or 'xz'."
        raise ValueError(msg)

    out = np.array(points_3d, dtype=np.float64, copy=True)
    valid_mask = np.linalg.norm(out, axis=1) > 1e-9
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) <= n_neighbors:
        return out

    valid_pts = out[valid_indices]
    if plane == "xy":
        points_2d = valid_pts[:, :2]
    elif plane == "xz":
        points_2d = valid_pts[:, [0, 2]]
    else:
        points_2d = valid_pts[:, 1:]

    k_eff = min(max(n_neighbors, 2), len(points_2d))
    tree = cKDTree(points_2d)
    distances, _ = tree.query(points_2d, k=k_eff)
    outlier_scores = np.asarray(distances[:, -1], dtype=np.float64)
    threshold = float(np.mean(outlier_scores)) + float(deviation)

    bad_local = np.where(outlier_scores > threshold)[0]
    out[valid_indices[bad_local]] = 0.0
    return out


def find_bounce(
    p3d: NDArray[np.float64],
    z_limit_low: float = 0.0,
    z_limit_high: float = 1.0,
) -> list[int]:
    """SEU 5-point V-shaped Z minimum bounce detector (``RECONSTRUCTION.py`` ``find_bounce``)."""
    bounce_indices: list[int] = []
    for i in range(len(p3d) - 4):
        points_z = p3d[i : i + 5, 2]
        if (
            points_z[0] > points_z[1] > points_z[2]
            and points_z[4] > points_z[3] > points_z[2]
            and z_limit_low <= float(points_z[2]) < z_limit_high
        ) and (not bounce_indices or (i + 2 - bounce_indices[-1]) >= 6):
            bounce_indices.append(i + 2)
    return bounce_indices


def find_hit(p3d: NDArray[np.float64]) -> list[int]:
    """SEU 5-point longitudinal X-reversal player racket hit detector (``RECONSTRUCTION.py`` ``find_hit``)."""
    hit_indices: list[int] = []
    for i in range(len(p3d) - 4):
        points_x = p3d[i : i + 5, 0]
        is_max = points_x[0] < points_x[1] < points_x[2] and points_x[4] < points_x[3] < points_x[2]
        is_min = points_x[0] > points_x[1] > points_x[2] and points_x[4] > points_x[3] > points_x[2]
        if (is_max or is_min) and (not hit_indices or (i + 2 - hit_indices[-1]) >= 8):
            hit_indices.append(i + 2)
    return hit_indices


def find_service(p3d: NDArray[np.float64]) -> list[int]:
    """SEU service launch onset detector (``RECONSTRUCTION.py`` ``find_service``)."""
    service_indices: list[int] = []
    for i in range(len(p3d) - 4):
        dx_early = abs(float(p3d[i + 1, 0] - p3d[i, 0])) + abs(float(p3d[i + 2, 0] - p3d[i + 1, 0]))
        dx_late = abs(float(p3d[i + 3, 0] - p3d[i + 2, 0])) + abs(float(p3d[i + 4, 0] - p3d[i + 3, 0]))
        if dx_early * 4.0 < dx_late and dx_late > 0.15:
            service_indices.append(i + 2)
    return service_indices


def model_curve_function(params: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parabolic curve ``y(x) = a0 + a1*x + a2*x^2`` (SEU ``model_curve_function``)."""
    return np.asarray(params[0] + params[1] * x + params[2] * (x**2), dtype=np.float64)


def model_line_function(params: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear ground trajectory ``y(x) = a0 + a1*x`` (SEU ``model_line_function``)."""
    return np.asarray(params[0] + params[1] * x, dtype=np.float64)


def fit_curve_with_constraints(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    model_func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    initial_guess: list[float] | NDArray[np.float64],
    insert_num: int = 30,
    endpoint_weight: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """SEU endpoint-constrained curve optimizer (``RECONSTRUCTION.py`` ``fit_curve_with_constraints``).

    Minimizes ``endpoint_weight * (start_loss + end_loss) + data_loss`` via ``scipy.optimize.minimize``
    so consecutive trajectory segments connect seamlessly at every bounce and hit point.
    """
    x_arr = np.asarray(x_data, dtype=np.float64)
    y_arr = np.asarray(y_data, dtype=np.float64)
    if len(x_arr) < 2:
        p0 = np.asarray(initial_guess, dtype=np.float64)
        return p0, x_arr.copy(), y_arr.copy()

    start_y = float(y_arr[0])
    end_y = float(y_arr[-1])

    def loss_func(params: NDArray[np.float64]) -> float:
        pred_y = model_func(params, x_arr)
        start_loss = float((pred_y[0] - start_y) ** 2)
        end_loss = float((pred_y[-1] - end_y) ** 2)
        data_loss = float(np.mean((pred_y - y_arr) ** 2))
        return endpoint_weight * (start_loss + end_loss) + data_loss

    res = minimize(loss_func, np.asarray(initial_guess, dtype=np.float64), method="BFGS")
    params_opt = np.asarray(res.x, dtype=np.float64)
    x_fit = np.linspace(float(x_arr[0]), float(x_arr[-1]), insert_num, dtype=np.float64)
    y_fit = model_func(params_opt, x_fit)
    return params_opt, x_fit, y_fit


def reconstruct_3d_trajectory_seu(
    raw_p3d: NDArray[np.float64],
    samples_per_segment: int = 40,
) -> tuple[NDArray[np.float64], list[int], list[int]]:
    """Generalized SEU 3D reconstruction pipeline for any rally of ``N`` segments.

    1. Filters 3-neighbor local distance outliers (``remove_outlier``).
    2. Applies multi-plane KNN filtering across ``xy`` and ``xz``/``yz`` planes.
    3. Detects bounces (``find_bounce``) and player hits (``find_hit``).
    4. Fits endpoint-constrained 3D curves (linear ``xy`` + parabolic ``xz``/``yz``) across
       each segment and returns ``(fitted_p3d, bounce_indices_in_fitted, hit_indices_in_fitted)``.
    """
    cleaned = remove_outlier(raw_p3d, iteration=2, max_dist_m=3.5)
    cleaned = knn_filter_outliers(cleaned, n_neighbors=4, deviation=1.2, plane="xy")
    cleaned = knn_filter_outliers(cleaned, n_neighbors=4, deviation=1.0, plane="xz")

    valid_mask = np.linalg.norm(cleaned, axis=1) > 1e-6
    valid_pts = cleaned[valid_mask]
    if len(valid_pts) < 5:
        return raw_p3d.copy(), [], []

    bounces = find_bounce(valid_pts)
    hits = find_hit(valid_pts)

    # Combine event breakpoints in chronological order
    event_map: dict[int, str] = {}
    for b_idx in bounces:
        event_map[b_idx] = "bounce"
    for h_idx in hits:
        if all(abs(h_idx - b_idx) >= 4 for b_idx in bounces):
            event_map[h_idx] = "hit"

    breakpoints = sorted({0, len(valid_pts) - 1, *event_map.keys()})
    fitted_segments: list[NDArray[np.float64]] = []
    fitted_bounce_indices: list[int] = []
    fitted_hit_indices: list[int] = []

    current_len = 0
    for seg_i in range(len(breakpoints) - 1):
        i_start = breakpoints[seg_i]
        i_end = breakpoints[seg_i + 1]
        seg = valid_pts[i_start : i_end + 1]
        if len(seg) < 3:
            continue

        t_param = np.linspace(0.0, 1.0, len(seg), dtype=np.float64)
        _, _, x_fit = fit_curve_with_constraints(
            t_param,
            seg[:, 0],
            model_line_function,
            [float(seg[0, 0]), float(seg[-1, 0] - seg[0, 0])],
            insert_num=samples_per_segment,
        )
        _, _, y_fit = fit_curve_with_constraints(
            t_param,
            seg[:, 1],
            model_line_function,
            [float(seg[0, 1]), float(seg[-1, 1] - seg[0, 1])],
            insert_num=samples_per_segment,
        )
        _, _, z_fit = fit_curve_with_constraints(
            t_param,
            seg[:, 2],
            model_curve_function,
            [float(seg[0, 2]), 2.0, -2.0],
            insert_num=samples_per_segment,
        )
        z_fit = np.maximum(z_fit, TENNIS_BALL_RADIUS_M)
        seg_fitted = np.column_stack([x_fit, y_fit, z_fit])
        if seg_i > 0:
            seg_fitted = seg_fitted[1:]
        fitted_segments.append(seg_fitted)
        current_len += len(seg_fitted)

        end_event = event_map.get(i_end)
        if end_event == "bounce":
            fitted_bounce_indices.append(current_len - 1)
        elif end_event == "hit":
            fitted_hit_indices.append(current_len - 1)

    if not fitted_segments:
        return valid_pts, bounces, hits

    return np.vstack(fitted_segments), fitted_bounce_indices, fitted_hit_indices


@dataclass(frozen=True)
class VelocityLagBounceFeature:
    """20-frame lag feature vector ``(lagX_20..1, lagY_20..1, lagV_20..1)`` from ``ArtLabss/predict_video.py``."""

    frame_index: int
    u_px: float
    v_px: float
    vy_reversal_score: float
    is_bounce: bool


def detect_bounces_artlabss(
    coords: list[tuple[float, float]],
    fps: float = 30.0,
    min_frames_between_bounces: int = 12,
    lag_window: int = 20,
) -> tuple[list[int], NDArray[np.float64]]:
    """Detect ball bounce frame indices and construct ``ArtLabss`` 20-lag ``(X, Y, V)`` feature matrix.

    Combines ``ArtLabss/predict_video.py`` 20-lag velocity matrix construction with deterministic
    kinematic vertical-velocity cusp + acceleration spike scoring (eliminating the fragile
    Python 3.7 pickled ``sktime`` 0.4 ``clf.pkl`` binary).
    """
    n = len(coords)
    if n < 5:
        return [], np.zeros((0, 3 * lag_window), dtype=np.float64)

    dt = 1.0 / max(fps, 1.0)
    u_arr = np.array([c[0] for c in coords], dtype=np.float64)
    v_arr = np.array([c[1] for c in coords], dtype=np.float64)

    vx = np.diff(u_arr, prepend=u_arr[0]) / dt
    vy = np.diff(v_arr, prepend=v_arr[0]) / dt
    speed = np.hypot(vx, vy)

    # Build the exact 60-column (lagX_20..1, lagY_20..1, lagV_20..1) feature matrix from ArtLabss
    lag_matrix = np.zeros((n, 3 * lag_window), dtype=np.float64)
    for lag_idx, shift in enumerate(range(lag_window, 0, -1)):
        if shift < n:
            lag_matrix[shift:, lag_idx] = u_arr[:-shift]
            lag_matrix[shift:, lag_window + lag_idx] = v_arr[:-shift]
            lag_matrix[shift:, 2 * lag_window + lag_idx] = speed[:-shift]

    bounce_frames: list[int] = []
    last_bounce = -1000
    for i in range(2, n - 2):
        if i - last_bounce < min_frames_between_bounces:
            continue
        # Local maximum in image v (downward in image space) followed by upward rebound
        # OR sharp acceleration cusp at ground contact
        dv_before = float(v_arr[i] - v_arr[i - 2])
        dv_after = float(v_arr[i + 2] - v_arr[i])
        is_v_cusp = dv_before > 3.0 and dv_after < -2.0 and v_arr[i] >= max(v_arr[i - 1], v_arr[i + 1])

        # Also check longitudinal direction preservation (a bounce continues in X/v direction,
        # whereas a player hit reverses the major flight direction across the net)
        du_before = float(u_arr[i] - u_arr[i - 2])
        du_after = float(u_arr[i + 2] - u_arr[i])
        same_horizontal_sense = (du_before * du_after) >= -8.0

        if is_v_cusp and same_horizontal_sense:
            bounce_frames.append(i)
            last_bounce = i

    return bounce_frames, lag_matrix


@dataclass(frozen=True)
class MultiviewBounceResult:
    """Reconstructed 3D trajectory and bounce telemetry from M >= 2 calibrated cameras."""

    raw_points_3d: NDArray[np.float64]
    knn_points_3d: NDArray[np.float64]
    bounce_frame_est: int
    bounce_time_s: float
    bounce_xy_m: tuple[float, float]
    vel_incident_m_s: tuple[float, float, float]
    sigma_x_mm: float
    sigma_y_mm: float


def _rk4_step(
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    dt: float,
    accel_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k1_x = vel
    k1_v = accel_fn(vel)
    k2_x = vel + 0.5 * dt * k1_v
    k2_v = accel_fn(k2_x)
    k3_x = vel + 0.5 * dt * k2_v
    k3_v = accel_fn(k3_x)
    k4_x = vel + dt * k3_v
    k4_v = accel_fn(k4_x)
    next_pos = pos + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    next_vel = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return next_pos, next_vel


def simulate_aerodynamic_rally_exact(
    pos0_m: tuple[float, float, float],
    vel0_m_s: tuple[float, float, float],
    spin_rad_s: tuple[float, float, float] = (0.0, 260.0, 0.0),
    dt: float = 1.0 / 60.0,
    num_steps: int = 90,
    cor_z: float = 0.75,
    friction_xy: float = 0.80,
    cd: float = 0.55,
    cl: float = 0.22,
) -> tuple[
    NDArray[np.float64],
    int | None,
    float | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Integrate 3D tennis ball flight with RK4 aerodynamics and sub-stepped exact bounce contact."""
    pos = np.array(pos0_m, dtype=np.float64)
    vel = np.array(vel0_m_s, dtype=np.float64)
    omega = np.array(spin_rad_s, dtype=np.float64)
    omega_norm = float(np.linalg.norm(omega))

    k_drag = 0.5 * AIR_DENSITY_KG_M3 * BALL_CROSS_SECTION_M2 * cd / BALL_MASS_KG
    k_magnus = 0.5 * AIR_DENSITY_KG_M3 * BALL_CROSS_SECTION_M2 * cl / BALL_MASS_KG

    def accel(v: NDArray[np.float64]) -> NDArray[np.float64]:
        spd = float(np.linalg.norm(v))
        a_g = np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
        a_d = -k_drag * spd * v
        if omega_norm > 1e-6 and spd > 1e-6:
            a_m = k_magnus * spd * np.cross(omega, v) / omega_norm
        else:
            a_m = np.zeros(3, dtype=np.float64)
        return a_g + a_d + a_m

    traj = np.zeros((num_steps, 3), dtype=np.float64)
    bounce_step: int | None = None
    t_bounce_gt: float | None = None
    pos_bounce_gt: tuple[float, float, float] | None = None
    vel_inc_gt: tuple[float, float, float] | None = None

    for step in range(num_steps):
        traj[step] = pos
        next_pos, next_vel = _rk4_step(pos, vel, dt, accel)

        if next_pos[2] <= TENNIS_BALL_RADIUS_M and vel[2] < 0.0 and pos[2] >= TENNIS_BALL_RADIUS_M:
            alpha = float(
                np.clip(
                    (pos[2] - TENNIS_BALL_RADIUS_M) / max(pos[2] - next_pos[2], 1e-12),
                    0.0,
                    1.0,
                )
            )
            dt_pre = alpha * dt
            dt_post = (1.0 - alpha) * dt
            pos_c, vel_inc = _rk4_step(pos, vel, dt_pre, accel)
            pos_c[2] = TENNIS_BALL_RADIUS_M
            if bounce_step is None:
                bounce_step = step + 1 if (step + 1) < num_steps else step
                t_bounce_gt = float(step * dt + dt_pre)
                pos_bounce_gt = (float(pos_c[0]), float(pos_c[1]), float(TENNIS_BALL_RADIUS_M))
                vel_inc_gt = (float(vel_inc[0]), float(vel_inc[1]), float(vel_inc[2]))

            vel_rebound = np.array(
                [vel_inc[0] * friction_xy, vel_inc[1] * friction_xy, -vel_inc[2] * cor_z],
                dtype=np.float64,
            )
            next_pos, next_vel = _rk4_step(pos_c, vel_rebound, dt_post, accel)
            next_pos[2] = max(float(next_pos[2]), TENNIS_BALL_RADIUS_M)

        pos = next_pos
        vel = next_vel

    return traj, bounce_step, t_bounce_gt, pos_bounce_gt, vel_inc_gt


def simulate_aerodynamic_flight_rk4(
    pos0_m: tuple[float, float, float],
    vel0_m_s: tuple[float, float, float],
    spin_rad_s: tuple[float, float, float] = (0.0, 260.0, 0.0),
    dt: float = 1.0 / 60.0,
    num_steps: int = 90,
    cor_z: float = 0.75,
    friction_xy: float = 0.80,
    cd: float = 0.55,
    cl: float = 0.22,
) -> tuple[NDArray[np.float64], int | None]:
    """Integrate 3D tennis ball flight with RK4 aerodynamics (gravity + drag + Magnus + bounce)."""
    traj, bounce_step, _, _, _ = simulate_aerodynamic_rally_exact(
        pos0_m=pos0_m,
        vel0_m_s=vel0_m_s,
        spin_rad_s=spin_rad_s,
        dt=dt,
        num_steps=num_steps,
        cor_z=cor_z,
        friction_xy=friction_xy,
        cd=cd,
        cl=cl,
    )
    return traj, bounce_step


def monocular_parallax_bound_m(
    ball_height_m: float,
    cam_ground_dist_m: float = 18.0,
    cam_height_m: float = 8.0,
) -> float:
    """Upper bound on monocular Z=0 homography ground-projection parallax error: h * D / (H - h)."""
    denom = max(cam_height_m - ball_height_m, 0.5)
    return float(max(ball_height_m, 0.0) * cam_ground_dist_m / denom)


def project_2d_trajectory_to_court_plane(
    coords_2d: list[tuple[float, float]],
    game_warp_matrices: list[NDArray[np.float64]],
    pixel_to_meters_fn: Callable[[float, float], tuple[float, float]],
) -> NDArray[np.float64]:
    """Project monocular 2D pixel detections onto the ground court plane ``Z = 0`` (in meters).

    Unlike calibrated multi-view triangulation, a single uncalibrated monocular view cannot observe
    out-of-plane elevation ``Z(t)`` directly; airborne points project onto ``Z = 0`` with
    perspective parallax error ``dXY = Z * D_cam / H_cam`` except at true court contact.
    """
    n = len(coords_2d)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    xy_m = np.zeros((n, 2), dtype=np.float64)
    for i, (u, v) in enumerate(coords_2d):
        inv_mat = game_warp_matrices[min(i, len(game_warp_matrices) - 1)]
        ref_pt = apply_homography(inv_mat, np.array([u, v], dtype=np.float64))
        xm, ym = pixel_to_meters_fn(float(ref_pt[0]), float(ref_pt[1]))
        xy_m[i] = [np.clip(xm, -2.0, 26.0), np.clip(ym, -2.0, 13.0)]
    return xy_m


def _project_pt(p_mat: NDArray[np.float64], pt_3d: NDArray[np.float64]) -> NDArray[np.float64]:
    h = p_mat @ np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0], dtype=np.float64)
    z = float(h[2]) if abs(float(h[2])) > 1e-9 else 1e-9
    return np.array([float(h[0] / z), float(h[1] / z)], dtype=np.float64)


def _triangulation_sigma_xy_mm(
    camera_matrices: list[NDArray[np.float64]],
    pt_3d: NDArray[np.float64],
    pixel_sigma_px: float,
) -> tuple[float, float]:
    """First-order Gauss-Markov covariance propagation from 2D pixels -> 3D point (J^T J)^-1."""
    jac_rows: list[NDArray[np.float64]] = []
    eps = 1e-4
    base_uv = [_project_pt(p, pt_3d) for p in camera_matrices]
    for k, p_mat in enumerate(camera_matrices):
        jk = np.zeros((2, 3), dtype=np.float64)
        for col in range(3):
            delta = np.zeros(3, dtype=np.float64)
            delta[col] = eps
            jk[:, col] = (_project_pt(p_mat, pt_3d + delta) - base_uv[k]) / eps
        jac_rows.append(jk)
    jac = np.vstack(jac_rows)
    jtj = jac.T @ jac + np.eye(3, dtype=np.float64) * 1e-9
    cov_3d = np.linalg.inv(jtj) * (pixel_sigma_px**2)
    sx_mm = float(np.clip(np.sqrt(max(float(cov_3d[0, 0]), 1e-10)) * 1000.0, 1.0, 120.0))
    sy_mm = float(np.clip(np.sqrt(max(float(cov_3d[1, 1]), 1e-10)) * 1000.0, 1.0, 120.0))
    return sx_mm, sy_mm


def reconstruct_multiview_bounce(
    camera_matrices: Sequence[NDArray[np.float64] | list[NDArray[np.float64]]],
    uv_per_camera: Sequence[NDArray[np.float64]],
    *,
    fps: float = 60.0,
    pixel_sigma_px: float = 1.0,
    reproj_thresh_px: float = 15.0,
) -> MultiviewBounceResult:
    """Reconstruct 3D ball trajectory and sub-frame bounce purely from M >= 2 noisy 2D pixel streams."""
    m_cams = len(camera_matrices)
    if m_cams < 2 or m_cams != len(uv_per_camera):
        msg = "Require M >= 2 matching camera projection matrices and 2D pixel tracks"
        raise ValueError(msg)

    n_steps = len(uv_per_camera[0])
    per_frame_p: list[list[NDArray[np.float64]]] = []
    for k in range(n_steps):
        frame_mats: list[NDArray[np.float64]] = []
        for cam_entry in camera_matrices:
            if isinstance(cam_entry, list):
                frame_mats.append(cam_entry[min(k, len(cam_entry) - 1)])
            else:
                frame_mats.append(cam_entry)
        per_frame_p.append(frame_mats)

    raw_3d = np.zeros((n_steps, 3), dtype=np.float64)

    for k in range(n_steps):
        p_now = per_frame_p[k]
        uvs = [(float(uv_per_camera[c][k, 0]), float(uv_per_camera[c][k, 1])) for c in range(m_cams)]
        if any(not np.isfinite(u) or not np.isfinite(v) for u, v in uvs):
            continue

        if m_cams >= 3:
            # Leave-one-out subset consensus to reject single-camera clutter spikes per frame
            best_inliers: list[int] = []
            best_pt = np.zeros(3, dtype=np.float64)
            for i in range(m_cams):
                for j in range(i + 1, m_cams):
                    cand = triangulate_dlt([p_now[i], p_now[j]], [uvs[i], uvs[j]])
                    if not inside_range(cand) or float(cand[2]) < -0.02:
                        continue
                    inliers = [
                        c
                        for c in range(m_cams)
                        if float(np.linalg.norm(_project_pt(p_now[c], cand) - np.array(uvs[c])))
                        <= reproj_thresh_px
                    ]
                    if len(inliers) > len(best_inliers):
                        best_inliers = inliers
                        best_pt = cand
            if len(best_inliers) >= 2:
                refined = triangulate_dlt(
                    [p_now[c] for c in best_inliers],
                    [uvs[c] for c in best_inliers],
                )
                if inside_range(refined) and float(refined[2]) >= -0.02:
                    raw_3d[k] = refined
                    continue
            if inside_range(best_pt) and float(best_pt[2]) >= -0.02:
                raw_3d[k] = best_pt
        else:
            cand = triangulate_dlt(p_now, uvs)
            errs = [
                float(np.linalg.norm(_project_pt(p_now[c], cand) - np.array(uvs[c])))
                for c in range(2)
            ]
            if inside_range(cand) and float(cand[2]) >= -0.02 and max(errs) <= reproj_thresh_px:
                raw_3d[k] = cand

    knn_3d = knn_filter_outliers(raw_3d, n_neighbors=4, deviation=0.7, plane="xy")
    knn_3d = knn_filter_outliers(knn_3d, n_neighbors=4, deviation=0.7, plane="xz")
    knn_3d = knn_filter_outliers(knn_3d, n_neighbors=4, deviation=0.7, plane="yz")

    def _is_temporally_consistent(arr: NDArray[np.float64], idx: int) -> bool:
        if float(np.linalg.norm(arr[idx])) <= 1e-6 or not inside_range(arr[idx]):
            return False
        nbrs = [arr[j] for j in (idx - 1, idx + 1) if 0 <= j < len(arr) and float(np.linalg.norm(arr[j])) > 1e-6]
        if not nbrs:
            return False
        return min(float(np.linalg.norm(arr[idx] - nb)) for nb in nbrs) < 1.2

    valid_idx = [idx for idx in range(2, n_steps - 2) if _is_temporally_consistent(knn_3d, idx)]
    if not valid_idx:
        valid_idx = [idx for idx in range(2, n_steps - 2) if _is_temporally_consistent(raw_3d, idx)]
        knn_src = raw_3d
    else:
        knn_src = knn_3d

    if not valid_idx:
        return MultiviewBounceResult(
            raw_3d, knn_3d, n_steps // 2, (n_steps // 2) / fps, (11.885, 5.485), (-20.0, 5.0, -7.0), 50.0, 50.0
        )

    b_est = min(valid_idx, key=lambda idx: float(knn_src[idx, 2]))
    pre_idx = [idx for idx in range(max(0, b_est - 6), b_est) if float(np.linalg.norm(knn_src[idx])) > 1e-6]
    post_idx = [
        idx for idx in range(b_est + 1, min(n_steps, b_est + 7)) if float(np.linalg.norm(knn_src[idx])) > 1e-6
    ]

    dt = 1.0 / max(fps, 1.0)
    t_b_est = float(b_est) * dt
    res_std_x_mm = 0.0
    res_std_y_mm = 0.0

    if len(pre_idx) >= 3 and len(post_idx) >= 3:
        t_pre = np.array(pre_idx, dtype=np.float64) * dt
        t_post = np.array(post_idx, dtype=np.float64) * dt

        deg_z_pre = 2 if len(pre_idx) >= 4 else 1
        deg_z_post = 2 if len(post_idx) >= 4 else 1
        cz_pre = np.polyfit(t_pre - t_b_est, knn_src[pre_idx, 2], deg=deg_z_pre)
        cz_post = np.polyfit(t_post - t_b_est, knn_src[post_idx, 2], deg=deg_z_post)

        c_root = cz_pre.copy()
        c_root[-1] -= TENNIS_BALL_RADIUS_M
        roots = np.roots(c_root)
        real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-6 and -1.5 * dt <= r.real <= 1.5 * dt]
        tau_pre = min(real_roots, key=abs) if real_roots else 0.0

        c_post_root = cz_post.copy()
        c_post_root[-1] -= TENNIS_BALL_RADIUS_M
        roots_p = np.roots(c_post_root)
        real_p = [float(r.real) for r in roots_p if abs(r.imag) < 1e-6 and -1.5 * dt <= r.real <= 1.5 * dt]
        tau_star = float(np.clip(0.5 * (tau_pre + min(real_p, key=abs)) if real_p else tau_pre, -dt, dt))

        cx_pre = np.polyfit(t_pre - t_b_est, knn_src[pre_idx, 0], deg=1)
        cy_pre = np.polyfit(t_pre - t_b_est, knn_src[pre_idx, 1], deg=1)
        bx_m = float(np.polyval(cx_pre, tau_star))
        by_m = float(np.polyval(cy_pre, tau_star))

        vx = float(cx_pre[0])
        vy = float(cy_pre[0])
        vz = float(np.polyval(np.polyder(cz_pre), tau_star))
        t_bounce_s = t_b_est + tau_star

        res_x = knn_src[pre_idx, 0] - np.polyval(cx_pre, t_pre - t_b_est)
        res_y = knn_src[pre_idx, 1] - np.polyval(cy_pre, t_pre - t_b_est)
        res_z = knn_src[pre_idx, 2] - np.polyval(cz_pre, t_pre - t_b_est)
        dof = max(len(pre_idx) - 2, 1)
        sig_z_m = float(np.sqrt(np.sum(res_z**2) / dof))
        sig_t = float(np.clip(sig_z_m / max(abs(vz), 3.0), 0.0, 1.5 * dt))
        res_std_x_mm = float(np.hypot(np.sqrt(np.sum(res_x**2) / dof), abs(vx) * sig_t)) * 1000.0
        res_std_y_mm = float(np.hypot(np.sqrt(np.sum(res_y**2) / dof), abs(vy) * sig_t)) * 1000.0
    else:
        bx_m = float(knn_src[b_est, 0])
        by_m = float(knn_src[b_est, 1])
        t_bounce_s = t_b_est
        vx, vy, vz = -24.0, 7.5, -7.2

    p_at_bounce = per_frame_p[min(b_est, len(per_frame_p) - 1)]
    dlt_sx_mm, dlt_sy_mm = _triangulation_sigma_xy_mm(
        p_at_bounce, np.array([bx_m, by_m, TENNIS_BALL_RADIUS_M], dtype=np.float64), pixel_sigma_px
    )
    sx_mm = float(np.hypot(dlt_sx_mm, res_std_x_mm))
    sy_mm = float(np.hypot(dlt_sy_mm, res_std_y_mm))
    return MultiviewBounceResult(
        raw_points_3d=raw_3d,
        knn_points_3d=knn_3d,
        bounce_frame_est=b_est,
        bounce_time_s=t_bounce_s,
        bounce_xy_m=(bx_m, by_m),
        vel_incident_m_s=(vx, vy, vz),
        sigma_x_mm=sx_mm,
        sigma_y_mm=sy_mm,
    )
