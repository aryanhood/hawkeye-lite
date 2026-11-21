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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from geometry.court_reference import TENNIS_BALL_RADIUS_M
from geometry.homography import apply_homography
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
    """Integrate 3D tennis ball flight with RK4 aerodynamics (gravity + drag + Magnus spin + bounce).

    State: ``[X, Y, Z, Vx, Vy, Vz]`` in ITF court meters.
    """
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

    for step in range(num_steps):
        traj[step] = pos
        k1_x = vel
        k1_v = accel(vel)

        k2_x = vel + 0.5 * dt * k1_v
        k2_v = accel(k2_x)

        k3_x = vel + 0.5 * dt * k2_v
        k3_v = accel(k3_x)

        k4_x = vel + dt * k3_v
        k4_v = accel(k4_x)

        next_pos = pos + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        next_vel = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Ground contact at Z = TENNIS_BALL_RADIUS_M
        if next_pos[2] <= TENNIS_BALL_RADIUS_M and vel[2] < 0.0:
            next_pos[2] = TENNIS_BALL_RADIUS_M
            next_vel[2] = -next_vel[2] * cor_z
            next_vel[0] *= friction_xy
            next_vel[1] *= friction_xy
            if bounce_step is None:
                bounce_step = step + 1 if (step + 1) < num_steps else step

        pos = next_pos
        vel = next_vel

    return traj, bounce_step


def lift_2d_trajectory_to_3d_court(
    coords_2d: list[tuple[float, float]],
    game_warp_matrices: list[NDArray[np.float64]],
    bounce_frames: list[int],
    pixel_to_meters_fn: Callable[[float, float], tuple[float, float]],
    fps: float = 30.0,
) -> NDArray[np.float64]:
    """Lift a monocular 2D pixel trajectory into 3D ITF court coordinates ``(X_m, Y_m, Z_m)``.

    At every detected bounce frame ``t_b``, the ball contacts the court plane ``Z(t_b) = r_ball``,
    so the inverse court homography ``game_warp_matrix[t_b]`` fixes its exact metric ground coordinates
    ``(X_b, Y_b, r_ball)``. Between endpoints and bounces, ``(X(t), Y(t))`` follows the court-projected
    trajectory while ``Z(t)`` follows a physical ballistic/restitution arc anchored at ``Z(t_b) = r_ball``.
    """
    n = len(coords_2d)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    xy_m = np.zeros((n, 2), dtype=np.float64)
    for i, (u, v) in enumerate(coords_2d):
        inv_mat = game_warp_matrices[min(i, len(game_warp_matrices) - 1)]
        ref_pt = apply_homography(inv_mat, np.array([u, v], dtype=np.float64))
        xm, ym = pixel_to_meters_fn(float(ref_pt[0]), float(ref_pt[1]))
        xy_m[i] = [np.clip(xm, -2.0, 26.0), np.clip(ym, -2.0, 13.0)]

    z_m = np.full(n, 0.95, dtype=np.float64)
    anchors = sorted({0, n - 1, *[b for b in bounce_frames if 0 <= b < n]})
    bounce_set = set(bounce_frames)

    for idx in range(len(anchors) - 1):
        a0 = anchors[idx]
        a1 = anchors[idx + 1]
        span = max(a1 - a0, 1)
        t_dur = float(span) / max(fps, 1.0)
        z0 = TENNIS_BALL_RADIUS_M if a0 in bounce_set else 1.05
        z1 = TENNIS_BALL_RADIUS_M if a1 in bounce_set else 0.95

        for k in range(a0, a1 + 1):
            tau = float(k - a0) / float(span)
            t_sec = tau * t_dur
            # Ballistic height profile z(t) connecting z0 -> z1 under effective gravity
            g_eff = min(GRAVITY_M_S2, 16.0 / max(t_dur * t_dur, 0.25))
            vz0 = (z1 - z0 + 0.5 * g_eff * (t_dur**2)) / max(t_dur, 1e-3)
            zk = z0 + vz0 * t_sec - 0.5 * g_eff * (t_sec**2)
            z_m[k] = max(float(zk), TENNIS_BALL_RADIUS_M)

    return np.column_stack([xy_m, z_m])
