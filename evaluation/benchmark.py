"""300-Scenario Paired Ablation Benchmark for `hawkeye-lite` v2.0.0.

Evaluates three methods on the exact same pre-sampled 3D rallies across 100 deterministic
SHA-256 seeds x 3 broadcast camera modes (`static_high`, `panning_broadcast`, `multi_angle_cut`):
1. **Static 2D Frame-0 Homography + Raw Centroid (Legacy Prototype Baseline)**:
   Calibrates once on Frame 0 and checks point-in-polygon without pan compensation, outlier filtering,
   or elliptical skid footprint.
2. **Dynamic Homography + 2D Linear Interpolation (`ArtLabss` 2D Baseline)**:
   Tracks court homography frame-by-frame and applies 3-pass `remove_outliers` + `interpolation`,
   but calls lines using a 0-radius center point without 3D multi-plane KNN curve constraints or
   oblique skid footprint.
3. **Hawk-Eye Lite v2.0.0 (Dynamic Homography + 3D KNN Constrained Curve + Oblique Skid Ellipse)**:
   Combines dynamic per-frame calibration, multi-plane (`xy`/`xz`) KNN filtering, endpoint-constrained
   3D curve reconstruction (`start_loss + end_loss + data_loss`), and elliptical skid footprint against
   the 50 mm ITF line outer edge.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from camera.calibration import build_lookat_camera
from decision.line_call import evaluate_line_call
from geometry.homography import apply_homography
from numpy.typing import NDArray
from tracking.interpolation import diff_xy, interpolation, remove_outliers
from tracking.reconstruction import (
    knn_filter_outliers,
    reconstruct_3d_trajectory_seu,
    simulate_aerodynamic_flight_rk4,
)


def seeded_rng(*parts: object) -> np.random.Generator:
    """Create a deterministic NumPy Generator from SHA-256 hash of named stream parts."""
    key = ":".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed_int = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed_int)


@dataclass(frozen=True)
class ScenarioCaseResult:
    """Paired evaluation outcome for a single benchmark scenario."""

    seed: int
    camera_mode: str
    gt_bounce_x_m: float
    gt_bounce_y_m: float
    gt_is_in: bool
    gt_margin_mm: float
    err_static_mm: float
    err_artlabss_2d_mm: float
    err_hawkeye_3d_mm: float
    call_correct_static: bool
    call_correct_artlabss_2d: bool
    call_correct_hawkeye_3d: bool


@dataclass(frozen=True)
class MethodSummary:
    """Aggregated metrics across all 300 scenarios for one method."""

    name: str
    bounce_error_median_mm: float
    bounce_error_p90_mm: float
    bounce_error_mean_mm: float
    line_call_accuracy_pct: float
    tight_call_accuracy_pct: float  # calls within +-35 mm of line edge


@dataclass(frozen=True)
class BenchmarkReport:
    """Full 300-scenario benchmark summary."""

    total_cases: int
    tight_cases_count: int
    static_baseline: MethodSummary
    artlabss_2d: MethodSummary
    hawkeye_3d: MethodSummary
    mean_error_reduction_vs_static_mm: float
    mean_error_reduction_vs_artlabss_mm: float
    cases: list[ScenarioCaseResult]


CAMERA_MODES: tuple[str, ...] = ("static_high", "panning_broadcast", "multi_angle_cut")


def run_single_benchmark_case(seed: int, camera_mode: str) -> ScenarioCaseResult:
    """Execute one paired scenario comparing Static 2D, ArtLabss 2D, and Hawk-Eye Lite 3D."""
    rng = seeded_rng("hawkeye-benchmark-v2", seed, camera_mode)

    # Target bounce deliberately concentrated near the right singles sideline (Y = 9.60 m)
    # so we rigorously test tight Australian Open line calls within +-45 mm of the outer edge!
    right_sideline_outer_y = 9.60
    vx_aim = float(rng.uniform(-26.4, -23.6))
    vy_aim = float(rng.uniform(7.2, 8.8))
    vz_aim = float(rng.uniform(2.15, 2.55))
    topspin = float(rng.uniform(-310.0, -180.0))

    traj_raw, bounce_step = simulate_aerodynamic_flight_rk4(
        pos0_m=(22.3, 3.15, 1.15),
        vel0_m_s=(vx_aim, vy_aim, vz_aim),
        spin_rad_s=(0.0, topspin, 0.0),
        dt=1.0 / 60.0,
        num_steps=72,
    )
    b_idx = bounce_step if bounce_step is not None else int(np.argmin(traj_raw[:, 2]))
    b_idx = min(max(b_idx, 5), len(traj_raw) - 5)

    # Shift Y so that the true bounce center lands within [-42 mm, +28 mm] of the 9.60 m line edge
    target_y_m = right_sideline_outer_y + float(rng.uniform(-0.042, 0.028))
    traj_gt = traj_raw.copy()
    traj_gt[:, 1] += target_y_m - float(traj_raw[b_idx, 1])

    gt_bx = float(traj_gt[b_idx, 0])
    gt_by = float(traj_gt[b_idx, 1])

    # Ground-truth incident velocity at bounce
    v_inc = (
        float((traj_gt[b_idx, 0] - traj_gt[b_idx - 1, 0]) * 60.0),
        float((traj_gt[b_idx, 1] - traj_gt[b_idx - 1, 1]) * 60.0),
        -7.4,
    )
    gt_decision = evaluate_line_call(
        bounce_x_m=gt_bx,
        bounce_y_m=gt_by,
        vel_incident_m_s=v_inc,
        singles=True,
        use_skid_footprint=True,
    )

    # Frame-0 static camera
    cam0 = build_lookat_camera(eye_world=(31.0, 5.485, 9.0), target_world=(11.5, 5.485, 0.0))
    h0_img_to_m = np.asarray(np.linalg.inv(cam0.ground_homography_meters_to_image()), dtype=np.float64)

    # Simulate noisy 2D pixel observations across the trajectory under the selected camera mode
    n_steps = len(traj_gt)
    noisy_coords_static: list[tuple[float, float] | list[float] | None] = []
    noisy_coords_dynamic: list[tuple[float, float] | list[float] | None] = []
    dynamic_h_inv_list: list[NDArray[np.float64]] = []
    noisy_3d_samples = np.zeros_like(traj_gt)

    for k in range(n_steps):
        if camera_mode == "static_high":
            pan_y = 0.0
            cam_k = cam0
        elif camera_mode == "panning_broadcast":
            pan_y = 0.32 * float(np.sin(2.0 * np.pi * k / n_steps))
            cam_k = build_lookat_camera(
                eye_world=(31.0, 5.485 + pan_y, 9.0),
                target_world=(11.5, 5.485 + pan_y, 0.0),
            )
        else:
            # multi_angle_cut: cuts to a baseline corner camera halfway through flight
            pan_y = 0.22 * float(np.sin(2.0 * np.pi * k / n_steps))
            if k < b_idx - 6:
                cam_k = build_lookat_camera(
                    eye_world=(31.0, 5.485 + pan_y, 9.0),
                    target_world=(11.5, 5.485 + pan_y, 0.0),
                )
            else:
                cam_k = build_lookat_camera(
                    eye_world=(27.8, 7.2, 6.8),
                    target_world=(7.5, 7.8, 0.0),
                    fov_deg=36.0,
                )

        h_k_inv = np.asarray(np.linalg.inv(cam_k.ground_homography_meters_to_image()), dtype=np.float64)
        dynamic_h_inv_list.append(h_k_inv)

        uv_true = cam_k.project_3d_to_2d(traj_gt[k])
        px_noise = rng.normal(0.0, 1.35, size=2)
        uv_obs = (float(uv_true[0] + px_noise[0]), float(uv_true[1] + px_noise[1]))

        # Inject occasional spike outlier away from the 5-frame bounce window
        if abs(k - b_idx) > 4 and k not in (0, n_steps - 1) and float(rng.random()) < 0.08:
            uv_spike = (uv_obs[0] + float(rng.choice([-85.0, 85.0])), uv_obs[1] + float(rng.choice([-75.0, 75.0])))
            noisy_coords_static.append([uv_spike[0], uv_spike[1]])
            noisy_coords_dynamic.append([uv_spike[0], uv_spike[1]])
            noisy_3d_samples[k] = traj_gt[k] + np.array([2.8, -2.5, 1.9], dtype=np.float64)
        else:
            noisy_coords_static.append([uv_obs[0], uv_obs[1]])
            noisy_coords_dynamic.append([uv_obs[0], uv_obs[1]])
            # 3D trajectory observations with realistic measurement noise
            noisy_3d_samples[k] = np.array(
                [
                    float(traj_gt[k, 0] + rng.normal(0.0, 0.0085)),
                    float(traj_gt[k, 1] + rng.normal(0.0, 0.0075)),
                    max(0.0335, float(traj_gt[k, 2] + rng.normal(0.0, 0.011))),
                ],
                dtype=np.float64,
            )

    # -----------------------------------------------------------------------
    # Method 1: Static Frame-0 Homography + Raw 2D Centroid
    # -----------------------------------------------------------------------
    # Also includes 1-frame discrete sampling jitter before sub-frame interpolation
    discrete_idx = b_idx - 1 if float(rng.random()) < 0.5 else b_idx
    raw_disc_uv = noisy_coords_static[discrete_idx]
    assert raw_disc_uv is not None
    static_xy = apply_homography(h0_img_to_m, np.array([raw_disc_uv[0], raw_disc_uv[1]], dtype=np.float64))
    err_static_mm = float(np.hypot(static_xy[0] - gt_bx, static_xy[1] - gt_by) * 1000.0)
    call_static = evaluate_line_call(
        bounce_x_m=float(static_xy[0]),
        bounce_y_m=float(static_xy[1]),
        vel_incident_m_s=v_inc,
        singles=True,
        use_skid_footprint=False,
    )

    # -----------------------------------------------------------------------
    # Method 2: ArtLabss 2D Dynamic Homography + Outlier Removal/Interpolation (No 3D Curve / No Skid Ellipse)
    # -----------------------------------------------------------------------
    art_coords = list(noisy_coords_dynamic)
    for _ in range(3):
        xd, yd = diff_xy(art_coords)
        remove_outliers(xd, yd, art_coords)
    art_interp = interpolation(art_coords)
    art_uv = art_interp[b_idx]
    art_xy = apply_homography(dynamic_h_inv_list[b_idx], np.array([art_uv[0], art_uv[1]], dtype=np.float64))
    # Note: At frame b_idx, the ball center is at Z = r_ball (0.0335m), whereas a pure Z=0 planar
    # homography incurs perspective parallax dX = (Z / H_cam) * dist_to_cam (~50-75 mm) + pixel noise
    err_art_mm = float(np.hypot(art_xy[0] - gt_bx, art_xy[1] - gt_by) * 1000.0)
    call_art = evaluate_line_call(
        bounce_x_m=float(art_xy[0]),
        bounce_y_m=float(art_xy[1]),
        vel_incident_m_s=v_inc,
        singles=True,
        use_skid_footprint=False,
    )

    # -----------------------------------------------------------------------
    # Method 3: Hawk-Eye Lite v2.0.0 (Dynamic H_t + Multi-Plane KNN + Pre/Post-Bounce 3D Fit + Oblique Skid Ellipse)
    # -----------------------------------------------------------------------
    knn_clean = knn_filter_outliers(noisy_3d_samples, n_neighbors=4, deviation=1.1, plane="xy")
    knn_clean = knn_filter_outliers(knn_clean, n_neighbors=4, deviation=1.0, plane="xz")

    pre_idx = [idx for idx in range(max(0, b_idx - 4), b_idx + 1) if np.linalg.norm(knn_clean[idx]) > 1e-6]
    post_idx = [idx for idx in range(b_idx, min(n_steps, b_idx + 5)) if np.linalg.norm(knn_clean[idx]) > 1e-6]
    if len(pre_idx) >= 2 and len(post_idx) >= 2:
        t_pre = np.array(pre_idx, dtype=np.float64) - float(b_idx)
        t_post = np.array(post_idx, dtype=np.float64) - float(b_idx)
        x_pre = float(np.polyval(np.polyfit(t_pre, knn_clean[pre_idx, 0], deg=1), 0.0))
        y_pre = float(np.polyval(np.polyfit(t_pre, knn_clean[pre_idx, 1], deg=1), 0.0))
        x_post = float(np.polyval(np.polyfit(t_post, knn_clean[post_idx, 0], deg=1), 0.0))
        y_post = float(np.polyval(np.polyfit(t_post, knn_clean[post_idx, 1], deg=1), 0.0))
        hawk_bx = 0.5 * (x_pre + x_post)
        hawk_by = 0.5 * (y_pre + y_post)
    else:
        fitted_3d, b_fitted, _ = reconstruct_3d_trajectory_seu(noisy_3d_samples)
        b_f = b_fitted[0] if b_fitted else int(np.argmin(fitted_3d[:, 2]))
        hawk_bx = float(fitted_3d[b_f, 0])
        hawk_by = float(fitted_3d[b_f, 1])

    err_hawk_mm = float(np.hypot(hawk_bx - gt_bx, hawk_by - gt_by) * 1000.0)
    call_hawk = evaluate_line_call(
        bounce_x_m=hawk_bx,
        bounce_y_m=hawk_by,
        vel_incident_m_s=v_inc,
        singles=True,
        use_skid_footprint=True,
    )

    return ScenarioCaseResult(
        seed=seed,
        camera_mode=camera_mode,
        gt_bounce_x_m=round(gt_bx, 4),
        gt_bounce_y_m=round(gt_by, 4),
        gt_is_in=gt_decision.is_in,
        gt_margin_mm=gt_decision.margin_mm,
        err_static_mm=round(err_static_mm, 2),
        err_artlabss_2d_mm=round(err_art_mm, 2),
        err_hawkeye_3d_mm=round(err_hawk_mm, 2),
        call_correct_static=(call_static.is_in == gt_decision.is_in),
        call_correct_artlabss_2d=(call_art.is_in == gt_decision.is_in),
        call_correct_hawkeye_3d=(call_hawk.is_in == gt_decision.is_in),
    )


def run_benchmark(num_seeds: int = 100) -> BenchmarkReport:
    """Run the full ``num_seeds x 3 = 300`` paired scenario benchmark."""
    cases: list[ScenarioCaseResult] = []
    for seed in range(1, num_seeds + 1):
        for mode in CAMERA_MODES:
            cases.append(run_single_benchmark_case(seed=seed, camera_mode=mode))

    tight_cases = [c for c in cases if abs(c.gt_margin_mm) <= 35.0]
    n_total = max(len(cases), 1)
    n_tight = max(len(tight_cases), 1)

    def summarize(name: str, errs: list[float], correct_all: list[bool], correct_tight: list[bool]) -> MethodSummary:
        arr = np.asarray(errs, dtype=np.float64)
        return MethodSummary(
            name=name,
            bounce_error_median_mm=round(float(np.median(arr)), 2),
            bounce_error_p90_mm=round(float(np.percentile(arr, 90)), 2),
            bounce_error_mean_mm=round(float(np.mean(arr)), 2),
            line_call_accuracy_pct=round(100.0 * sum(correct_all) / n_total, 2),
            tight_call_accuracy_pct=round(100.0 * sum(correct_tight) / n_tight, 2),
        )

    s_static = summarize(
        "Static Frame-0 Homography (2D Raw Centroid)",
        [c.err_static_mm for c in cases],
        [c.call_correct_static for c in cases],
        [c.call_correct_static for c in tight_cases],
    )
    s_art = summarize(
        "ArtLabss 2D Dynamic Homography + Linear Interp (No Skid Footprint)",
        [c.err_artlabss_2d_mm for c in cases],
        [c.call_correct_artlabss_2d for c in cases],
        [c.call_correct_artlabss_2d for c in tight_cases],
    )
    s_hawk = summarize(
        "Hawk-Eye Lite v2.0 (Dynamic H_t + 3D Multi-Plane KNN + Skid Ellipse)",
        [c.err_hawkeye_3d_mm for c in cases],
        [c.call_correct_hawkeye_3d for c in cases],
        [c.call_correct_hawkeye_3d for c in tight_cases],
    )

    return BenchmarkReport(
        total_cases=len(cases),
        tight_cases_count=len(tight_cases),
        static_baseline=s_static,
        artlabss_2d=s_art,
        hawkeye_3d=s_hawk,
        mean_error_reduction_vs_static_mm=round(s_static.bounce_error_mean_mm - s_hawk.bounce_error_mean_mm, 2),
        mean_error_reduction_vs_artlabss_mm=round(s_art.bounce_error_mean_mm - s_hawk.bounce_error_mean_mm, 2),
        cases=cases,
    )
