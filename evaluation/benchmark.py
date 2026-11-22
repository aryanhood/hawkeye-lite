"""300-Scenario Honest 2D-Pixel-Only Multi-View vs. Monocular Benchmark.

All 4 methods operate strictly from independent noisy 2D pixel observations
(sigma = 1.0 px + 6% random +/-80 px clutter spikes), with:
- ZERO oracle 3D coordinates
- ZERO oracle bounce frame indices
- ZERO oracle impact velocities
- IDENTICAL ITF skid-footprint rule across all primary comparisons
- Separate controlled ablation isolating skid-footprint impact and uncertainty calibration
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
    detect_bounces_artlabss,
    monocular_parallax_bound_m,
    reconstruct_multiview_bounce,
    simulate_aerodynamic_rally_exact,
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
    err_static_1cam_mm: float
    err_dynamic_1cam_mm: float
    err_dlt_2cam_mm: float
    err_dlt_4cam_mm: float
    call_correct_static_1cam: bool
    call_correct_dynamic_1cam: bool
    call_correct_dlt_2cam: bool
    call_correct_dlt_4cam: bool
    call_correct_2cam_no_skid: bool
    brier_2cam: float
    brier_4cam: float
    ci95_covered_2cam: bool
    ci95_covered_4cam: bool


@dataclass(frozen=True)
class MethodSummary:
    """Aggregated metrics across all 300 scenarios for one method."""

    name: str
    bounce_error_median_mm: float
    bounce_error_p90_mm: float
    bounce_error_mean_mm: float
    line_call_accuracy_pct: float
    tight_call_accuracy_pct: float
    brier_score: float | None = None
    ci95_coverage_pct: float | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Full 300-scenario honest 2D-pixel-only benchmark summary."""

    total_cases: int
    tight_cases_count: int
    tight_in_rate_pct: float
    static_1cam: MethodSummary
    dynamic_1cam: MethodSummary
    dlt_2cam: MethodSummary
    dlt_4cam: MethodSummary
    skid_ablation_2cam_no_skid_acc_pct: float
    skid_ablation_2cam_no_skid_tight_pct: float
    mean_error_reduction_2cam_vs_1cam_mm: float
    mean_error_reduction_4cam_vs_2cam_mm: float
    cases: list[ScenarioCaseResult]


CAMERA_MODES: tuple[str, ...] = ("static_high", "panning_broadcast", "multi_angle_cut")


def _estimate_monocular_bounce_idx(coords: list[tuple[float, float]]) -> int:
    """Estimate bounce frame purely from 2D pixel vertical impulse cusp (no ground truth)."""
    b_frames, _ = detect_bounces_artlabss(coords, fps=60.0)
    valid = [b for b in b_frames if 4 <= b <= len(coords) - 5]
    if valid:
        return valid[0]
    # Upward ground restitution produces a strong negative peak in image-v second difference d2v/dt2
    v_vals = np.array([c[1] for c in coords], dtype=np.float64)
    interior = np.arange(4, len(coords) - 4)
    d2v = v_vals[interior + 2] - 2.0 * v_vals[interior] + v_vals[interior - 2]
    return int(interior[int(np.argmin(d2v))])


def run_single_benchmark_case(seed: int, camera_mode: str) -> ScenarioCaseResult:
    """Execute one paired scenario where all 4 methods start strictly from noisy 2D pixels."""
    rng = seeded_rng("hawkeye-honest-v1", seed, camera_mode)

    right_sideline_outer_y = 9.60
    vx_aim = float(rng.uniform(-26.4, -23.6))
    vy_aim = float(rng.uniform(7.2, 8.8))
    vz_aim = float(rng.uniform(2.15, 2.55))
    topspin = float(rng.uniform(-310.0, -180.0))

    traj_raw, b_step, _, pos_b_raw, vel_inc_raw = simulate_aerodynamic_rally_exact(
        pos0_m=(22.3, 3.15, 1.15),
        vel0_m_s=(vx_aim, vy_aim, vz_aim),
        spin_rad_s=(0.0, topspin, 0.0),
        dt=1.0 / 60.0,
        num_steps=72,
    )
    b_idx_gt = b_step if b_step is not None else int(np.argmin(traj_raw[:, 2]))
    assert pos_b_raw is not None
    assert vel_inc_raw is not None

    # Center tight calls around (right_sideline_outer_y + 23mm nominal skid) so ~50% IN / ~50% OUT
    target_y_m = right_sideline_outer_y + 0.023 + float(rng.uniform(-0.045, 0.045))
    y_shift = target_y_m - float(pos_b_raw[1])
    traj_gt = traj_raw.copy()
    traj_gt[:, 1] += y_shift

    gt_bx = float(pos_b_raw[0])
    gt_by = float(pos_b_raw[1] + y_shift)

    gt_decision = evaluate_line_call(
        bounce_x_m=gt_bx,
        bounce_y_m=gt_by,
        vel_incident_m_s=vel_inc_raw,
        singles=True,
        use_skid_footprint=True,
    )

    # Calibrated stadium cameras
    cam0 = build_lookat_camera(eye_world=(31.0, 5.485, 9.0), target_world=(11.5, 5.485, 0.0))
    cam_side = build_lookat_camera(eye_world=(-5.5, 11.5, 8.5), target_world=(10.5, 5.5, 0.0))
    cam_far = build_lookat_camera(eye_world=(-7.0, 2.0, 9.0), target_world=(12.0, 6.0, 0.0))
    cam_opp = build_lookat_camera(eye_world=(24.0, -4.5, 8.0), target_world=(11.5, 6.0, 0.0))

    h0_inv = np.linalg.inv(cam0.ground_homography_meters_to_image())
    n_steps = len(traj_gt)

    p_cam0_seq: list[NDArray[np.float64]] = []
    h_cam0_inv_seq: list[NDArray[np.float64]] = []
    uv_cams: list[NDArray[np.float64]] = [np.zeros((n_steps, 2), dtype=np.float64) for _ in range(4)]
    raw_cam0_list: list[tuple[float, float] | list[float] | None] = []

    other_cams = [cam_side, cam_far, cam_opp]

    for k in range(n_steps):
        if camera_mode == "static_high":
            cam_k = cam0
        elif camera_mode == "panning_broadcast":
            pan_y = 0.32 * float(np.sin(2.0 * np.pi * k / n_steps))
            cam_k = build_lookat_camera(
                eye_world=(31.0, 5.485 + pan_y, 9.0),
                target_world=(11.5, 5.485 + pan_y, 0.0),
            )
        else:
            pan_y = 0.22 * float(np.sin(2.0 * np.pi * k / n_steps))
            if k < b_idx_gt - 6:
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

        p_cam0_seq.append(cam_k.p_mat)
        h_cam0_inv_seq.append(np.linalg.inv(cam_k.ground_homography_meters_to_image()))

        cams_k = [cam_k, *other_cams]
        for c_idx, cam_obj in enumerate(cams_k):
            uv_true = cam_obj.project_3d_to_2d(traj_gt[k])
            uv_noisy = uv_true + rng.normal(0.0, 1.0, size=2)
            # Independent 6% clutter spikes (including near bounce, NO oracle exclusion!)
            if k not in (0, n_steps - 1) and float(rng.random()) < 0.06:
                uv_noisy += np.array(
                    [float(rng.choice([-80.0, 80.0])), float(rng.choice([-70.0, 70.0]))],
                    dtype=np.float64,
                )
            uv_cams[c_idx][k] = uv_noisy
        raw_cam0_list.append([float(uv_cams[0][k, 0]), float(uv_cams[0][k, 1])])

    # -----------------------------------------------------------------------
    # 1. 1-Cam Static Frame-0 Homography (no outlier removal, no dynamic H_t)
    # -----------------------------------------------------------------------
    b_est_static = _estimate_monocular_bounce_idx([(float(p[0]), float(p[1])) for p in uv_cams[0]])
    uv_s = uv_cams[0][b_est_static]
    xy_s = apply_homography(h0_inv, uv_s)
    err_static_mm = float(np.hypot(xy_s[0] - gt_bx, xy_s[1] - gt_by) * 1000.0)
    call_static = evaluate_line_call(
        bounce_x_m=float(xy_s[0]),
        bounce_y_m=float(xy_s[1]),
        vel_incident_m_s=(-24.0, 7.5, -7.2),
        singles=True,
        use_skid_footprint=True,
        homography_img_to_m=h0_inv,
        bounce_uv_px=(float(uv_s[0]), float(uv_s[1])),
        extra_sigma_mm=monocular_parallax_bound_m(0.06, 20.0, 9.0) * 1000.0,
    )

    # -----------------------------------------------------------------------
    # 2. 1-Cam Dynamic Per-Frame Homography + Outlier Removal + Interp
    # -----------------------------------------------------------------------
    clean_2d = list(raw_cam0_list)
    for _ in range(3):
        xd, yd = diff_xy(clean_2d)
        remove_outliers(xd, yd, clean_2d)
    interp_2d = interpolation(clean_2d)
    b_est_dyn = _estimate_monocular_bounce_idx(interp_2d)
    uv_d = interp_2d[b_est_dyn]
    h_dyn = h_cam0_inv_seq[b_est_dyn]
    xy_d = apply_homography(h_dyn, np.array([uv_d[0], uv_d[1]], dtype=np.float64))
    prev_k = max(0, b_est_dyn - 2)
    xy_prev = apply_homography(
        h_cam0_inv_seq[prev_k],
        np.array([interp_2d[prev_k][0], interp_2d[prev_k][1]], dtype=np.float64),
    )
    dt_2d = max(1, b_est_dyn - prev_k) / 60.0
    vx_2d = float(xy_d[0] - xy_prev[0]) / dt_2d
    vy_2d = float(xy_d[1] - xy_prev[1]) / dt_2d

    err_dyn_mm = float(np.hypot(xy_d[0] - gt_bx, xy_d[1] - gt_by) * 1000.0)
    call_dyn = evaluate_line_call(
        bounce_x_m=float(xy_d[0]),
        bounce_y_m=float(xy_d[1]),
        vel_incident_m_s=(vx_2d, vy_2d, -7.2),
        singles=True,
        use_skid_footprint=True,
        homography_img_to_m=h_dyn,
        bounce_uv_px=(float(uv_d[0]), float(uv_d[1])),
        extra_sigma_mm=monocular_parallax_bound_m(0.06, 20.0, 9.0) * 1000.0,
    )

    # -----------------------------------------------------------------------
    # 3. 2-Cam Calibrated DLT 3D + KNN + Sub-Frame Ballistic Fit
    # -----------------------------------------------------------------------
    res2 = reconstruct_multiview_bounce(
        [p_cam0_seq, cam_side.p_mat],
        [uv_cams[0], uv_cams[1]],
        fps=60.0,
        pixel_sigma_px=1.0,
    )
    bx2, by2 = res2.bounce_xy_m
    err_2cam_mm = float(np.hypot(bx2 - gt_bx, by2 - gt_by) * 1000.0)
    call_2cam = evaluate_line_call(
        bounce_x_m=bx2,
        bounce_y_m=by2,
        vel_incident_m_s=res2.vel_incident_m_s,
        singles=True,
        use_skid_footprint=True,
        explicit_sigma_xy_mm=(res2.sigma_x_mm, res2.sigma_y_mm),
    )
    call_2cam_noskid = evaluate_line_call(
        bounce_x_m=bx2,
        bounce_y_m=by2,
        vel_incident_m_s=res2.vel_incident_m_s,
        singles=True,
        use_skid_footprint=False,
        explicit_sigma_xy_mm=(res2.sigma_x_mm, res2.sigma_y_mm),
    )

    # -----------------------------------------------------------------------
    # 4. 4-Cam Stadium DLT 3D + KNN + Sub-Frame Ballistic Fit
    # -----------------------------------------------------------------------
    res4 = reconstruct_multiview_bounce(
        [p_cam0_seq, cam_side.p_mat, cam_far.p_mat, cam_opp.p_mat],
        uv_cams,
        fps=60.0,
        pixel_sigma_px=1.0,
    )
    bx4, by4 = res4.bounce_xy_m
    err_4cam_mm = float(np.hypot(bx4 - gt_bx, by4 - gt_by) * 1000.0)
    call_4cam = evaluate_line_call(
        bounce_x_m=bx4,
        bounce_y_m=by4,
        vel_incident_m_s=res4.vel_incident_m_s,
        singles=True,
        use_skid_footprint=True,
        explicit_sigma_xy_mm=(res4.sigma_x_mm, res4.sigma_y_mm),
    )

    gt_y_num = 1.0 if gt_decision.is_in else 0.0
    brier2 = float((call_2cam.probability_in - gt_y_num) ** 2)
    brier4 = float((call_4cam.probability_in - gt_y_num) ** 2)
    cov2 = abs(by2 - gt_by) * 1000.0 <= 1.96 * res2.sigma_y_mm
    cov4 = abs(by4 - gt_by) * 1000.0 <= 1.96 * res4.sigma_y_mm

    return ScenarioCaseResult(
        seed=seed,
        camera_mode=camera_mode,
        gt_bounce_x_m=round(gt_bx, 4),
        gt_bounce_y_m=round(gt_by, 4),
        gt_is_in=gt_decision.is_in,
        gt_margin_mm=gt_decision.margin_mm,
        err_static_1cam_mm=round(err_static_mm, 2),
        err_dynamic_1cam_mm=round(err_dyn_mm, 2),
        err_dlt_2cam_mm=round(err_2cam_mm, 2),
        err_dlt_4cam_mm=round(err_4cam_mm, 2),
        call_correct_static_1cam=(call_static.is_in == gt_decision.is_in),
        call_correct_dynamic_1cam=(call_dyn.is_in == gt_decision.is_in),
        call_correct_dlt_2cam=(call_2cam.is_in == gt_decision.is_in),
        call_correct_dlt_4cam=(call_4cam.is_in == gt_decision.is_in),
        call_correct_2cam_no_skid=(call_2cam_noskid.is_in == gt_decision.is_in),
        brier_2cam=round(brier2, 5),
        brier_4cam=round(brier4, 5),
        ci95_covered_2cam=cov2,
        ci95_covered_4cam=cov4,
    )


def run_benchmark(num_seeds: int = 100) -> BenchmarkReport:
    """Run the full ``num_seeds x 3 = 300`` honest 2D-pixel-only paired scenario benchmark."""
    cases: list[ScenarioCaseResult] = []
    for seed in range(1, num_seeds + 1):
        for mode in CAMERA_MODES:
            cases.append(run_single_benchmark_case(seed=seed, camera_mode=mode))

    tight_cases = [c for c in cases if abs(c.gt_margin_mm) <= 35.0]
    n_total = max(len(cases), 1)
    n_tight = max(len(tight_cases), 1)
    tight_in_rate = 100.0 * sum(1 for c in tight_cases if c.gt_is_in) / n_tight

    def summarize(
        name: str,
        errs: list[float],
        corr_all: list[bool],
        corr_tight: list[bool],
        briers: list[float] | None = None,
        cov95: list[bool] | None = None,
    ) -> MethodSummary:
        arr = np.asarray(errs, dtype=np.float64)
        return MethodSummary(
            name=name,
            bounce_error_median_mm=round(float(np.median(arr)), 2),
            bounce_error_p90_mm=round(float(np.percentile(arr, 90)), 2),
            bounce_error_mean_mm=round(float(np.mean(arr)), 2),
            line_call_accuracy_pct=round(100.0 * sum(corr_all) / n_total, 2),
            tight_call_accuracy_pct=round(100.0 * sum(corr_tight) / n_tight, 2),
            brier_score=round(float(np.mean(briers)), 4) if briers else None,
            ci95_coverage_pct=round(100.0 * sum(cov95) / n_total, 2) if cov95 else None,
        )

    s_static = summarize(
        "1-Cam Static Frame-0 Homography",
        [c.err_static_1cam_mm for c in cases],
        [c.call_correct_static_1cam for c in cases],
        [c.call_correct_static_1cam for c in tight_cases],
    )
    s_dyn = summarize(
        "1-Cam Dynamic Homography + 2D Outlier Interp",
        [c.err_dynamic_1cam_mm for c in cases],
        [c.call_correct_dynamic_1cam for c in cases],
        [c.call_correct_dynamic_1cam for c in tight_cases],
    )
    s_2cam = summarize(
        "2-Cam DLT 3D + KNN + Sub-Frame Ballistic Fit",
        [c.err_dlt_2cam_mm for c in cases],
        [c.call_correct_dlt_2cam for c in cases],
        [c.call_correct_dlt_2cam for c in tight_cases],
        [c.brier_2cam for c in cases],
        [c.ci95_covered_2cam for c in cases],
    )
    s_4cam = summarize(
        "4-Cam Stadium DLT 3D + KNN + Sub-Frame Ballistic Fit",
        [c.err_dlt_4cam_mm for c in cases],
        [c.call_correct_dlt_4cam for c in cases],
        [c.call_correct_dlt_4cam for c in tight_cases],
        [c.brier_4cam for c in cases],
        [c.ci95_covered_4cam for c in cases],
    )

    no_skid_all = round(100.0 * sum(c.call_correct_2cam_no_skid for c in cases) / n_total, 2)
    no_skid_tight = round(100.0 * sum(c.call_correct_2cam_no_skid for c in tight_cases) / n_tight, 2)

    return BenchmarkReport(
        total_cases=len(cases),
        tight_cases_count=len(tight_cases),
        tight_in_rate_pct=round(tight_in_rate, 2),
        static_1cam=s_static,
        dynamic_1cam=s_dyn,
        dlt_2cam=s_2cam,
        dlt_4cam=s_4cam,
        skid_ablation_2cam_no_skid_acc_pct=no_skid_all,
        skid_ablation_2cam_no_skid_tight_pct=no_skid_tight,
        mean_error_reduction_2cam_vs_1cam_mm=round(s_dyn.bounce_error_mean_mm - s_2cam.bounce_error_mean_mm, 2),
        mean_error_reduction_4cam_vs_2cam_mm=round(s_2cam.bounce_error_mean_mm - s_4cam.bounce_error_mean_mm, 2),
        cases=cases,
    )
