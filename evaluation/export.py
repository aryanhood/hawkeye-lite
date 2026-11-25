"""Deterministic exporter for `docs/benchmark.md` and `web/public/data/hawkeye_data.json`.

Supports `--check` mode for CI drift verification.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from camera.calibration import build_lookat_camera
from decision.line_call import evaluate_line_call
from geometry.court_reference import (
    ITF_COURT_LENGTH_M,
    ITF_DOUBLES_WIDTH_M,
    SEU_WIREFRAME_GROUPS,
    WORLD_COORDINATES_3D,
    CourtReference,
)
from tracking.reconstruction import (
    reconstruct_3d_trajectory_seu,
    reconstruct_multiview_bounce,
    simulate_aerodynamic_rally_exact,
)

from evaluation.benchmark import BenchmarkReport, run_benchmark, seeded_rng
from evaluation.synthetic_video import SCENARIO_PRESETS


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    """Format the 300-case honest 2D-pixel-only benchmark report into Markdown."""
    s1 = report.static_1cam
    s2 = report.dynamic_1cam
    m2 = report.dlt_2cam
    m4 = report.dlt_4cam

    lines = [
        "# Benchmark Report: Multi-View DLT vs. Monocular Court-Plane Projection",
        "",
        "All 4 methods operate **strictly from independent noisy 2D pixel observations**",
        "(`sigma = 1.0 px`, `6%` random `+/-80 px` clutter spikes across `100 SHA-256 seeds x 3 camera modes`)",
        "with **zero oracle 3D coordinates, zero oracle bounce indices, zero oracle impact speeds**,",
        "and **identical ITF skid-footprint rules** across every primary comparison.",
        f"Tight calls within `+/-35 mm` (`N = {report.tight_cases_count}`) are balanced at "
        f"`{report.tight_in_rate_pct:.1f}%` ground-truth `IN`.",
        "",
        "## Primary 300-Scenario Comparison (2D Pixels Only)",
        "",
        f"| Metric | {s1.name} | {s2.name} | {m2.name} | **{m4.name}** |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Bounce localization error — median | {s1.bounce_error_median_mm:.2f} mm "
            f"| {s2.bounce_error_median_mm:.2f} mm | {m2.bounce_error_median_mm:.2f} mm "
            f"| **{m4.bounce_error_median_mm:.2f} mm** |"
        ),
        (
            f"| Bounce localization error — p90 | {s1.bounce_error_p90_mm:.2f} mm "
            f"| {s2.bounce_error_p90_mm:.2f} mm | {m2.bounce_error_p90_mm:.2f} mm "
            f"| **{m4.bounce_error_p90_mm:.2f} mm** |"
        ),
        (
            f"| Bounce localization error — mean | {s1.bounce_error_mean_mm:.2f} mm "
            f"| {s2.bounce_error_mean_mm:.2f} mm | {m2.bounce_error_mean_mm:.2f} mm "
            f"| **{m4.bounce_error_mean_mm:.2f} mm** |"
        ),
        (
            f"| Overall line-call accuracy (`N = {report.total_cases}`) | {s1.line_call_accuracy_pct:.2f}% "
            f"| {s2.line_call_accuracy_pct:.2f}% | {m2.line_call_accuracy_pct:.2f}% "
            f"| **{m4.line_call_accuracy_pct:.2f}%** |"
        ),
        (
            f"| Tight-call accuracy within `+/-35 mm` (`N = {report.tight_cases_count}`) "
            f"| {s1.tight_call_accuracy_pct:.2f}% | {s2.tight_call_accuracy_pct:.2f}% "
            f"| {m2.tight_call_accuracy_pct:.2f}% | **{m4.tight_call_accuracy_pct:.2f}%** |"
        ),
        (
            f"| Uncertainty calibration — Brier score (lower is better) | — | — "
            f"| {m2.brier_score:.4f} | **{m4.brier_score:.4f}** |"
        ),
        (
            f"| Lateral 95% CI coverage (`|err_y| <= 1.96 * sigma_y`) | — | — "
            f"| {m2.ci95_coverage_pct:.2f}% | **{m4.ci95_coverage_pct:.2f}%** |"
        ),
        "",
        "## Controlled Skid-Footprint Ablation (2-Cam DLT 3D Fixed)",
        "",
        "| Decision Rule (Same 2-Cam DLT 3D Coordinates) | Overall Accuracy | Tight Accuracy (`+/-35 mm`) |",
        "|---|---:|---:|",
        f"| Center-point only (`use_skid_footprint=False`) | {report.skid_ablation_2cam_no_skid_acc_pct:.2f}% "
        f"| {report.skid_ablation_2cam_no_skid_tight_pct:.2f}% |",
        f"| **Oblique skid ellipse (`use_skid_footprint=True`)** | **{m2.line_call_accuracy_pct:.2f}%** "
        f"| **{m2.tight_call_accuracy_pct:.2f}%** |",
        "",
        "## Key Technical Findings",
        "",
        f"1. **Monocular `Z=0` Parallax Bottleneck ({s2.bounce_error_median_mm:.2f} mm median error):** "
        "Even with exact per-frame court homography `H_t`, 2D bounce-cusp detection suffers ~50-90 ms "
        "timing jitter because out-of-plane elevation `Z(t)` foreshortens along the viewing ray.",
        f"2. **Multi-View Geometry + Ballistic Fitting ({m2.bounce_error_median_mm:.2f} mm in 2-Cam, "
        f"{m4.bounce_error_median_mm:.2f} mm in 4-Cam):** Triangulating calibrated pixel streams via DLT, "
        "stripping spikes with multi-plane KNN, and intersecting pre/post-bounce ballistic curves at `Z = r_ball` "
        f"cuts mean error by **{report.mean_error_reduction_2cam_vs_1cam_mm:.1f} mm** over dynamic 1-cam tracking.",
        "3. **Honest Limitations:** Real uncalibrated single-camera broadcast videos (`hawkeye-lite predict`) "
        "still operate strictly on the `Z=0` court plane unless a second calibrated view is available.",
        "",
    ]
    return "\n".join(lines)


def build_web_studio_payload(report: BenchmarkReport) -> dict[str, object]:
    """Build deterministic JSON payload from real 2-cam DLT triangulation for the Web Studio."""
    ref = CourtReference()
    cam0 = build_lookat_camera(eye_world=(31.0, 5.485, 9.0), target_world=(11.5, 5.485, 0.0))
    cam_side = build_lookat_camera(eye_world=(-5.5, 11.5, 8.5), target_world=(10.5, 5.5, 0.0))
    replays: list[dict[str, object]] = []

    for preset_key, spec in SCENARIO_PRESETS.items():
        traj_gt, bounce_step, _, _, _ = simulate_aerodynamic_rally_exact(
            pos0_m=spec.pos0_m,
            vel0_m_s=spec.vel0_m_s,
            spin_rad_s=spec.spin_rad_s,
            dt=1.0 / spec.fps,
            num_steps=48,
        )
        b_idx_gt = bounce_step if bounce_step is not None else int(np.argmin(traj_gt[:, 2]))

        rng_p = seeded_rng("web-studio-2cam", preset_key)
        uv0 = cam0.project_3d_to_2d(traj_gt) + rng_p.normal(0.0, 1.0, size=(len(traj_gt), 2))
        uv1 = cam_side.project_3d_to_2d(traj_gt) + rng_p.normal(0.0, 1.0, size=(len(traj_gt), 2))
        for outlier_idx in (6, 15, 29, 38):
            if outlier_idx < len(uv0) and abs(outlier_idx - b_idx_gt) > 2:
                uv0[outlier_idx] += np.array([75.0, -65.0], dtype=np.float64)

        mv = reconstruct_multiview_bounce(
            [cam0.p_mat, cam_side.p_mat],
            [uv0, uv1],
            fps=spec.fps,
            pixel_sigma_px=1.0,
        )
        fitted_3d, _, _ = reconstruct_3d_trajectory_seu(mv.knn_points_3d, samples_per_segment=25)
        bx_m, by_m = mv.bounce_xy_m
        decision = evaluate_line_call(
            bounce_x_m=bx_m,
            bounce_y_m=by_m,
            vel_incident_m_s=mv.vel_incident_m_s,
            singles=True,
            use_skid_footprint=True,
            explicit_sigma_xy_mm=(mv.sigma_x_mm, mv.sigma_y_mm),
        )

        replays.append(
            {
                "id": preset_key,
                "name": spec.name,
                "description": spec.description,
                "fps": spec.fps,
                "cutaway_frames": spec.include_cutaway_frames,
                "has_angle_switch": spec.include_angle_switch,
                "bounce_index": mv.bounce_frame_est,
                "bounce_x_m": round(bx_m, 3),
                "bounce_y_m": round(by_m, 3),
                "verdict": decision.verdict,
                "is_in": decision.is_in,
                "margin_mm": decision.margin_mm,
                "sigma_mm": decision.sigma_mm,
                "nearest_line": decision.nearest_line_name,
                "raw_points_3d": [[round(float(v), 3) for v in pt] for pt in mv.raw_points_3d],
                "knn_points_3d": [
                    [round(float(v), 3) for v in pt] for pt in mv.knn_points_3d if float(np.linalg.norm(pt)) > 1e-6
                ],
                "fitted_curve_3d": [[round(float(v), 3) for v in pt] for pt in fitted_3d],
                "ground_truth_3d": [[round(float(v), 3) for v in pt] for pt in traj_gt],
            }
        )

    return {
        "version": "0.1.0",
        "court": {
            "length_m": ITF_COURT_LENGTH_M,
            "doubles_width_m": ITF_DOUBLES_WIDTH_M,
            "world_coordinates_3d": [[round(float(v), 3) for v in pt] for pt in WORLD_COORDINATES_3D],
            "wireframe_groups": [list(g) for g in SEU_WIREFRAME_GROUPS],
            "configurations_12": {str(k): [list(pt) for pt in v] for k, v in ref.court_conf.items()},
        },
        "benchmark": {
            "total_cases": report.total_cases,
            "tight_cases_count": report.tight_cases_count,
            "tight_in_rate_pct": report.tight_in_rate_pct,
            "static_1cam": asdict(report.static_1cam),
            "dynamic_1cam": asdict(report.dynamic_1cam),
            "dlt_2cam": asdict(report.dlt_2cam),
            "dlt_4cam": asdict(report.dlt_4cam),
            "skid_ablation_2cam_no_skid_acc_pct": report.skid_ablation_2cam_no_skid_acc_pct,
            "skid_ablation_2cam_no_skid_tight_pct": report.skid_ablation_2cam_no_skid_tight_pct,
            "mean_error_reduction_2cam_vs_1cam_mm": report.mean_error_reduction_2cam_vs_1cam_mm,
            "mean_error_reduction_4cam_vs_2cam_mm": report.mean_error_reduction_4cam_vs_2cam_mm,
            "sample_cases": [asdict(c) for c in report.cases[:60]],
        },
        "replays": replays,
    }


def export_all(repo_root: str | Path, check_only: bool = False) -> bool:
    """Export `docs/benchmark.md` and `web/public/data/hawkeye_data.json` (or check for drift)."""
    root = Path(repo_root)
    report = run_benchmark(num_seeds=100)
    md_content = render_benchmark_markdown(report)
    payload = build_web_studio_payload(report)
    json_content = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    md_path = root / "docs" / "benchmark.md"
    json_path = root / "web" / "public" / "data" / "hawkeye_data.json"

    if check_only:
        if not md_path.exists() or not json_path.exists():
            return False
        current_md = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        current_json = json_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        return current_md == md_content.replace("\r\n", "\n") and current_json == json_content.replace("\r\n", "\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8", newline="\n")
    json_path.write_text(json_content, encoding="utf-8", newline="\n")
    return True
