"""Synthetic Multi-Angle Broadcast Tennis Rally Video Generator (.mp4).

Generates realistic 1280x720 Australian Open ("Melbourne Blue") and Dubai ("Blue/Green")
tennis rally `.mp4` videos with known 3D ground-truth trajectories, player movement,
camera pan, optional cutaway shots, and angle switches so ``hawkeye-lite predict`` can be
tested and demonstrated end-to-end without requiring external 80 MB video downloads.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from camera.calibration import PinholeCamera, build_lookat_camera
from geometry.court_reference import (
    ITF_COURT_LENGTH_M,
    ITF_DOUBLES_WIDTH_M,
    SEU_WIREFRAME_GROUPS,
    WORLD_COORDINATES_3D,
)
from numpy.typing import NDArray
from tracking.reconstruction import simulate_aerodynamic_flight_rk4


@dataclass(frozen=True)
class RallyScenarioSpec:
    """Specification for a synthetic multi-shot or single-shot tennis rally video."""

    name: str
    description: str
    pos0_m: tuple[float, float, float]
    vel0_m_s: tuple[float, float, float]
    spin_rad_s: tuple[float, float, float]
    num_frames: int = 60
    fps: float = 30.0
    include_cutaway_frames: int = 0
    include_angle_switch: bool = False
    pan_amplitude_m: float = 0.0


SCENARIO_PRESETS: dict[str, RallyScenarioSpec] = {
    "ao_sideline_in": RallyScenarioSpec(
        name="ao_sideline_in",
        description="Australian Open crosscourt forehand skidding ON the right singles sideline (IN by 18 mm)",
        pos0_m=(22.4, 3.2, 1.15),
        vel0_m_s=(-25.2, 8.12, 2.35),
        spin_rad_s=(0.0, -265.0, 0.0),
        num_frames=54,
        fps=30.0,
        include_cutaway_frames=0,
        include_angle_switch=False,
        pan_amplitude_m=0.18,
    ),
    "ao_baseline_out": RallyScenarioSpec(
        name="ao_baseline_out",
        description="Deep flat down-the-line drive landing just long past the far baseline (OUT by 34 mm)",
        pos0_m=(22.8, 7.6, 1.22),
        vel0_m_s=(-28.6, 0.4, 2.85),
        spin_rad_s=(0.0, -120.0, 0.0),
        num_frames=54,
        fps=30.0,
        include_cutaway_frames=0,
        include_angle_switch=False,
        pan_amplitude_m=0.12,
    ),
    "broadcast_multi_angle": RallyScenarioSpec(
        name="broadcast_multi_angle",
        description="Multi-shot broadcast clip: Cutaway -> Main High Rally Cam (with pan) -> Low Baseline Angle Cut",
        pos0_m=(22.2, 3.8, 1.18),
        vel0_m_s=(-24.8, 6.4, 2.30),
        spin_rad_s=(0.0, -250.0, 0.0),
        num_frames=60,
        fps=30.0,
        include_cutaway_frames=8,
        include_angle_switch=True,
        pan_amplitude_m=0.25,
    ),
}


def render_court_3d_frame(
    cam: PinholeCamera,
    ball_pos_3d: NDArray[np.float64] | None,
    p1_xy_m: tuple[float, float],
    p2_xy_m: tuple[float, float],
) -> NDArray[np.uint8]:
    """Render a photorealistic 1280x720 broadcast court frame from 3D camera ``cam``."""
    w, h = cam.width, cam.height
    frame = np.full((h, w, 3), (44, 58, 34), dtype=np.uint8)

    # Outer surround polygon
    surround_3d = np.array(
        [
            [-4.5, -3.2, 0.0],
            [-4.5, ITF_DOUBLES_WIDTH_M + 3.2, 0.0],
            [ITF_COURT_LENGTH_M + 4.5, ITF_DOUBLES_WIDTH_M + 3.2, 0.0],
            [ITF_COURT_LENGTH_M + 4.5, -3.2, 0.0],
        ],
        dtype=np.float64,
    )
    surround_2d = np.round(cam.project_3d_to_2d(surround_3d)).astype(np.int32)
    cv2.fillConvexPoly(frame, surround_2d, (165, 118, 42))

    # Inner doubles court polygon (Melbourne Blue)
    court_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, ITF_DOUBLES_WIDTH_M, 0.0],
            [ITF_COURT_LENGTH_M, ITF_DOUBLES_WIDTH_M, 0.0],
            [ITF_COURT_LENGTH_M, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    court_2d = np.round(cam.project_3d_to_2d(court_3d)).astype(np.int32)
    cv2.fillConvexPoly(frame, court_2d, (152, 82, 24))

    # Draw crisp white court lines (thickness 4 px so CourtDetector._threshold & _filter_pixels lock cleanly)
    pts_2d = cam.project_3d_to_2d(WORLD_COORDINATES_3D)
    for group in SEU_WIREFRAME_GROUPS[:4]:
        for a, b in itertools.pairwise(group):
            p1 = (round(float(pts_2d[a, 0])), round(float(pts_2d[a, 1])))
            p2 = (round(float(pts_2d[b, 0])), round(float(pts_2d[b, 1])))
            cv2.line(frame, p1, p2, (250, 250, 250), 4, cv2.LINE_AA)

    # Draw far player (Player 2) and near player (Player 1) silhouettes
    for (px_m, py_m), shirt_bgr in ((p2_xy_m, (45, 55, 210)), (p1_xy_m, (210, 180, 50))):
        feet_2d = cam.project_3d_to_2d(np.array([px_m, py_m, 0.0], dtype=np.float64))
        head_2d = cam.project_3d_to_2d(np.array([px_m, py_m, 1.78], dtype=np.float64))
        fx, fy = round(float(feet_2d[0])), round(float(feet_2d[1]))
        _hx, hy = round(float(head_2d[0])), round(float(head_2d[1]))
        ph = max(24, abs(fy - hy))
        pw = max(12, int(ph * 0.42))
        cv2.rectangle(frame, (fx - pw // 2, hy), (fx + pw // 2, fy), shirt_bgr, -1)
        cv2.circle(frame, (fx, hy + 6), max(5, pw // 4), (160, 190, 230), -1)

    # Draw ball ground shadow (Z=0) and optic-yellow tennis ball (Z>0)
    if ball_pos_3d is not None:
        shadow_2d = cam.project_3d_to_2d(np.array([ball_pos_3d[0], ball_pos_3d[1], 0.0], dtype=np.float64))
        ball_2d = cam.project_3d_to_2d(ball_pos_3d)
        sx, sy = round(float(shadow_2d[0])), round(float(shadow_2d[1]))
        bx, by = round(float(ball_2d[0])), round(float(ball_2d[1]))
        cv2.ellipse(frame, (sx, sy), (6, 3), 0, 0, 360, (55, 35, 12), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 6, (35, 245, 225), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx - 1, by - 1), 3, (120, 255, 250), -1, cv2.LINE_AA)

    return frame


def render_cutaway_frame(width: int = 1280, height: int = 720, frame_idx: int = 0) -> NDArray[np.uint8]:
    """Render a non-court broadcast cutaway frame (stadium scoreboard / crowd graphic wipe)."""
    frame = np.full((height, width, 3), (28, 24, 32), dtype=np.uint8)
    cv2.rectangle(frame, (180, 140), (width - 180, height - 140), (48, 42, 58), -1)
    cv2.rectangle(frame, (180, 140), (width - 180, height - 140), (90, 180, 240), 3)
    cv2.putText(
        frame,
        "AUSTRALIAN OPEN - MELBOURNE PARK",
        (260, 310),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"BROADCAST CUTAWAY / REPLAY WIPE (FRAME {frame_idx})",
        (295, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (120, 210, 250),
        2,
        cv2.LINE_AA,
    )
    return frame


def generate_synthetic_rally_video(
    output_path: str | Path,
    scenario: str | RallyScenarioSpec = "ao_sideline_in",
    width: int = 1280,
    height: int = 720,
) -> dict[str, object]:
    """Generate a synthetic broadcast `.mp4` video and return its 3D ground-truth metadata."""
    spec = SCENARIO_PRESETS[scenario] if isinstance(scenario, str) else scenario
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    active_frames = spec.num_frames - spec.include_cutaway_frames
    traj_3d, bounce_step = simulate_aerodynamic_flight_rk4(
        pos0_m=spec.pos0_m,
        vel0_m_s=spec.vel0_m_s,
        spin_rad_s=spec.spin_rad_s,
        dt=1.0 / spec.fps,
        num_steps=max(active_frames, 10),
    )

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, spec.fps, (width, height))

    # Write initial cutaway frames if configured
    for c_idx in range(spec.include_cutaway_frames):
        writer.write(render_cutaway_frame(width=width, height=height, frame_idx=c_idx))

    # Write active court frames (with optional camera pan and mid-rally angle switch)
    for k in range(active_frames):
        pan_y = spec.pan_amplitude_m * float(np.sin(2.0 * np.pi * k / max(active_frames, 1)))
        if spec.include_angle_switch and k >= (active_frames * 2) // 3:
            # Angle 2: Higher / closer baseline review angle
            cam = build_lookat_camera(
                eye_world=(29.2, 5.485 + pan_y * 0.5, 7.6),
                target_world=(8.5, 5.485 + pan_y * 0.5, 0.0),
                width=width,
                height=height,
                fov_deg=38.0,
            )
        else:
            # Angle 1: Primary broadcast high rally camera
            cam = build_lookat_camera(
                eye_world=(31.2, 5.485 + pan_y, 9.2),
                target_world=(11.2, 5.485 + pan_y, 0.0),
                width=width,
                height=height,
                fov_deg=42.0,
            )

        p1_xy = (22.6, 4.0 + 0.035 * k)
        p2_xy = (1.1, 6.2 + 0.04 * k)
        ball_3d = traj_3d[k]
        frame = render_court_3d_frame(cam, ball_3d, p1_xy_m=p1_xy, p2_xy_m=p2_xy)
        writer.write(frame)

    writer.release()

    b_idx = bounce_step if bounce_step is not None else int(np.argmin(traj_3d[:, 2]))
    bounce_gt = traj_3d[min(b_idx, len(traj_3d) - 1)]
    return {
        "scenario": spec.name,
        "description": spec.description,
        "output_path": str(out_file),
        "num_frames": spec.num_frames,
        "cutaway_frames": spec.include_cutaway_frames,
        "bounce_frame_gt": b_idx + spec.include_cutaway_frames,
        "bounce_x_m_gt": round(float(bounce_gt[0]), 4),
        "bounce_y_m_gt": round(float(bounce_gt[1]), 4),
        "bounce_z_m_gt": round(float(bounce_gt[2]), 4),
        "traj_3d_gt": traj_3d.tolist(),
    }
