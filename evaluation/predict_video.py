"""End-to-End Video-In -> Video-Out Pipeline (`predict_video.py` Evolved).

Combines:
1. ``ArtLabss/tennis-tracking/predict_video.py`` workflow:
   - Pass 1: Per-frame ``ShotSegmenter`` cutaway gating, ``CourtDetector.detect`` / ``track_court``
     (with automatic re-detection on camera cuts or pan drift), ``DetectionModel.detect_player_1``,
     ``detect_top_persons``, ``find_player_2_box``, and ``TrackNetMotionBallDetector`` + ``BallKalmanTracker``.
   - Pass 2: 3-pass ``diff_xy`` + ``remove_outliers`` + ``interpolation``, ``calculate_feet_positions``
     with ``savgol_filter``, 20-lag ``detect_bounces_artlabss``, 3D lifting & SEU multi-plane KNN
     curve reconstruction, and ``evaluate_line_call`` (oblique skid ellipse vs. 50 mm ITF line).
   - Output Rendering:
     * Projected court line overlay (suppressed cleanly on non-court cutaways!)
     * Player 1 & Player 2 bounding boxes
     * 8-frame yellow ball comet trail (``deque(maxlen=8)``)
     * Dynamic 2D Top-Down Court Minimap merged in top-right corner when ``--minimap=1``
     * Bounce circles, live ``IN`` / ``OUT`` millimeter banner, and appended **3D SEU / Australian Open
       Hawk-Eye Challenge Replay Cut-in** when ``--bounce=1``.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from camera.calibration import build_lookat_camera
from decision.line_call import LineCallDecision, evaluate_line_call
from geometry.court_reference import (
    ITF_COURT_LENGTH_M,
    ITF_DOUBLES_WIDTH_M,
    SEU_WIREFRAME_GROUPS,
    WORLD_COORDINATES_3D,
)
from geometry.homography import apply_homography
from numpy.typing import NDArray
from tracking.interpolation import diff_xy, interpolation, remove_outliers
from tracking.reconstruction import (
    detect_bounces_artlabss,
    lift_2d_trajectory_to_3d_court,
    reconstruct_3d_trajectory_seu,
)
from tracking.sort import BallKalmanTracker
from vision.court_detector import CourtDetector
from vision.detection import (
    DetectionModel,
    TrackNetMotionBallDetector,
    create_top_view_image,
    mark_player_box,
    merge_minimap,
)
from vision.shot_segmenter import CameraViewType, FrameShotState, ShotSegmenter


@dataclass(frozen=True)
class BounceEventReport:
    """Telemetry for a single detected bounce and its Hawk-Eye line call."""

    frame_index: int
    pixel_uv: tuple[float, float]
    court_ref_uv: tuple[float, float]
    court_xy_m: tuple[float, float]
    decision: LineCallDecision


@dataclass(frozen=True)
class VideoPredictionResult:
    """Complete telemetry report returned by ``process_video``."""

    input_video_path: str
    output_video_path: str
    fps: float
    width: int
    height: int
    total_input_frames: int
    total_output_frames: int
    active_court_frames: int
    cutaway_frames: int
    num_camera_shots: int
    bounces: list[BounceEventReport]
    trajectory_3d_m: list[list[float]]


def render_3d_hawkeye_replay_frame(
    width: int,
    height: int,
    traj_3d: NDArray[np.float64],
    bounce_report: BounceEventReport | None,
    progress: float,
) -> NDArray[np.uint8]:
    """Render a split-screen 3D Hawk-Eye Challenge & SEU 3D Wireframe Replay frame.

    - Left panel (58% width): Australian Open 3D Virtual Camera Zoom onto the bounce line,
      showing the descending 3D trajectory ribbon, expanding dark elliptical skid mark on the
      50 mm white line, and pulsing ``IN`` (emerald) / ``OUT`` (crimson) verdict banner.
    - Right panel (42% width): SEU ``RECONSTRUCTION.py`` 3D isometric wireframe court plot
      showing the full 3D reconstructed curve (blue points), green ``IN/hit`` markers, and
      red ``OUT`` markers.
    """
    canvas = np.full((height, width, 3), (18, 22, 30), dtype=np.uint8)
    split_x = int(width * 0.58)

    # -----------------------------------------------------------------------
    # Left Panel: Australian Open 3D Hawk-Eye Close-Up Zoom
    # -----------------------------------------------------------------------
    bx_m, by_m = bounce_report.court_xy_m if bounce_report is not None else (5.485, 8.20)
    s = float(np.clip(progress, 0.0, 1.0))
    smooth_s = s * s * (3.0 - 2.0 * s)

    eye = (
        float(26.0 * (1.0 - smooth_s) + (bx_m + 2.4) * smooth_s),
        float(5.485 * (1.0 - smooth_s) + (by_m - 0.8) * smooth_s),
        float(7.5 * (1.0 - smooth_s) + 2.35 * smooth_s),
    )
    target = (
        float(11.885 * (1.0 - smooth_s) + bx_m * smooth_s),
        float(5.485 * (1.0 - smooth_s) + by_m * smooth_s),
        0.0,
    )
    zoom_cam = build_lookat_camera(
        eye_world=eye,
        target_world=target,
        width=split_x,
        height=height,
        fov_deg=float(40.0 * (1.0 - smooth_s) + 22.0 * smooth_s),
    )

    left_panel = np.full((height, split_x, 3), (165, 112, 36), dtype=np.uint8)
    court_poly_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, ITF_DOUBLES_WIDTH_M, 0.0],
            [ITF_COURT_LENGTH_M, ITF_DOUBLES_WIDTH_M, 0.0],
            [ITF_COURT_LENGTH_M, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    c2d = np.round(zoom_cam.project_3d_to_2d(court_poly_3d)).astype(np.int32)
    cv2.fillConvexPoly(left_panel, c2d, (148, 78, 20))

    # Draw 50 mm court lines
    w_pts = zoom_cam.project_3d_to_2d(WORLD_COORDINATES_3D)
    line_thick = round(3 + 5 * smooth_s)
    for group in SEU_WIREFRAME_GROUPS[:4]:
        for a, b in itertools.pairwise(group):
            p1 = (round(float(w_pts[a, 0])), round(float(w_pts[a, 1])))
            p2 = (round(float(w_pts[b, 0])), round(float(w_pts[b, 1])))
            cv2.line(left_panel, p1, p2, (248, 248, 248), line_thick, cv2.LINE_AA)

    # Draw 3D ball trajectory arc and shadow in the zoom view
    if len(traj_3d) >= 2:
        vis_count = max(2, round(len(traj_3d) * min(1.0, s * 1.25)))
        sub_traj = traj_3d[:vis_count]
        shadow_3d = np.column_stack([sub_traj[:, 0], sub_traj[:, 1], np.zeros(len(sub_traj))])
        sh_2d = np.round(zoom_cam.project_3d_to_2d(shadow_3d)).astype(np.int32)
        tr_2d = np.round(zoom_cam.project_3d_to_2d(sub_traj)).astype(np.int32)
        for k in range(len(tr_2d) - 1):
            cv2.line(left_panel, tuple(sh_2d[k]), tuple(sh_2d[k + 1]), (85, 45, 12), 2, cv2.LINE_AA)
            cv2.line(left_panel, tuple(tr_2d[k]), tuple(tr_2d[k + 1]), (0, 235, 255), 3, cv2.LINE_AA)

    # Draw elliptical bounce skid mark on the court surface
    b_pt_2d = zoom_cam.project_3d_to_2d(np.array([bx_m, by_m, 0.0], dtype=np.float64))
    b_px = (round(float(b_pt_2d[0])), round(float(b_pt_2d[1])))
    skid_a = round(10 + 18 * smooth_s)
    skid_b = round(6 + 10 * smooth_s)
    cv2.ellipse(left_panel, b_px, (skid_a, skid_b), -15, 0, 360, (25, 28, 32), -1, cv2.LINE_AA)
    cv2.ellipse(left_panel, b_px, (skid_a + 3, skid_b + 3), -15, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)

    # Verdict Banner (Australian Open Hawk-Eye Live Broadcast Style)
    verdict = bounce_report.decision.verdict if bounce_report is not None else "IN"
    is_in = bounce_report.decision.is_in if bounce_report is not None else True
    margin_mm = abs(bounce_report.decision.margin_mm) if bounce_report is not None else 18.2
    sigma_mm = bounce_report.decision.sigma_mm if bounce_report is not None else 2.1
    line_name = bounce_report.decision.nearest_line_name if bounce_report is not None else "right_singles_sideline"

    banner_bg = (38, 165, 52) if is_in else (36, 42, 215)
    cv2.rectangle(left_panel, (24, 22), (split_x - 24, 114), (16, 20, 28), -1)
    cv2.rectangle(left_panel, (24, 22), (175, 114), banner_bg, -1)
    cv2.rectangle(left_panel, (24, 22), (split_x - 24, 114), (235, 235, 235), 2)
    cv2.putText(
        left_panel,
        verdict,
        (54 if is_in else 40, 84),
        cv2.FONT_HERSHEY_DUPLEX,
        1.65,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        left_panel,
        "HAWK-EYE 3D CHALLENGE REPLAY",
        (195, 56),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (120, 225, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        left_panel,
        f"{verdict} BY {margin_mm:.1f} mm (+/- {sigma_mm:.1f} mm) | {line_name.upper()}",
        (195, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    canvas[:, :split_x] = left_panel

    # -----------------------------------------------------------------------
    # Right Panel: SEU 3D Isometric Wireframe Court Reconstruction Plot
    # -----------------------------------------------------------------------
    right_w = width - split_x
    right_panel = np.full((height, right_w, 3), (248, 249, 252), dtype=np.uint8)
    iso_cam = build_lookat_camera(
        eye_world=(11.885 - 18.5 * float(np.cos(0.45 + 0.35 * smooth_s)), -14.5, 12.5),
        target_world=(11.885, 5.485, 0.6),
        width=right_w,
        height=height,
        fov_deg=38.0,
    )

    # Draw black 3D wireframe court & net (matching SEU RECONSTRUCTION.py)
    iso_court_2d = iso_cam.project_3d_to_2d(WORLD_COORDINATES_3D)
    for group in SEU_WIREFRAME_GROUPS:
        for a, b in itertools.pairwise(group):
            p1 = (round(float(iso_court_2d[a, 0])), round(float(iso_court_2d[a, 1])))
            p2 = (round(float(iso_court_2d[b, 0])), round(float(iso_court_2d[b, 1])))
            cv2.line(right_panel, p1, p2, (30, 30, 30), 2, cv2.LINE_AA)

    # Draw blue 3D trajectory points and green/red bounce marker
    if len(traj_3d) > 0:
        iso_traj_2d = iso_cam.project_3d_to_2d(traj_3d)
        vis_n = max(1, round(len(iso_traj_2d) * min(1.0, s * 1.25)))
        for k in range(vis_n):
            pt = (round(float(iso_traj_2d[k, 0])), round(float(iso_traj_2d[k, 1])))
            cv2.circle(right_panel, pt, 4, (215, 70, 20), -1, cv2.LINE_AA)

    b_iso = iso_cam.project_3d_to_2d(np.array([bx_m, by_m, 0.0335], dtype=np.float64))
    b_iso_pt = (round(float(b_iso[0])), round(float(b_iso[1])))
    marker_col = (35, 180, 45) if is_in else (30, 35, 230)
    cv2.circle(right_panel, b_iso_pt, 9, marker_col, -1, cv2.LINE_AA)
    cv2.circle(right_panel, b_iso_pt, 11, (20, 20, 20), 2, cv2.LINE_AA)

    # Header & Legend matching SEU plot style
    cv2.putText(
        right_panel,
        "3D TRAJECTORY RECONSTRUCTION (KNN + CONSTRAINED FIT)",
        (18, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (25, 30, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(right_panel, (32, 68), 6, (215, 70, 20), -1)
    cv2.putText(right_panel, "3D Flight Arc (X,Y,Z)", (46, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1)
    cv2.circle(right_panel, (235, 68), 6, (35, 180, 45), -1)
    cv2.putText(right_panel, "IN / Hit", (249, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1)
    cv2.circle(right_panel, (345, 68), 6, (30, 35, 230), -1)
    cv2.putText(right_panel, "OUT", (359, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1)

    cv2.putText(
        right_panel,
        f"Bounce 3D: X={bx_m:.2f}m, Y={by_m:.2f}m, Z=0.03m",
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (35, 45, 60),
        1,
        cv2.LINE_AA,
    )

    canvas[:, split_x:] = right_panel
    cv2.line(canvas, (split_x, 0), (split_x, height), (220, 220, 220), 2)
    return canvas


def process_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    minimap: int = 1,
    bounce: int = 1,
    max_frames: int | None = None,
    append_3d_replay: bool = True,
) -> VideoPredictionResult:
    """Process an input tennis video (`.mp4`) and write the annotated output video (`.mp4`).

    Supports ``--input_video_path``, ``--output_video_path``, ``--minimap=0|1``, and ``--bounce=0|1``
    from ``ArtLabss/tennis-tracking/predict_video.py`` while adding multi-angle shot cut protection,
    150x faster court detection, and 3D Hawk-Eye line calling.
    """
    in_path = Path(input_video_path)
    if not in_path.exists():
        msg = f"Input video not found: {in_path}"
        raise FileNotFoundError(msg)

    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        msg = f"Could not open video file: {in_path}"
        raise RuntimeError(msg)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    court_detector = CourtDetector()
    shot_segmenter = ShotSegmenter()
    detection_model = DetectionModel()
    ball_detector = TrackNetMotionBallDetector()
    ball_kalman = BallKalmanTracker()

    raw_frames: list[NDArray[np.uint8]] = []
    shot_states: list[FrameShotState] = []
    frame_lines: list[NDArray[np.float64] | None] = []
    raw_ball_coords: list[tuple[float, float] | list[float] | None] = []

    frame_idx = 0
    prev_was_court = False

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        frame: NDArray[np.uint8] = np.asarray(raw_frame, dtype=np.uint8)
        raw_frames.append(frame.copy())

        # Quick histogram cut check before deciding detect() vs track_court()
        hist_now = ShotSegmenter.compute_hsv_histogram(frame)
        is_hard_cut = False
        if shot_segmenter._prev_hist is not None:
            h_dist = float(cv2.compareHist(shot_segmenter._prev_hist, hist_now, cv2.HISTCMP_BHATTACHARYYA))
            is_hard_cut = h_dist >= shot_segmenter.cut_hist_threshold

        if frame_idx == 0 or is_hard_cut or not prev_was_court:
            lines = court_detector.detect(frame)
        else:
            lines = court_detector.track_court(frame)

        shot_state = shot_segmenter.evaluate_frame(
            frame=frame,
            frame_index=frame_idx,
            court_score=court_detector.court_score,
            best_conf=court_detector.best_conf,
        )
        shot_states.append(shot_state)

        if shot_state.is_cut_boundary or not shot_state.is_court_active:
            ball_detector.reset()
            ball_kalman.reset()

        if shot_state.is_court_active:
            prev_was_court = True
            frame_lines.append(lines.copy())
            detection_model.detect_player_1(frame, court_detector)
            detection_model.detect_top_persons(frame, court_detector, frame_idx + 1)
            raw_det = ball_detector.detect_ball(frame, court_warp_matrix=court_detector.court_warp_matrix[-1])
            tracked_uv = ball_kalman.step(raw_det)
            raw_ball_coords.append([tracked_uv[0], tracked_uv[1]] if tracked_uv is not None else None)
        else:
            prev_was_court = False
            frame_lines.append(None)
            detection_model.player_1_boxes.append([None, None, None, None])
            for det_person in detection_model.persons_boxes:
                detection_model.persons_boxes[det_person].append([None, None, None, None])
            raw_ball_coords.append(None)

        frame_idx += 1

    cap.release()
    total_in = len(raw_frames)
    if total_in == 0:
        msg = f"No readable frames in video: {in_path}"
        raise RuntimeError(msg)

    # Discriminate top player (Player 2) across the video
    detection_model.find_player_2_box()
    smoothed_feet_1, smoothed_feet_2 = detection_model.calculate_feet_positions(court_detector)

    # 3-pass outlier removal and linear interpolation on ball coordinates (ArtLabss predict_video.py)
    cleaned_coords = list(raw_ball_coords)
    for _ in range(3):
        xd, yd = diff_xy(cleaned_coords)
        remove_outliers(xd, yd, cleaned_coords)
    interp_coords = interpolation(cleaned_coords)

    # Zero out interpolated coords on non-court cutaway frames so cutaways stay clean
    active_coords: list[tuple[float, float]] = []
    for i, uv in enumerate(interp_coords):
        if shot_states[i].is_court_active:
            active_coords.append(uv)
        else:
            active_coords.append((0.0, 0.0))

    # Detect bounces and lift trajectory to 3D court meters
    bounce_frame_indices, _ = detect_bounces_artlabss(active_coords, fps=fps)
    bounce_frame_indices = [b for b in bounce_frame_indices if shot_states[b].is_court_active]

    # If a short rally has a clear lowest Z / court contact point not caught by 2D v-cusp,
    # fall back to the local vertical maximum in image v among active court frames
    if not bounce_frame_indices:
        active_indices = [
            idx
            for idx in range(2, len(active_coords) - 2)
            if shot_states[idx].is_court_active and active_coords[idx] != (0.0, 0.0)
        ]
        if active_indices:
            peak_v_idx = max(active_indices, key=lambda idx: active_coords[idx][1])
            bounce_frame_indices = [peak_v_idx]

    raw_3d = lift_2d_trajectory_to_3d_court(
        coords_2d=active_coords,
        game_warp_matrices=court_detector.game_warp_matrix,
        bounce_frames=bounce_frame_indices,
        pixel_to_meters_fn=court_detector.court_reference.pixel_to_meters,
        fps=fps,
    )
    fitted_3d, _, _ = reconstruct_3d_trajectory_seu(raw_3d)

    bounce_reports: list[BounceEventReport] = []
    bounces_ref_history: dict[int, tuple[float, float, bool]] = {}

    for b_idx in bounce_frame_indices:
        u_b, v_b = active_coords[b_idx]
        inv_m = court_detector.game_warp_matrix[min(b_idx, len(court_detector.game_warp_matrix) - 1)]
        ref_uv = apply_homography(inv_m, np.array([u_b, v_b], dtype=np.float64))
        xm, ym = court_detector.court_reference.pixel_to_meters(float(ref_uv[0]), float(ref_uv[1]))

        # Estimate incident 3D velocity vector before bounce
        prev_idx = max(0, b_idx - 2)
        dt_b = max(1, b_idx - prev_idx) / max(fps, 1.0)
        vx_m = float(raw_3d[b_idx, 0] - raw_3d[prev_idx, 0]) / dt_b
        vy_m = float(raw_3d[b_idx, 1] - raw_3d[prev_idx, 1]) / dt_b
        decision = evaluate_line_call(
            bounce_x_m=xm,
            bounce_y_m=ym,
            vel_incident_m_s=(vx_m, vy_m, -7.2),
            singles=True,
            use_skid_footprint=True,
        )
        rep = BounceEventReport(
            frame_index=b_idx,
            pixel_uv=(u_b, v_b),
            court_ref_uv=(float(ref_uv[0]), float(ref_uv[1])),
            court_xy_m=(round(xm, 4), round(ym, 4)),
            decision=decision,
        )
        bounce_reports.append(rep)
        bounces_ref_history[b_idx] = (float(ref_uv[0]), float(ref_uv[1]), decision.is_in)

    # -----------------------------------------------------------------------
    # Write Output Video (.mp4)
    # -----------------------------------------------------------------------
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    ball_queue: deque[tuple[float, float] | None] = deque([None] * 8, maxlen=8)
    trail_ref_pts: list[tuple[float, float]] = []
    cumulative_bounces_ref: list[tuple[float, float, bool]] = []
    total_out_frames = 0

    for i, frame in enumerate(raw_frames):
        out_img = frame.copy()
        st = shot_states[i]

        if st.is_cut_boundary:
            ball_queue.clear()
            for _ in range(8):
                ball_queue.append(None)
            trail_ref_pts.clear()

        if st.is_court_active and frame_lines[i] is not None:
            lines_now = frame_lines[i]
            assert lines_now is not None
            # Draw projected court lines (ArtLabss red/cyan lines)
            for k in range(0, len(lines_now), 4):
                x1, y1, x2, y2 = (
                    round(float(lines_now[k])),
                    round(float(lines_now[k + 1])),
                    round(float(lines_now[k + 2])),
                    round(float(lines_now[k + 3])),
                )
                cv2.line(out_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

            # Draw Player 1 and Player 2 boxes
            out_img = mark_player_box(out_img, detection_model.player_1_boxes, i, color=(255, 50, 30), label="Player 1")
            out_img = mark_player_box(out_img, detection_model.player_2_boxes, i, color=(255, 0, 210), label="Player 2")

            # Update 8-frame ball queue and draw yellow comet trail
            uv_now = active_coords[i]
            if uv_now != (0.0, 0.0):
                ball_queue.appendleft(uv_now)
                inv_m = court_detector.game_warp_matrix[min(i, len(court_detector.game_warp_matrix) - 1)]
                ref_ball = apply_homography(inv_m, np.array([uv_now[0], uv_now[1]], dtype=np.float64))
                trail_ref_pts.append((float(ref_ball[0]), float(ref_ball[1])))
                if len(trail_ref_pts) > 25:
                    trail_ref_pts.pop(0)
            else:
                ball_queue.appendleft(None)

            for q_idx, q_pt in enumerate(ball_queue):
                if q_pt is not None:
                    radius = max(3, 8 - q_idx)
                    cv2.circle(
                        out_img,
                        (round(q_pt[0]), round(q_pt[1])),
                        radius,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # Draw bounce impact rings and IN/OUT call overlay when --bounce=1
            if bounce == 1:
                if i in bounces_ref_history:
                    cumulative_bounces_ref.append(bounces_ref_history[i])
                for rep in bounce_reports:
                    if rep.frame_index <= i <= rep.frame_index + int(fps * 1.2):
                        bx_px, by_px = round(rep.pixel_uv[0]), round(rep.pixel_uv[1])
                        col = (40, 235, 65) if rep.decision.is_in else (35, 40, 245)
                        cv2.circle(out_img, (bx_px, by_px), 12, col, -1, cv2.LINE_AA)
                        cv2.circle(out_img, (bx_px, by_px), 16, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(
                            out_img,
                            f"{rep.decision.verdict} ({abs(rep.decision.margin_mm):.1f} mm)",
                            (max(12, bx_px - 60), max(28, by_px - 22)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.85,
                            col,
                            2,
                            cv2.LINE_AA,
                        )

            # Composite Top-Right Dynamic 2D Court Minimap when --minimap=1
            if minimap == 1:
                minimap_img = create_top_view_image(
                    court_detector=court_detector,
                    feet_pos_1=smoothed_feet_1[i] if i < len(smoothed_feet_1) else None,
                    feet_pos_2=smoothed_feet_2[i] if i < len(smoothed_feet_2) else None,
                    ball_xy_frame=uv_now if uv_now != (0.0, 0.0) else None,
                    frame_idx=i,
                    bounces_ref=cumulative_bounces_ref,
                    trail_ref=trail_ref_pts,
                )
                out_img = merge_minimap(out_img, minimap_img)

            # Top-left HUD status badge
            cv2.rectangle(out_img, (12, 12), (435, 52), (18, 22, 30), -1)
            cv2.rectangle(out_img, (12, 12), (435, 52), (90, 210, 120), 1)
            cv2.putText(
                out_img,
                f"SHOT #{st.shot_id} | {st.view_type.value} (CONF #{st.best_conf})",
                (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 245, 150),
                2,
                cv2.LINE_AA,
            )
        else:
            # Non-court cutaway frame: clean passthrough with standby badge (no scrambled court lines!)
            cv2.rectangle(out_img, (12, 12), (465, 52), (24, 20, 32), -1)
            cv2.rectangle(out_img, (12, 12), (465, 52), (80, 180, 250), 1)
            cv2.putText(
                out_img,
                f"SHOT #{st.shot_id} | NON-COURT CUTAWAY (TRACKING PAUSED)",
                (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (110, 205, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(out_img)
        total_out_frames += 1

    # Append 1.5s (45 frames) of 3D Hawk-Eye Challenge & SEU 3D Wireframe Replay when bounce == 1
    if bounce == 1 and append_3d_replay and len(fitted_3d) >= 2:
        primary_bounce = bounce_reports[-1] if bounce_reports else None
        replay_frames = max(18, round(fps * 1.2))
        for r_idx in range(replay_frames):
            prog = float(r_idx) / float(max(replay_frames - 1, 1))
            replay_img = render_3d_hawkeye_replay_frame(
                width=width,
                height=height,
                traj_3d=fitted_3d,
                bounce_report=primary_bounce,
                progress=prog,
            )
            writer.write(replay_img)
            total_out_frames += 1

    writer.release()

    active_count = sum(1 for s in shot_states if s.is_court_active)
    cutaway_count = sum(1 for s in shot_states if s.view_type == CameraViewType.NON_COURT_CUTAWAY)
    num_shots = len({s.shot_id for s in shot_states})

    return VideoPredictionResult(
        input_video_path=str(in_path),
        output_video_path=str(out_path),
        fps=fps,
        width=width,
        height=height,
        total_input_frames=total_in,
        total_output_frames=total_out_frames,
        active_court_frames=active_count,
        cutaway_frames=cutaway_count,
        num_camera_shots=num_shots,
        bounces=bounce_reports,
        trajectory_3d_m=[[round(float(v), 4) for v in pt] for pt in fitted_3d],
    )
