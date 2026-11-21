"""Tests for CourtDetector, ShotSegmenter, DetectionModel, TrackNetMotionBallDetector, and Minimap."""

from __future__ import annotations

import numpy as np
from camera.calibration import build_lookat_camera
from evaluation.synthetic_video import render_court_3d_frame, render_cutaway_frame
from vision.court_detector import CourtDetector
from vision.detection import (
    DetectionModel,
    TrackNetMotionBallDetector,
    area_of_box,
    boxes_dist,
    center_of_box,
    create_top_view_image,
    mark_player_box,
    merge_minimap,
    sections_intersect,
)
from vision.shot_segmenter import CameraViewType, ShotSegmenter


def test_court_detector_detect_and_track_and_cutaway_fallback() -> None:
    cam1 = build_lookat_camera(eye_world=(31.0, 5.485, 9.2), target_world=(11.2, 5.485, 0.0))
    cam2 = build_lookat_camera(eye_world=(31.0, 5.620, 9.2), target_world=(11.2, 5.620, 0.0))

    frame1 = render_court_3d_frame(
        cam1,
        ball_pos_3d=np.array([15.0, 5.2, 1.4], dtype=np.float64),
        p1_xy_m=(22.5, 4.8),
        p2_xy_m=(1.2, 6.0),
    )
    frame2 = render_court_3d_frame(
        cam2,
        ball_pos_3d=np.array([13.2, 5.6, 1.2], dtype=np.float64),
        p1_xy_m=(22.5, 4.9),
        p2_xy_m=(1.2, 6.1),
    )

    detector = CourtDetector()
    lines1 = detector.detect(frame1)
    assert len(lines1) == 40
    assert detector.court_score > 400.0
    assert detector.best_conf is not None
    assert detector.get_court_accuracy(frame1) > 40.0

    overlay = detector.add_court_overlay(frame1)
    assert overlay.shape == frame1.shape

    # Track court across camera pan
    lines2 = detector.track_court(frame2)
    assert len(lines2) == 40
    assert len(detector.court_warp_matrix) >= 2

    # Classify vertical lines helper coverage
    filtered = detector._filter_pixels(detector._threshold(frame1))
    _, v_lines = detector._detect_lines(filtered)
    vc, vl, vr = detector._classify_vertical(v_lines, frame1.shape[1])
    assert (len(vc) + len(vl) + len(vr)) == len(v_lines)

    # Test cutaway frame triggers fallback re-detection cleanly without crashing
    cutaway = render_cutaway_frame()
    lines_cut = detector.track_court(cutaway)
    assert len(lines_cut) == 40


def test_shot_segmenter_and_player_ball_detection_and_minimap() -> None:
    cam = build_lookat_camera(eye_world=(31.0, 5.485, 9.2), target_world=(11.2, 5.485, 0.0))
    f_court_1 = render_court_3d_frame(
        cam,
        ball_pos_3d=np.array([16.0, 5.0, 1.3], dtype=np.float64),
        p1_xy_m=(22.5, 4.8),
        p2_xy_m=(1.2, 6.0),
    )
    f_court_2 = render_court_3d_frame(
        cam,
        ball_pos_3d=np.array([14.0, 5.5, 1.1], dtype=np.float64),
        p1_xy_m=(22.5, 5.0),
        p2_xy_m=(1.2, 6.2),
    )
    f_cut = render_cutaway_frame()

    detector = CourtDetector()
    seg = ShotSegmenter()

    detector.detect(f_court_1)
    st1 = seg.evaluate_frame(f_court_1, 0, detector.court_score, detector.best_conf)
    assert st1.is_court_active

    detector.detect(f_cut)
    st_cut = seg.evaluate_frame(f_cut, 1, detector.court_score, detector.best_conf)
    assert st_cut.view_type == CameraViewType.NON_COURT_CUTAWAY
    assert not st_cut.is_court_active
    assert st_cut.is_cut_boundary

    # Test player & ball detector and top-down minimap
    det_model = DetectionModel()
    ball_det = TrackNetMotionBallDetector()

    detector.detect(f_court_1)
    det_model.detect_player_1(f_court_1, detector)
    det_model.detect_top_persons(f_court_1, detector, 1)
    b1 = ball_det.detect_ball(f_court_1, court_warp_matrix=detector.court_warp_matrix[-1])

    detector.track_court(f_court_2)
    det_model.detect_player_1(f_court_2, detector)
    det_model.detect_top_persons(f_court_2, detector, 2)
    b2 = ball_det.detect_ball(f_court_2, court_warp_matrix=detector.court_warp_matrix[-1])

    assert b1 is not None or b2 is not None
    # Inject synthetic multi-person top tracks to exercise find_player_2_box discrimination
    det_model.persons_boxes[10] = [[100.0, 50.0, 120.0, 90.0], [130.0, 55.0, 150.0, 95.0]]
    det_model.persons_first_appearance[10] = 1
    det_model.persons_boxes[11] = [[300.0, 50.0, 320.0, 90.0], [301.0, 50.0, 321.0, 90.0]]
    det_model.persons_first_appearance[11] = 1
    p2_boxes = det_model.find_player_2_box()
    assert len(p2_boxes) == 2

    feet1, feet2 = det_model.calculate_feet_positions(detector)
    assert feet1.shape == (2, 2)
    assert feet2.shape == (2, 2)

    minimap = create_top_view_image(
        court_detector=detector,
        feet_pos_1=feet1[1],
        feet_pos_2=feet2[1],
        ball_xy_frame=b2 or b1,
        frame_idx=1,
        bounces_ref=[(832.0, 1110.0, True)],
        trail_ref=[(832.0, 2000.0), (832.0, 1500.0)],
    )
    merged = merge_minimap(f_court_2, minimap)
    assert merged.shape == f_court_2.shape

    marked = mark_player_box(f_court_2.copy(), det_model.player_1_boxes, 1, label="P1")
    assert marked.shape == f_court_2.shape

    assert center_of_box([None, None, None, None]) == (None, None)
    assert center_of_box([10.0, 20.0, 30.0, 40.0]) == (20.0, 30.0)
    assert area_of_box([10.0, 20.0, 30.0, 50.0]) == 600.0
    assert boxes_dist([[0.0, 0.0, 10.0, 10.0], [3.0, 4.0, 13.0, 14.0]]) == 5.0
    assert sections_intersect([1, 10], [5, 15])
    assert not sections_intersect([1, 4], [5, 10])
