# System Architecture: `hawkeye-lite` v2.0.0

`hawkeye-lite` is an offline **Video Input (`.mp4`) → Video Output (`.mp4`)** Electronic Line Calling (ELC) and 3D broadcast replay engine for tennis. It combines:

1. **Monocular Court, Player & Ball Tracking Fundamentals** adapted and vectorized from [`ArtLabss/tennis-tracking`](https://github.com/ArtLabss/tennis-tracking) (`CourtReference` 12-configuration template model, `CourtDetector`, `DetectionModel`, `Sort`, 20-lag bounce velocity matrix, and dynamic top-down minimap).
2. **26-Point 3D World Coordinate Reconstruction & Constrained Curve Fitting** adapted and generalized from [`GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system`](https://github.com/GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system) (`world_coordinates.py`, `least_square_solution_net.py`, multi-plane `xy`/`yz`/`xz` KNN outlier filtering, and endpoint-constrained 3D curve optimization).
3. **2027 Broadcast Multi-Angle Engineering:** 150× vectorized NumPy line filtering, HSV histogram + court-score shot boundary & cutaway gating, per-frame dynamic homography $\mathbf{H}_t$, and an **oblique elliptical skid footprint** line-call engine (`IN` / `OUT` in millimeters against the 50 mm ITF line outer edge).

---

## End-to-End Data Flow

```mermaid
flowchart LR
    VIN["Input Video (.mp4)\n(Broadcast or Tripod)"] --> SEG["1. Shot Segmenter\n(vision/shot_segmenter.py)\nHard-cut & cutaway gate"]
    SEG --> CD["2. 12-Config CourtDetector\n(vision/court_detector.py)\nVectorized H(t) & track_court"]
    CD --> DET["3. Player & Ball Detection\n(vision/detection.py)\n3-Frame Heatmap + SORT"]
    DET --> REC["4. 3D KNN Reconstruction\n(tracking/reconstruction.py)\nxy/yz/xz KNN + Constrained Curve"]
    REC --> CALL["5. Millimeter Line Caller\n(decision/line_call.py)\nOblique Skid Ellipse vs 50mm Line"]
    CALL --> VOUT["Output Video (.mp4)\n• Court & Player Overlay\n• Top-Right 2D Minimap\n• 3D Hawk-Eye Replay Cut-in"]
```

---

## Package Responsibilities

| Package | Modules | Responsibility |
|---|---|---|
| `geometry/` | `court_reference.py`<br>`homography.py`<br>`triangulation.py` | • `CourtReference`: 12 four-point configurations (`court_conf[1..12]`), ROI masks (`get_court_mask`), and 26-point 3D ITF world coordinates (`23.77 m × 10.97 m`, `1.2 m` net).<br>• Fast analytical `line_intersection` (replacing slow `sympy.Line`), Hartley-normalized DLT homography & projection matrix $\mathbf{P}$, and multi-view / plane-constrained 3D triangulation. |
| `camera/` | `calibration.py` | • `PinholeCamera` ($\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{P} = \mathbf{K}[\mathbf{R}|\mathbf{t}]$).<br>• `pose_from_court_homography`: Decomposes planar court homography into an orthonormal 3D rotation $\mathbf{R} \in \text{SO}(3)$ via SVD + `cv2.solvePnP`.<br>• `interpolate_replay_camera`: Smooth 3D virtual camera zoom for the Australian Open Hawk-Eye replay cut-in. |
| `vision/` | `court_detector.py`<br>`shot_segmenter.py`<br>`detection.py` | • `CourtDetector`: White-line threshold, 150× vectorized `_filter_pixels`, Hough line classification & merging, 12-config homography scoring (`c_p - 0.5 * w_p`), and 100-sample `track_court` with automatic re-detection when `CAMERA ... HAS BEEN MOVED`.<br>• `ShotSegmenter`: Detects broadcast cuts and gates out non-court cutaway frames.<br>• `DetectionModel` & `TrackNetMotionBallDetector`: Near/far player tracking, feet projection, pan-compensated 3-frame ball heatmap + `HoughCircles`, and `create_top_view_image` + `merge_minimap`. |
| `tracking/` | `sort.py`<br>`interpolation.py`<br>`reconstruction.py` | • `Sort`, `KalmanBoxTracker`, `BallKalmanTracker`, and pure-NumPy `KalmanFilter`.<br>• `diff_xy`, 3-pass `remove_outliers`, `nan_helper` linear `interpolation`, and Savitzky-Golay smoothing.<br>• SEU 3D `remove_outlier`, `knn_filter_outliers` (`xy`, `yz`, `xz`), `find_bounce`/`find_hit`/`find_service`, 20-lag bounce detector, `fit_curve_with_constraints` (`start_loss + end_loss + data_loss`), and RK4 aerodynamic + Magnus spin integrator. |
| `decision/` | `line_call.py` | • `compute_skid_footprint` & `evaluate_line_call`: Computes the oblique elliptical contact patch from 3D impact velocity and checks overlap against the 50 mm ITF line outer edge (`IN` / `OUT`, `margin_mm`, Jacobian uncertainty `sigma_mm`, and `confidence`). |
| `evaluation/` | `predict_video.py`<br>`synthetic_video.py`<br>`benchmark.py`<br>`export.py`<br>`cli.py` | • `process_video` (`hawkeye-lite predict --minimap 1 --bounce 1`): Full Video-In → Video-Out processor.<br>• `run_benchmark`: 300-scenario paired ablation benchmark.<br>• `export_all`: Deterministic exporter for `docs/benchmark.md` and `web/public/data/hawkeye_data.json` with `--check` CI drift gate. |
