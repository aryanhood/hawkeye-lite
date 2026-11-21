# Changelog

All notable changes to `hawkeye-lite` are documented in this file.

## [2.0.0] - 2026-11-21

### Added
- **Video-In → Video-Out Broadcast Pipeline (`evaluation/predict_video.py`, `hawkeye-lite predict`):** End-to-end offline processor accepting `.mp4` tennis clips and outputting annotated `.mp4` videos with live court line overlays, player bounding boxes, 8-frame ball trajectory trail, top-right dynamic 2D court minimap (`--minimap=1`), bounce calls (`--bounce=1`), and an appended 3D Australian Open / SEU Hawk-Eye wireframe & zoom replay cut-in.
- **12-Configuration Court Reference & 150× Vectorized Court Detector (`geometry/court_reference.py`, `vision/court_detector.py`):** Adapted and vectorized from `ArtLabss/tennis-tracking` with 26-point 3D ITF world coordinates (`23.77 m × 10.97 m`) from `SEU-Robot-Vision-Hawkeye`.
- **Multi-Angle Broadcast Shot Segmenter (`vision/shot_segmenter.py`):** Automatically detects hard camera cuts and dissolves, gates out non-court cutaway frames (spectators, logos, close-ups), and re-locks court homography across camera angle switches and pans.
- **SORT Multi-Object Player & Ball Tracking + 3D KNN Trajectory Reconstruction (`tracking/sort.py`, `tracking/interpolation.py`, `tracking/reconstruction.py`):** 3-pass outlier removal, linear/Savitzky-Golay interpolation, multi-plane (`xy`, `yz`, `xz`) KNN outlier filtering, event segmentation (`find_service`, `find_bounce`, `find_hit`), and endpoint-constrained 3D parabolic/aerodynamic curve fitting (`start_loss + end_loss + data_loss`).
- **Oblique Skid Footprint Line-Call Engine (`decision/line_call.py`):** Millimeter-exact overlap verification against the 50 mm ITF line outer edge (`IN` / `OUT` + margin in `mm` + confidence).
- **300-Scenario Paired Benchmark & Zero-Backend React 19 + TypeScript Web Studio (`evaluation/benchmark.py`, `web/`):** Compares Raw 2D Homography vs. 2D Interpolated vs. Full 3D Constrained Hawk-Eye Lite across camera angles and pan/cut conditions.
