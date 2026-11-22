# System Architecture: `hawkeye-lite`

`hawkeye-lite` is a **Tennis Court Calibration, Multi-View 3D Triangulation, and Electronic Line-Calling Engine** combining:

1. **Vectorized 12-Config Court Calibration & Cutaway Gating** (`vision/court_detector.py`, `vision/shot_segmenter.py`): Closed-form homogeneous line intersections and vectorized pixel filtering across 12 four-point configurations (`ArtLabss/tennis-tracking`), plus HSV Bhattacharyya shot-cut and cutaway gating.
2. **Classical Chroma-Motion Tracking** (`vision/detection.py`, `tracking/sort.py`): `ChromaMotionBallDetector` (deterministic optic-yellow HSV gating + homography-stabilized frame differencing + morphological top-hat + `HoughCircles`) paired with Kalman filtering and SORT player tracking. *(Note: No neural network weights are bundled; learned ball detectors such as TrackNetV2/WASB are natural future replacements.)*
3. **Calibrated Multi-View DLT 3D Reconstruction vs. Monocular Court Projection** (`geometry/triangulation.py`, `tracking/reconstruction.py`):
   - **Multi-View Mode (`reconstruct_multiview_bounce`):** Given $M \ge 2$ calibrated pinhole cameras $\mathbf{P}_k$, performs consensus DLT triangulation, multi-plane `xy`/`xz` KNN outlier filtering, pre/post-bounce ballistic curve fitting to solve sub-frame contact $Z(t^*) = r_{\text{ball}}$, and Gauss-Markov $(\mathbf{J}^\top \mathbf{J})^{-1}$ uncertainty propagation.
   - **Monocular Video Mode (`process_video`):** Single uncalibrated broadcast clips cannot observe elevation $Z(t)$ directly; detections are projected strictly onto the court plane $Z = 0$ (`project_2d_trajectory_to_court_plane`) with explicit parallax uncertainty bounds (`monocular_parallax_bound_m`).
4. **Millimeter Electronic Line Calling** (`decision/line_call.py`): Oblique elliptical contact skid footprint evaluated against the 50 mm ITF outer boundary with propagated spatial uncertainty $\sigma_{\text{mm}}$ and Gaussian $P(\text{IN})$.
