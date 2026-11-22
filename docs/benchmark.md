# Benchmark Report: Multi-View DLT vs. Monocular Court-Plane Projection

All 4 methods operate **strictly from independent noisy 2D pixel observations**
(`sigma = 1.0 px`, `6%` random `+/-80 px` clutter spikes across `100 SHA-256 seeds x 3 camera modes`)
with **zero oracle 3D coordinates, zero oracle bounce indices, zero oracle impact speeds**,
and **identical ITF skid-footprint rules** across every primary comparison.
Tight calls within `+/-35 mm` (`N = 242`) are balanced at `49.6%` ground-truth `IN`.

## Primary 300-Scenario Comparison (2D Pixels Only)

| Metric | 1-Cam Static Frame-0 Homography | 1-Cam Dynamic Homography + 2D Outlier Interp | 2-Cam DLT 3D + KNN + Sub-Frame Ballistic Fit | **4-Cam Stadium DLT 3D + KNN + Sub-Frame Ballistic Fit** |
|---|---:|---:|---:|---:|
| Bounce localization error — median | 3625.14 mm | 2708.21 mm | 52.64 mm | **31.27 mm** |
| Bounce localization error — p90 | 7259.74 mm | 8029.80 mm | 146.97 mm | **87.62 mm** |
| Bounce localization error — mean | 3990.64 mm | 3260.23 mm | 82.58 mm | **58.87 mm** |
| Overall line-call accuracy (`N = 300`) | 51.33% | 53.33% | 79.33% | **83.33%** |
| Tight-call accuracy within `+/-35 mm` (`N = 242`) | 53.72% | 54.96% | 76.86% | **80.17%** |
| Uncertainty calibration — Brier score (lower is better) | — | — | 0.1568 | **0.1149** |
| Lateral 95% CI coverage (`|err_y| <= 1.96 * sigma_y`) | — | — | 76.33% | **84.67%** |

## Controlled Skid-Footprint Ablation (2-Cam DLT 3D Fixed)

| Decision Rule (Same 2-Cam DLT 3D Coordinates) | Overall Accuracy | Tight Accuracy (`+/-35 mm`) |
|---|---:|---:|
| Center-point only (`use_skid_footprint=False`) | 73.33% | 70.25% |
| **Oblique skid ellipse (`use_skid_footprint=True`)** | **79.33%** | **76.86%** |

## Key Technical Findings

1. **Monocular `Z=0` Parallax Bottleneck (2708.21 mm median error):** Even with exact per-frame court homography `H_t`, 2D bounce-cusp detection suffers ~50-90 ms timing jitter because out-of-plane elevation `Z(t)` foreshortens along the viewing ray.
2. **Multi-View Geometry + Ballistic Fitting (52.64 mm in 2-Cam, 31.27 mm in 4-Cam):** Triangulating calibrated pixel streams via DLT, stripping spikes with multi-plane KNN, and intersecting pre/post-bounce ballistic curves at `Z = r_ball` cuts mean error by **3177.7 mm** over dynamic 1-cam tracking.
3. **Honest Limitations:** Real uncalibrated single-camera broadcast videos (`hawkeye-lite predict`) still operate strictly on the `Z=0` court plane unless a second calibrated view is available.
