# Benchmark Report: `hawkeye-lite` v2.0.0 (300 Paired Broadcast Rally Scenarios)

All three methods run against the **exact same pre-sampled 3D aerodynamic rallies and pixel noise streams**
across `100 SHA-256 seeds x 3 broadcast camera modes` (`static_high`, `panning_broadcast`, `multi_angle_cut`).

## Summary Results (300 Paired Scenarios)

| Metric | Static Frame-0 Homography (Legacy 2D) | `ArtLabss` 2D Dynamic Homography + Interp | **`hawkeye-lite` v2.0 (3D KNN + Skid Ellipse)** |
|---|---:|---:|---:|
| Bounce localization error — median | 165.67 mm | 89.24 mm | **7.12 mm** |
| Bounce localization error — p90 | 4322.26 mm | 153.10 mm | **12.52 mm** |
| Bounce localization error — mean | 1505.61 mm | 91.09 mm | **7.45 mm** |
| Overall line-call accuracy (`N = 300`) | 76.33% | 47.00% | **96.00%** |
| Tight line-call accuracy within `±35 mm` (`N = 171`) | 69.01% | 26.32% | **92.98%** |

## Key Engineering Takeaways

1. **Why Static Frame-0 Homography fails (`1505.61 mm` mean error):** Even a small camera pan or broadcast angle switch invalidates a static 4-corner perspective matrix.
2. **Why 2D Center-Point Interpolation isn't enough on tight calls (`26.32%` vs `92.98%`):** A tennis ball compressing and skidding at `90-140 km/h` leaves an elliptical footprint (`28-46 mm` semi-major radius). Checking only the 0-radius ball center misclassifies balls whose outer skid edge clips the 50 mm line.
3. **Multi-Plane KNN + Constrained 3D Curve Reconstruction saves `83.64 mm` over 2D interpolation** and **`1498.16 mm` over static homography**.
