# ADR-002: Multi-Plane KNN Outlier Filtering & Endpoint-Constrained 3D Curve Reconstruction

**Status:** Accepted

## Context

In `GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system`, 3D ball coordinates reconstructed from noisy 2D detections exhibit two structural problems:
1. **Clumpy trajectory noise:** Outliers cluster near the flight path and cannot be removed by a simple 1D threshold.
2. **Segment discontinuity:** Fitting independent polynomials to each flight segment creates gaps at bounce and racket-hit points.

## Decision

1. **Multi-Plane 2D KNN Filtering (`knn_filter_outliers`):** Project 3D candidate points onto orthogonal 2D view planes (`xy` top-down, `xz` side view, `yz` end view) and use a `scipy.spatial.cKDTree` K-Nearest Neighbor distance filter (`d_k > mean(d_k) + deviation`) to strip clumpy outliers before curve fitting.
2. **Endpoint-Constrained Curve Optimization (`fit_curve_with_constraints`):** Segment the rally at detected events (`find_service`, `find_bounce`, `find_hit`) and optimize each segment with `scipy.optimize.minimize` (BFGS) on:
   $$\mathcal{L}(\theta) = w_{\text{end}}\Big((\hat{y}_0 - y_{\text{start}})^2 + (\hat{y}_{-1} - y_{\text{end}})^2\Big) + \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$$
   anchoring every bounce strictly to the calibrated court plane $Z(t_{\text{bounce}}) = r_{\text{ball}}$ (`0.0335 m`).

## Consequences

- Consecutive 3D flight arcs stitch continuously at every bounce and player hit without gaps.
- Generalizes `SEU`'s hardcoded 6-segment script to handle arbitrary rallies of any length.
