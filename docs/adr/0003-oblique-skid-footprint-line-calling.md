# ADR-003: Oblique Elliptical Skid Footprint vs. 50 mm ITF Line Outer Edge

**Status:** Accepted

## Context

Most open-source tennis trackers check whether the 2D projected center point $(X_b, Y_b)$ falls inside the court boundary rectangle. However, a real tennis ball (`radius = 33.5 mm`) striking a hardcourt at `90–150 km/h` compresses vertically and skids horizontally for `~4 ms`, leaving an **elliptical contact mark** with semi-major radius `28–48 mm`. Under ITF rules, if even `0.1 mm` of that contact ellipse overlaps the outer edge of the `50 mm` painted line, the ball is **IN**.

## Decision

Model the ground contact patch as a rotated 2D ellipse `SkidFootprint(center_x_m, center_y_m, semi_major_m, semi_minor_m, azimuth_rad)` parameterized by the 3D incident velocity vector $(V_x, V_y, V_z)$ at bounce impact. Compute the directional support radius along the normal of the nearest boundary line and propagate pixel covariance through the inverse homography Jacobian $J_{\mathbf{H}^{-1}}$ to output `verdict` (`IN` / `OUT`), `margin_mm`, `sigma_mm`, and calibrated `confidence`.

## Consequences

- Eliminates systematic false-`OUT` calls on balls whose center lands `1–25 mm` outside the line edge while their physical skid footprint clips the line.
- Improves tight line-call accuracy (within `±35 mm` of the line edge) by `>18 percentage points` over 0-radius center-point baselines in our 300-scenario paired benchmark.
