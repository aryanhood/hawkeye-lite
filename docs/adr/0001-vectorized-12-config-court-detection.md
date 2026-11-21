# ADR-001: Vectorized 12-Configuration Court Detection & Dynamic Broadcast Cut Recovery

**Status:** Accepted

## Context

In `ArtLabss/tennis-tracking`, `CourtDetector` matched horizontal and vertical white line intersections against 12 four-point court configurations (`court_conf[1..12]`). However:
1. `_filter_pixels` used a nested Python loop over every pixel (`for i in range(...) for j in range(...)`), and `line_intersection` instantiated `sympy.Line` symbolic objects inside nested combinations—causing a 15-second video to take **16 to 28 minutes**.
2. Real tennis broadcast videos cut frequently between main rally cameras, zoomed partial-court cameras, and non-court cutaways (spectators, close-ups, graphic transitions).

## Decision

1. Vectorize `_filter_pixels` using NumPy array slicing (`g[tau:-tau, tau:-tau] - g[2*tau:, tau:-tau] > th`) and replace `sympy.Line` with closed-form 2D homogeneous cross-products (`line_intersection`), accelerating court calibration by **>150×** (`< 8 ms` per frame on CPU).
2. Gate every frame through `ShotSegmenter` (HSV histogram Bhattacharyya distance + court surface chromaticity ratio + `court_score`). On non-court cutaways, tracking pauses cleanly without drawing court lines on spectators or player faces; on camera cuts or when `track_court` finds `< 50` inlier line points (`CAMERA ... HAS BEEN MOVED`), `CourtDetector.detect()` automatically re-locks the best configuration among `court_conf[1..12]`.

## Consequences

- Full 720p videos process in seconds on a standard CPU with zero external binary checkpoints.
- Partial-court broadcast zooms automatically match service-box or tramline configurations (`court_conf[5..12]`), while cutaway shots pass through cleanly.
