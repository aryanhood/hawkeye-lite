"""Millimeter-exact Australian Open Hawk-Eye Line-Calling Engine with Oblique Skid Footprint.

When a tennis ball impacts an acrylic hardcourt (Plexicushion / GreenSet) at high horizontal
speed ``v_xy`` and vertical speed ``|v_z|``, it compresses and skids along its flight azimuth,
leaving an **elliptical contact footprint** with:
- Semi-minor axis ``b`` (transverse compression radius, typically ``18-29 mm`` depending on ``|v_z|``)
- Semi-major axis ``a`` (longitudinal skid radius along horizontal velocity vector, typically ``28-52 mm``)

ITF Outer-Edge Rule:
A ball is **IN** if *any* fraction of its contact footprint touches the painted court line
(outer edge included). It is **OUT** only if 100% of the contact footprint lies outside the
outer edge of the boundary line.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
from geometry.court_reference import (
    ITF_ALLEY_WIDTH_M,
    ITF_COURT_LENGTH_M,
    ITF_DOUBLES_WIDTH_M,
    ITF_LINE_WIDTH_M,
    ITF_SINGLES_WIDTH_M,
    TENNIS_BALL_RADIUS_M,
)
from geometry.homography import homography_jacobian
from numpy.typing import NDArray


@dataclass(frozen=True)
class SkidFootprint:
    """Elliptical ball contact patch on the court surface ``Z = 0``."""

    center_x_m: float
    center_y_m: float
    semi_major_m: float  # along velocity azimuth
    semi_minor_m: float  # perpendicular to velocity azimuth
    azimuth_rad: float  # heading in court (X, Y) plane

    def directional_radius_m(self, normal_angle_rad: float) -> float:
        """Return effective support radius (meters) of the rotated ellipse along ``normal_angle_rad``."""
        delta = normal_angle_rad - self.azimuth_rad
        c = float(np.cos(delta))
        s = float(np.sin(delta))
        return float(np.sqrt((self.semi_major_m * c) ** 2 + (self.semi_minor_m * s) ** 2))


@dataclass(frozen=True)
class LineCallDecision:
    """Electronic Line Calling (ELC) verdict with millimeter margin and uncertainty confidence."""

    verdict: str  # "IN" or "OUT"
    is_in: bool
    margin_mm: float  # positive = inside / overlapping line; negative = clear gap outside line
    nearest_line_name: str
    sigma_mm: float
    confidence: float  # calibrated probability of the called verdict in [0.5, 1.0]
    probability_in: float  # P(IN) in [0.0, 1.0]
    footprint: SkidFootprint


def compute_skid_footprint(
    bounce_x_m: float,
    bounce_y_m: float,
    vel_incident_m_s: tuple[float, float, float] = (24.0, 2.0, -7.5),
) -> SkidFootprint:
    """Compute the physical elliptical contact patch from 3D impact velocity ``(Vx, Vy, Vz)``."""
    vx, vy, vz = float(vel_incident_m_s[0]), float(vel_incident_m_s[1]), abs(float(vel_incident_m_s[2]))
    v_horiz = float(np.hypot(vx, vy))
    azimuth = float(np.arctan2(vy, vx)) if v_horiz > 1e-6 else 0.0

    # Vertical compression radius scales with normal impact speed |Vz| (capped by ball radius)
    compression_ratio = float(np.clip(0.46 + 0.032 * vz, 0.48, 0.88))
    semi_minor_m = TENNIS_BALL_RADIUS_M * compression_ratio

    # Longitudinal skid elongation scales with tangential speed during ~4.2 ms contact dwell
    skid_elongation = float(np.clip(1.0 + 0.016 * v_horiz, 1.05, 1.68))
    semi_major_m = min(TENNIS_BALL_RADIUS_M * 1.48, semi_minor_m * skid_elongation)

    return SkidFootprint(
        center_x_m=float(bounce_x_m),
        center_y_m=float(bounce_y_m),
        semi_major_m=float(semi_major_m),
        semi_minor_m=float(semi_minor_m),
        azimuth_rad=azimuth,
    )


def evaluate_line_call(
    bounce_x_m: float,
    bounce_y_m: float,
    vel_incident_m_s: tuple[float, float, float] = (24.0, 2.0, -7.5),
    singles: bool = True,
    use_skid_footprint: bool = True,
    homography_img_to_m: NDArray[np.float64] | None = None,
    bounce_uv_px: tuple[float, float] | None = None,
    pixel_sigma_px: float = 0.65,
    explicit_sigma_xy_mm: tuple[float, float] | None = None,
    extra_sigma_mm: float = 0.0,
) -> LineCallDecision:
    """Evaluate an ``IN`` / ``OUT`` tennis line call against ITF singles or doubles court boundaries.

    Outer boundary coordinates (outer edge of the 50 mm line included in the court):
    - Far baseline outer edge: ``X = 0.0 m``
    - Near baseline outer edge: ``X = 23.77 m``
    - Left sideline outer edge: ``Y = 1.37 m`` (singles) or ``0.0 m`` (doubles)
    - Right sideline outer edge: ``Y = 9.60 m`` (singles) or ``10.97 m`` (doubles)
    """
    footprint = compute_skid_footprint(bounce_x_m, bounce_y_m, vel_incident_m_s)

    x_min_outer = 0.0
    x_max_outer = ITF_COURT_LENGTH_M
    if singles:
        y_min_outer = ITF_ALLEY_WIDTH_M
        y_max_outer = ITF_ALLEY_WIDTH_M + ITF_SINGLES_WIDTH_M  # 9.60 m
        left_name = "left_singles_sideline"
        right_name = "right_singles_sideline"
    else:
        y_min_outer = 0.0
        y_max_outer = ITF_DOUBLES_WIDTH_M
        left_name = "left_doubles_sideline"
        right_name = "right_doubles_sideline"

    # Directional footprint extension toward each boundary normal
    r_x = footprint.directional_radius_m(0.0) if use_skid_footprint else 0.0
    r_y = footprint.directional_radius_m(float(np.pi * 0.5)) if use_skid_footprint else 0.0

    # Signed margin to each of the 4 outer edges (positive = inside / touching line; negative = OUT)
    margins_m: list[tuple[str, float, int]] = [
        ("baseline_top", (bounce_x_m + r_x) - x_min_outer, 0),
        ("baseline_bottom", x_max_outer - (bounce_x_m - r_x), 0),
        (left_name, (bounce_y_m + r_y) - y_min_outer, 1),
        (right_name, y_max_outer - (bounce_y_m - r_y), 1),
    ]

    nearest_line_name, min_margin_m, axis_idx = min(margins_m, key=lambda item: item[1])
    margin_mm = float(min_margin_m * 1000.0)
    is_in = margin_mm >= 0.0
    verdict = "IN" if is_in else "OUT"

    # Propagate 3D triangulation covariance or 2D homography Jacobian uncertainty
    if explicit_sigma_xy_mm is not None:
        sigma_mm = float(np.hypot(explicit_sigma_xy_mm[axis_idx], extra_sigma_mm))
        sigma_mm = float(np.clip(sigma_mm, 0.8, 120.0))
    elif homography_img_to_m is not None and bounce_uv_px is not None:
        j_mat = homography_jacobian(homography_img_to_m, bounce_uv_px[0], bounce_uv_px[1])
        cov_uv = np.eye(2, dtype=np.float64) * (pixel_sigma_px**2)
        cov_xy_m2 = j_mat @ cov_uv @ j_mat.T
        sigma_m = float(np.sqrt(max(float(cov_xy_m2[axis_idx, axis_idx]), 1e-8)))
        sigma_mm = float(np.clip(np.hypot(sigma_m * 1000.0, extra_sigma_mm), 0.8, 250.0))
    else:
        sigma_mm = float(np.clip(np.hypot(5.0, extra_sigma_mm), 0.8, 250.0))

    # Gaussian cumulative probability P(IN) = Phi(margin_mm / sigma_mm)
    z_score = margin_mm / max(sigma_mm, 1e-6)
    prob_in = 0.5 * (1.0 + erf(z_score / sqrt(2.0)))
    confidence = prob_in if is_in else (1.0 - prob_in)

    return LineCallDecision(
        verdict=verdict,
        is_in=is_in,
        margin_mm=round(margin_mm, 2),
        nearest_line_name=nearest_line_name,
        sigma_mm=round(sigma_mm, 2),
        confidence=round(float(np.clip(confidence, 0.5, 1.0)), 4),
        probability_in=round(float(np.clip(prob_in, 0.0, 1.0)), 4),
        footprint=footprint,
    )


def distance_to_nearest_court_line_mm(bounce_x_m: float, bounce_y_m: float, singles: bool = True) -> tuple[str, float]:
    """Return ``(line_name, distance_to_line_center_mm)`` for display diagnostics."""
    half_w = ITF_LINE_WIDTH_M * 0.5
    y_left = (ITF_ALLEY_WIDTH_M + half_w) if singles else half_w
    y_right = (ITF_ALLEY_WIDTH_M + ITF_SINGLES_WIDTH_M - half_w) if singles else (ITF_DOUBLES_WIDTH_M - half_w)
    candidates = [
        ("baseline_top", abs(bounce_x_m - half_w) * 1000.0),
        ("baseline_bottom", abs(bounce_x_m - (ITF_COURT_LENGTH_M - half_w)) * 1000.0),
        ("left_sideline", abs(bounce_y_m - y_left) * 1000.0),
        ("right_sideline", abs(bounce_y_m - y_right) * 1000.0),
    ]
    return min(candidates, key=lambda item: item[1])
