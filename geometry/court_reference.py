"""CourtReference model (12 four-point configurations) and 26-point 3D ITF world coordinates.

Combines the 2D reference canvas and 12 homography configurations from
``ArtLabss/tennis-tracking`` (``court_reference.py``) with the 26-point 3D ITF
world coordinate table (``23.77 m x 10.97 m`` court + ``1.20 m`` net elevation)
from ``GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# ITF Regulation Metric Constants (Meters)
# ---------------------------------------------------------------------------
ITF_COURT_LENGTH_M: float = 23.77
ITF_DOUBLES_WIDTH_M: float = 10.97
ITF_SINGLES_WIDTH_M: float = 8.23
ITF_ALLEY_WIDTH_M: float = 1.37
ITF_HALF_LENGTH_M: float = 11.885
ITF_SERVICE_LINE_FROM_NET_M: float = 6.40
ITF_NEAR_SERVICE_X_M: float = 5.485
ITF_FAR_SERVICE_X_M: float = 18.285
ITF_CENTER_Y_M: float = 5.485
ITF_LINE_WIDTH_M: float = 0.05
ITF_NET_HEIGHT_CENTER_M: float = 0.914
ITF_NET_HEIGHT_POST_M: float = 1.07
TENNIS_BALL_RADIUS_M: float = 0.0335

# ---------------------------------------------------------------------------
# SEU 26-Point 3D World Coordinates Table (X in [0, 23.77], Y in [0, 10.97], Z)
# Indices 0..20 lie on the Z = 0 court plane; indices 21..25 lie along the net top.
# ---------------------------------------------------------------------------
WORLD_COORDINATES_3D: NDArray[np.float64] = np.array(
    [
        # Near baseline (X = 0.0)
        [0.0, 0.0, 0.0],
        [0.0, 1.370, 0.0],
        [0.0, 5.485, 0.0],
        [0.0, 9.600, 0.0],
        [0.0, 10.970, 0.0],
        # Near service line (X = 5.485)
        [5.485, 1.370, 0.0],
        [5.485, 5.485, 0.0],
        [5.485, 9.600, 0.0],
        # Net ground line (X = 11.885)
        [11.885, 0.0, 0.0],
        [11.885, 1.370, 0.0],
        [11.885, 5.485, 0.0],
        [11.885, 9.600, 0.0],
        [11.885, 10.970, 0.0],
        # Far service line (X = 18.285)
        [18.285, 1.370, 0.0],
        [18.285, 5.485, 0.0],
        [18.285, 9.600, 0.0],
        # Far baseline (X = 23.770)
        [23.770, 0.0, 0.0],
        [23.770, 1.370, 0.0],
        [23.770, 5.485, 0.0],
        [23.770, 9.600, 0.0],
        [23.770, 10.970, 0.0],
        # Net top elevation keypoints (X = 11.885, Z = 1.2)
        [11.885, 0.0, 1.2],
        [11.885, 1.370, 1.2],
        [11.885, 5.485, 1.2],
        [11.885, 9.600, 1.2],
        [11.885, 10.970, 1.2],
    ],
    dtype=np.float64,
)

# Wireframe index groups used by SEU RECONSTRUCTION.py 3D court plot
SEU_WIREFRAME_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 4, 20, 16, 0),  # Doubles outer boundary
    (1, 3, 19, 17, 1),  # Singles boundary
    (5, 7, 15, 13, 5),  # Service boxes perimeter
    (6, 14),  # Center service line
    (8, 21, 25, 12),  # 3D Net mesh post-to-post rectangle
)


@dataclass(frozen=True)
class CourtLineSegment:
    """Named court line segment in both metric (m) and reference canvas (px) coordinates."""

    name: str
    p1_px: tuple[int, int]
    p2_px: tuple[int, int]
    p1_m: tuple[float, float]
    p2_m: tuple[float, float]
    width_m: float = ITF_LINE_WIDTH_M


class CourtReference:
    """2D reference court canvas & 12 four-point configurations from ``ArtLabss/tennis-tracking``.

    Unlike the original ``ArtLabss`` implementation that crashed if
    ``court_configurations/court_reference.png`` was missing on disk, this class
    builds the binary reference mask deterministically in memory while maintaining
    100% coordinate compatibility with ``ArtLabss``.
    """

    def __init__(self) -> None:
        # Exact pixel coordinates on the 1665 x 3506 reference canvas
        self.baseline_top: tuple[tuple[int, int], tuple[int, int]] = ((286, 561), (1379, 561))
        self.baseline_bottom: tuple[tuple[int, int], tuple[int, int]] = ((286, 2935), (1379, 2935))
        self.net: tuple[tuple[int, int], tuple[int, int]] = ((286, 1748), (1379, 1748))
        self.left_court_line: tuple[tuple[int, int], tuple[int, int]] = ((286, 561), (286, 2935))
        self.right_court_line: tuple[tuple[int, int], tuple[int, int]] = ((1379, 561), (1379, 2935))
        self.left_inner_line: tuple[tuple[int, int], tuple[int, int]] = ((423, 561), (423, 2935))
        self.right_inner_line: tuple[tuple[int, int], tuple[int, int]] = ((1242, 561), (1242, 2935))
        self.middle_line: tuple[tuple[int, int], tuple[int, int]] = ((832, 1110), (832, 2386))
        self.top_inner_line: tuple[tuple[int, int], tuple[int, int]] = ((423, 1110), (1242, 1110))
        self.bottom_inner_line: tuple[tuple[int, int], tuple[int, int]] = ((423, 2386), (1242, 2386))
        self.top_extra_part: tuple[float, float] = (832.5, 580.0)
        self.bottom_extra_part: tuple[float, float] = (832.5, 2910.0)

        # 12 four-point configurations for full or partial court homography matching
        self.court_conf: dict[int, list[tuple[int, int]]] = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [
                self.left_inner_line[0],
                self.right_inner_line[0],
                self.left_inner_line[1],
                self.right_inner_line[1],
            ],
            3: [
                self.left_inner_line[0],
                self.right_court_line[0],
                self.left_inner_line[1],
                self.right_court_line[1],
            ],
            4: [
                self.left_court_line[0],
                self.right_inner_line[0],
                self.left_court_line[1],
                self.right_inner_line[1],
            ],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
            8: [
                self.right_inner_line[0],
                self.right_court_line[0],
                self.right_inner_line[1],
                self.right_court_line[1],
            ],
            9: [
                self.left_court_line[0],
                self.left_inner_line[0],
                self.left_court_line[1],
                self.left_inner_line[1],
            ],
            10: [
                self.top_inner_line[0],
                self.middle_line[0],
                self.bottom_inner_line[0],
                self.middle_line[1],
            ],
            11: [
                self.middle_line[0],
                self.top_inner_line[1],
                self.middle_line[1],
                self.bottom_inner_line[1],
            ],
            12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
        }

        self.line_width: int = 1
        self.court_width: int = 1117
        self.court_height: int = 2408
        self.top_bottom_border: int = 549
        self.right_left_border: int = 274
        self.court_total_width: int = self.court_width + self.right_left_border * 2
        self.court_total_height: int = self.court_height + self.top_bottom_border * 2

        self.court: NDArray[np.uint8] = self.build_court_reference()

    def build_court_reference(self, save_path: str | Path | None = None) -> NDArray[np.uint8]:
        """Create the binary court reference image (values 0 and 1) using line positions."""
        court = np.zeros(
            (
                self.court_height + 2 * self.top_bottom_border,
                self.court_width + 2 * self.right_left_border,
            ),
            dtype=np.uint8,
        )
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_line, 1, self.line_width)
        court = np.asarray(cv2.dilate(court, np.ones((5, 5), dtype=np.uint8)), dtype=np.uint8)
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), (court * 255).astype(np.uint8))
        self.court = court
        return court

    def get_important_lines(self) -> list[tuple[int, int]]:
        """Return flat list of 20 endpoints (10 lines x 2 endpoints) on the reference court."""
        return [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.net,
            *self.left_court_line,
            *self.right_court_line,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.middle_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
        ]

    def get_extra_parts(self) -> list[tuple[float, float]]:
        """Return center marks near the top and bottom baselines."""
        return [self.top_extra_part, self.bottom_extra_part]

    def get_court_mask(self, mask_type: int = 0) -> NDArray[np.uint8]:
        """Get binary region-of-interest mask of the reference court.

        ``mask_type``:
        - ``0``: Full canvas
        - ``1``: Bottom half court (near player ROI)
        - ``2``: Top half court (far player ROI)
        - ``3``: Inner court without outer margins
        """
        mask = np.ones_like(self.court, dtype=np.uint8)
        if mask_type == 1:
            mask[: self.net[0][1] - 1000, :] = 0
        elif mask_type == 2:
            mask[self.net[0][1] :, :] = 0
        elif mask_type == 3:
            mask[: self.baseline_top[0][1], :] = 0
            mask[self.baseline_bottom[0][1] :, :] = 0
            mask[:, : self.left_court_line[0][0]] = 0
            mask[:, self.right_court_line[0][0] :] = 0
        return mask

    def pixel_to_meters(self, u_px: float, v_px: float) -> tuple[float, float]:
        """Convert ``CourtReference`` canvas pixel ``(u_px, v_px)`` to SEU/ITF meters ``(X_m, Y_m)``.

        In SEU convention:
        - ``X_m`` runs longitudinally from ``0.0 m`` (top/far baseline ``v = 561``)
          to ``23.77 m`` (bottom/near baseline ``v = 2935``).
        - ``Y_m`` runs transversely from ``0.0 m`` (left doubles sideline ``u = 286``)
          to ``10.97 m`` (right doubles sideline ``u = 1379``).
        """
        x_span_px = float(self.baseline_bottom[0][1] - self.baseline_top[0][1])  # 2374 px
        y_span_px = float(self.right_court_line[0][0] - self.left_court_line[0][0])  # 1093 px
        x_m = ((float(v_px) - float(self.baseline_top[0][1])) / x_span_px) * ITF_COURT_LENGTH_M
        y_m = ((float(u_px) - float(self.left_court_line[0][0])) / y_span_px) * ITF_DOUBLES_WIDTH_M
        return (x_m, y_m)

    def meters_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Convert SEU/ITF meters ``(X_m, Y_m)`` to ``CourtReference`` canvas pixel ``(u_px, v_px)``."""
        x_span_px = float(self.baseline_bottom[0][1] - self.baseline_top[0][1])
        y_span_px = float(self.right_court_line[0][0] - self.left_court_line[0][0])
        v_px = float(self.baseline_top[0][1]) + (float(x_m) / ITF_COURT_LENGTH_M) * x_span_px
        u_px = float(self.left_court_line[0][0]) + (float(y_m) / ITF_DOUBLES_WIDTH_M) * y_span_px
        return (u_px, v_px)

    def get_line_segments(self) -> list[CourtLineSegment]:
        """Return all 9 painted court lines with both pixel and metric coordinates."""
        raw = [
            ("baseline_top", self.baseline_top),
            ("baseline_bottom", self.baseline_bottom),
            ("left_doubles_sideline", self.left_court_line),
            ("right_doubles_sideline", self.right_court_line),
            ("left_singles_sideline", self.left_inner_line),
            ("right_singles_sideline", self.right_inner_line),
            ("top_service_line", self.top_inner_line),
            ("bottom_service_line", self.bottom_inner_line),
            ("center_service_line", self.middle_line),
        ]
        segments: list[CourtLineSegment] = []
        for name, (p1, p2) in raw:
            segments.append(
                CourtLineSegment(
                    name=name,
                    p1_px=p1,
                    p2_px=p2,
                    p1_m=self.pixel_to_meters(p1[0], p1[1]),
                    p2_m=self.pixel_to_meters(p2[0], p2[1]),
                )
            )
        return segments
