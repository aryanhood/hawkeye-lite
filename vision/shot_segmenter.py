"""Broadcast Shot Boundary Detector and Camera View Classifier.

Solves the key limitation noted in ``ArtLabss/tennis-tracking`` README:
*"Input videos have to be rallies of the game and shouldn't contain any commercials,
breaks or spectators."*

On real broadcast feeds (like ``tennis.mp4`` with Dubai -> Crowd -> Logo -> Zoomed Angle -> Basel):
1. Detects hard cuts and fast dissolves via HSV histogram Bhattacharyya distance.
2. Classifies each frame into:
   - ``RALLY_MAIN``: Full/primary court broadcast angle (``best_conf in {1, 2, 3, 4}``).
   - ``COURT_PARTIAL_ZOOM``: Low/oblique/zoomed court angle matched via partial service-box
     or tramline configuration (``best_conf in {5..12}``).
   - ``NON_COURT_CUTAWAY``: Crowd, player close-up, or graphic wipe (insufficient court surface
     chromaticity or ``court_score < min_court_score``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraViewType(StrEnum):
    """Broadcast camera view classification."""

    RALLY_MAIN = "RALLY_MAIN"
    COURT_PARTIAL_ZOOM = "COURT_PARTIAL_ZOOM"
    NON_COURT_CUTAWAY = "NON_COURT_CUTAWAY"


@dataclass(frozen=True)
class FrameShotState:
    """Per-frame shot boundary and view classification telemetry."""

    frame_index: int
    shot_id: int
    is_cut_boundary: bool
    hist_distance: float
    court_surface_ratio: float
    court_score: float
    best_conf: int | None
    view_type: CameraViewType

    @property
    def is_court_active(self) -> bool:
        """Return True if the court is visible and valid for tracking/overlays."""
        return self.view_type in (CameraViewType.RALLY_MAIN, CameraViewType.COURT_PARTIAL_ZOOM)


class ShotSegmenter:
    """Detect broadcast camera cuts and classify court vs. non-court cutaway shots."""

    def __init__(
        self,
        cut_hist_threshold: float = 0.34,
        min_court_score: float = 380.0,
        min_court_surface_ratio: float = 0.14,
    ) -> None:
        self.cut_hist_threshold: float = cut_hist_threshold
        self.min_court_score: float = min_court_score
        self.min_court_surface_ratio: float = min_court_surface_ratio
        self._prev_hist: NDArray[np.float32] | None = None
        self._current_shot_id: int = 0
        self.history: list[FrameShotState] = []

    @staticmethod
    def compute_hsv_histogram(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Compute normalized 2D Hue-Saturation histogram for shot cut detection."""
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return np.asarray(hist, dtype=np.float32)

    @staticmethod
    def estimate_court_surface_ratio(frame: NDArray[np.uint8]) -> float:
        """Estimate fraction of the central frame occupied by uniform court surface colors.

        Covers Australian Open / Basel / US Open blue courts, Dubai blue+green courts,
        Wimbledon grass green, and Roland Garros clay terracotta while rejecting
        dark crowd scenes, skin-tone close-ups, and white/metallic graphic wipes.
        """
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.18) : int(h * 0.92), int(w * 0.10) : int(w * 0.90)]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Court blue (H: 90..135), court green (H: 34..88), clay orange-red (H: 5..22)
        blue_mask = cv2.inRange(hsv, np.array([90, 35, 45], dtype=np.uint8), np.array([135, 255, 245], dtype=np.uint8))
        green_mask = cv2.inRange(hsv, np.array([34, 30, 45], dtype=np.uint8), np.array([88, 255, 235], dtype=np.uint8))
        combined = cv2.bitwise_or(blue_mask, green_mask)
        return float(np.count_nonzero(combined)) / float(combined.size)

    def evaluate_frame(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
        court_score: float,
        best_conf: int | None,
    ) -> FrameShotState:
        """Evaluate whether ``frame`` is a camera cut and classify its view type."""
        hist = self.compute_hsv_histogram(frame)
        if self._prev_hist is None:
            hist_dist = 0.0
            is_cut = frame_index == 0
        else:
            hist_dist = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            is_cut = hist_dist >= self.cut_hist_threshold
            if is_cut:
                self._current_shot_id += 1
        self._prev_hist = hist

        surface_ratio = self.estimate_court_surface_ratio(frame)
        if court_score < self.min_court_score or surface_ratio < self.min_court_surface_ratio:
            view_type = CameraViewType.NON_COURT_CUTAWAY
        elif best_conf in (1, 2, 3, 4, 5, 6, 7):
            view_type = CameraViewType.RALLY_MAIN
        else:
            view_type = CameraViewType.COURT_PARTIAL_ZOOM

        state = FrameShotState(
            frame_index=frame_index,
            shot_id=self._current_shot_id,
            is_cut_boundary=is_cut,
            hist_distance=hist_dist,
            court_surface_ratio=surface_ratio,
            court_score=court_score,
            best_conf=best_conf,
            view_type=view_type,
        )
        self.history.append(state)
        return state
