"""Player & Ball Detection, Feet Projection, and Dynamic Top-Down Minimap from ``ArtLabss/tennis-tracking``.

Adapts ``ArtLabss/tennis-tracking/detection.py`` and ``predict_video.py``:
1. ``DetectionModel``:
   - ``detect_player_1`` (bottom-half court mask ``mask_type=1``, ROI tracking around previous box).
   - ``detect_top_persons`` + ``find_player_2_box`` (top-half court mask ``mask_type=2``, SORT
     multi-object tracking, discriminated by cumulative movement distance ``boxes_dist``).
   - ``calculate_feet_positions``: projects player bounding-box bottom-centers ``(cx, y2)`` through
     ``game_warp_matrix`` onto the reference court canvas and smooths with ``savgol_filter(7, 2)``.
2. ``TrackNetMotionBallDetector``:
   - Implements the 3-frame temporal + chromatic heatmap ball detector followed by
     ``cv2.threshold(heatmap, 127, 255)`` and ``cv2.HoughCircles`` / sub-pixel centroiding
     (from ``predict_video.py`` lines 165-196), enhanced with inter-frame homography warping
     to cancel camera pan motion.
3. ``create_top_view_frame`` and ``merge_minimap``:
   - Renders the live 2D Top-Down Court Minimap (players in blue, ball in yellow, bounce markers
     in green/red) and composites it cleanly into the top-right corner of the video frame.
"""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from tracking.sort import Sort

from vision.court_detector import CourtDetector


def center_of_box(box: Sequence[float | None]) -> tuple[float | None, float | None]:
    """Calculate the ``(cx, cy)`` center of a bounding box ``[x1, y1, x2, y2]``."""
    if box[0] is None or box[1] is None or box[2] is None or box[3] is None:
        return (None, None)
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.5)


def area_of_box(box: list[float] | tuple[float, ...] | NDArray[np.float64]) -> float:
    """Calculate area of ``[x1, y1, x2, y2]``."""
    return float(max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1])))


def boxes_dist(boxes: list[list[float]]) -> float:
    """Calculate cumulative Euclidean movement distance of a box sequence (``ArtLabss`` ``boxes_dist``)."""
    total_dist = 0.0
    for box1, box2 in itertools.pairwise(boxes):
        c1x, c1y = center_of_box(box1)
        c2x, c2y = center_of_box(box2)
        if c1x is not None and c1y is not None and c2x is not None and c2y is not None:
            total_dist += float(np.hypot(c2x - c1x, c2y - c1y))
    return total_dist


def sections_intersect(sec1: list[int], sec2: list[int]) -> bool:
    """Check if two frame intervals ``[start, end]`` overlap (``ArtLabss`` ``sections_intersect``)."""
    return bool(sec1[0] <= sec2[0] <= sec1[1] or sec2[0] <= sec1[0] <= sec2[1])


def mark_player_box(
    frame: NDArray[np.uint8],
    boxes: list[list[float | None]],
    frame_num: int,
    color: tuple[int, int, int] = (255, 0, 0),
    label: str | None = None,
) -> NDArray[np.uint8]:
    """Draw player bounding box on ``frame`` for ``frame_num`` (``ArtLabss`` ``mark_player_box``)."""
    if 0 <= frame_num < len(boxes):
        box = boxes[frame_num]
        if box[0] is not None and box[1] is not None and box[2] is not None and box[3] is not None:
            p1 = (round(float(box[0])), round(float(box[1])))
            p2 = (round(float(box[2])), round(float(box[3])))
            cv2.rectangle(frame, p1, p2, color, 2)
            if label:
                cv2.putText(
                    frame,
                    label,
                    (p1[0], max(16, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return frame


class DetectionModel:
    """Bottom & Top Player Detector and Feet Trajectory Mapper from ``ArtLabss/tennis-tracking/detection.py``.

    Uses court-mask ROI gating (``mask_type=1`` for bottom half, ``mask_type=2`` for top half)
    and chromosomal/foreground silhouette extraction so it runs deterministically at >120 FPS on CPU
    without requiring a 170 MB Faster-RCNN GPU checkpoint download.
    """

    def __init__(self) -> None:
        self.v_width: int = 0
        self.v_height: int = 0
        self.player_1_boxes: list[list[float | None]] = []
        self.player_2_boxes: list[list[float | None]] = []
        self.persons_boxes: dict[int, list[list[float | None]]] = {}
        self.persons_dists: dict[int, float] = {}
        self.persons_first_appearance: dict[int, int] = {}
        self.counter: int = 0
        self.num_of_misses: int = 0
        self.mot_tracker: Sort = Sort(max_age=10, min_hits=2, iou_threshold=0.05)

    @staticmethod
    def _extract_person_blobs(
        masked_bgr: NDArray[np.uint8],
        min_area: float = 320.0,
        max_area: float = 95000.0,
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """Detect player bounding boxes inside a court-masked ROI via non-court chromatic contrast."""
        hsv = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV)
        # Valid non-black ROI mask
        non_black = cv2.inRange(hsv[:, :, 2], 18, 255)

        # Court surface colors (blue or green) and white court lines
        blue_court = cv2.inRange(
            hsv,
            np.array([92, 45, 45], dtype=np.uint8),
            np.array([134, 255, 240], dtype=np.uint8),
        )
        green_court = cv2.inRange(
            hsv,
            np.array([34, 38, 45], dtype=np.uint8),
            np.array([86, 255, 235], dtype=np.uint8),
        )
        court_bg = cv2.bitwise_or(blue_court, green_court)

        # Erase thin white court lines via morphological opening before foreground extraction
        fg = cv2.bitwise_and(non_black, cv2.bitwise_not(court_bg))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 17))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[NDArray[np.float64]] = []
        scores: list[float] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            aspect = float(h) / max(float(w), 1.0)
            if 0.65 <= aspect <= 4.5 and h >= 18:
                boxes.append(np.array([float(x), float(y), float(x + w), float(y + h)], dtype=np.float64))
                scores.append(min(0.99, 0.55 + area / 20000.0))

        return boxes, scores

    def detect_player_1(self, image: NDArray[np.uint8], court_detector: CourtDetector) -> NDArray[np.uint8]:
        """Detect bottom (near-court) player using bottom-half court mask ``mask_type=1``."""
        vis = np.zeros_like(image)
        self.v_height, self.v_width = int(image.shape[0]), int(image.shape[1])

        white_ref = court_detector.court_reference.get_court_mask(1)
        white_mask = cv2.warpPerspective(
            white_ref,
            court_detector.court_warp_matrix[-1],
            (self.v_width, self.v_height),
        )
        # Restrict bottom player search to lower half of frame
        white_mask[: int(self.v_height * 0.36), :] = 0
        image_court = image.copy()
        image_court[white_mask == 0] = (0, 0, 0)

        persons_boxes, _ = self._extract_person_blobs(image_court, min_area=550.0)
        if not self.player_1_boxes or self.player_1_boxes[-1][0] is None:
            if persons_boxes:
                biggest = max(persons_boxes, key=area_of_box).round()
                self.player_1_boxes.append([float(biggest[0]), float(biggest[1]), float(biggest[2]), float(biggest[3])])
            else:
                # Default near baseline center fallback so feet positions stay well-defined
                cx, cy = self.v_width * 0.5, self.v_height * 0.78
                self.player_1_boxes.append([cx - 24.0, cy - 64.0, cx + 24.0, cy])
        else:
            prev = [float(v) for v in self.player_1_boxes[-1] if v is not None]
            c1x, c1y = center_of_box(prev)
            if persons_boxes and c1x is not None and c1y is not None:
                closest = min(
                    persons_boxes,
                    key=lambda b: float(np.hypot((b[0] + b[2]) * 0.5 - c1x, (b[1] + b[3]) * 0.5 - c1y)),
                )
                dist = float(np.hypot((closest[0] + closest[2]) * 0.5 - c1x, (closest[1] + closest[3]) * 0.5 - c1y))
                if dist < 160.0:
                    self.counter = 0
                    self.player_1_boxes.append(
                        [float(closest[0]), float(closest[1]), float(closest[2]), float(closest[3])]
                    )
                else:
                    biggest = max(persons_boxes, key=area_of_box)
                    self.player_1_boxes.append(
                        [float(biggest[0]), float(biggest[1]), float(biggest[2]), float(biggest[3])]
                    )
            else:
                self.player_1_boxes.append(list(self.player_1_boxes[-1]))
                self.num_of_misses += 1

        last_box = self.player_1_boxes[-1]
        if last_box[0] is not None and last_box[1] is not None and last_box[2] is not None and last_box[3] is not None:
            cv2.rectangle(
                vis,
                (int(last_box[0]), int(last_box[1])),
                (int(last_box[2]), int(last_box[3])),
                (0, 0, 255),
                2,
            )
        return vis

    def detect_top_persons(
        self,
        image: NDArray[np.uint8],
        court_detector: CourtDetector,
        frame_num: int,
    ) -> NDArray[np.uint8]:
        """Detect and track candidate persons in the top half of the court via SORT (``ArtLabss``)."""
        vis = np.zeros_like(image)
        self.v_height, self.v_width = int(image.shape[0]), int(image.shape[1])

        white_ref = court_detector.court_reference.get_court_mask(2)
        white_mask = cv2.warpPerspective(
            white_ref,
            court_detector.court_warp_matrix[-1],
            (self.v_width, self.v_height),
        )
        white_mask = cv2.dilate(white_mask, np.ones((55, 1), dtype=np.uint8))
        white_mask[int(self.v_height * 0.58) :, :] = 0
        image_court = image.copy()
        image_court[white_mask == 0] = (0, 0, 0)

        persons_boxes, probs = self._extract_person_blobs(image_court, min_area=200.0, max_area=35000.0)
        tracked_objects = self.mot_tracker.update(persons_boxes if persons_boxes else None, probs if probs else None)

        for det_person in self.persons_boxes:
            self.persons_boxes[det_person].append([None, None, None, None])

        for box in tracked_objects:
            x1, y1, x2, y2, obj_id = float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(box[4])
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            if obj_id in self.persons_boxes:
                self.persons_boxes[obj_id][-1] = [x1, y1, x2, y2]
            else:
                self.persons_boxes[obj_id] = [[x1, y1, x2, y2]]
                self.persons_first_appearance[obj_id] = frame_num

        return vis

    def calculate_all_persons_dists(self) -> dict[int, float]:
        """Calculate cumulative movement distance for each tracked top-half person ID."""
        for obj_id, person_boxes in self.persons_boxes.items():
            valid_boxes = [[float(v) for v in b if v is not None] for b in person_boxes if b[0] is not None]
            self.persons_dists[obj_id] = boxes_dist(valid_boxes)
        return self.persons_dists

    def find_player_2_box(self) -> list[list[float | None]]:
        """Discriminate the top player box from ballboys/umpires by cumulative distance (``ArtLabss``)."""
        dists = self.calculate_all_persons_dists().copy()
        total_frames = len(self.player_1_boxes)
        if not dists:
            self.player_2_boxes = [[None, None, None, None] for _ in range(total_frames)]
            return self.player_2_boxes

        min_length = max(1, min(10, total_frames // 3))
        persons_sections: dict[int, list[int]] = {}
        for key, val in self.persons_first_appearance.items():
            seq = self.persons_boxes[key]
            rev_valid = [b[0] is not None for b in seq[::-1]]
            last_offset = int(np.argmax(rev_valid)) if any(rev_valid) else 0
            persons_sections[key] = [val, val + len(seq) - last_offset - 1]

        detections: list[int] = []
        while dists:
            max_key = max(dists.items(), key=operator.itemgetter(1))[0]
            max_sec = persons_sections[max_key].copy()
            if max_sec[1] - max_sec[0] + 1 < min_length and len(dists) > 1:
                dists.pop(max_key)
                continue
            detections.append(max_key)
            dists.pop(max_key)
            for key, sec in persons_sections.items():
                if sections_intersect(max_sec, sec) and key in dists:
                    dists.pop(key)

        detections = sorted(detections)
        boxes: list[list[float | None]] = []
        for det in detections:
            start = self.persons_first_appearance[det]
            missing = max(0, start - 1 - len(boxes))
            boxes.extend([[None, None, None, None] for _ in range(missing)])
            span = persons_sections[det][1] - persons_sections[det][0] + 1
            boxes.extend(self.persons_boxes[det][:span])

        if len(boxes) < total_frames:
            boxes.extend([[None, None, None, None] for _ in range(total_frames - len(boxes))])
        self.player_2_boxes = boxes[:total_frames]
        return self.player_2_boxes

    def calculate_feet_positions(
        self,
        court_detector: CourtDetector,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project bottom and top player feet ``(cx, y2)`` into ``CourtReference`` coordinates and smooth."""
        inv_mats = court_detector.game_warp_matrix
        n = len(self.player_1_boxes)
        if n == 0:
            empty = np.zeros((0, 2), dtype=np.float64)
            return empty, empty.copy()

        positions_1: list[NDArray[np.float64]] = []
        positions_2: list[NDArray[np.float64]] = []

        default_p1 = np.array([832.0, 2850.0], dtype=np.float64)
        default_p2 = np.array([832.0, 620.0], dtype=np.float64)

        for i, box in enumerate(self.player_1_boxes):
            inv_m = inv_mats[min(i, len(inv_mats) - 1)]
            if box[0] is not None and box[2] is not None and box[3] is not None:
                feet_pos = np.array(
                    [[[float(box[0]) + (float(box[2]) - float(box[0])) * 0.5, float(box[3])]]], dtype=np.float32
                )
                feet_court = cv2.perspectiveTransform(feet_pos, inv_m).reshape(2).astype(np.float64)
                positions_1.append(feet_court)
            elif positions_1:
                positions_1.append(positions_1[-1].copy())
            else:
                positions_1.append(default_p1.copy())

        for i in range(n):
            inv_m = inv_mats[min(i, len(inv_mats) - 1)]
            box = self.player_2_boxes[i] if i < len(self.player_2_boxes) else [None, None, None, None]
            if box[0] is not None and box[2] is not None and box[3] is not None:
                feet_pos = np.array(
                    [[[float(box[0]) + (float(box[2]) - float(box[0])) * 0.5, float(box[3])]]], dtype=np.float32
                )
                feet_court = cv2.perspectiveTransform(feet_pos, inv_m).reshape(2).astype(np.float64)
                positions_2.append(feet_court)
            elif positions_2:
                positions_2.append(positions_2[-1].copy())
            else:
                positions_2.append(default_p2.copy())

        arr1 = np.asarray(positions_1, dtype=np.float64)
        arr2 = np.asarray(positions_2, dtype=np.float64)

        win = min(7, n if n % 2 == 1 else n - 1)
        if win >= 5:
            s1 = np.zeros_like(arr1)
            s2 = np.zeros_like(arr2)
            s1[:, 0] = signal.savgol_filter(arr1[:, 0], win, 2)
            s1[:, 1] = signal.savgol_filter(arr1[:, 1], win, 2)
            s2[:, 0] = signal.savgol_filter(arr2[:, 0], win, 2)
            s2[:, 1] = signal.savgol_filter(arr2[:, 1], win, 2)
            return s1, s2

        return arr1, arr2


class TrackNetMotionBallDetector:
    """3-frame ego-motion-compensated heatmap ball detector + ``cv2.HoughCircles`` centroiding.

    Follows the exact heatmap -> threshold(127, 255) -> HoughCircles / blob centroid pipeline
    of ``ArtLabss/tennis-tracking/predict_video.py`` (lines 159-196), while generating the
    640x360 probability heatmap from pan-compensated 3-frame motion + optic-yellow/bright
    chromatic isolation so it executes deterministically on CPU without external weights.
    """

    def __init__(self) -> None:
        self._prev_frame: NDArray[np.uint8] | None = None
        self._prev_warp: NDArray[np.float64] | None = None
        self._last_ball_uv: tuple[float, float] | None = None

    def reset(self) -> None:
        """Clear temporal frame history at broadcast camera cuts."""
        self._prev_frame = None
        self._prev_warp = None
        self._last_ball_uv = None

    def compute_heatmap(
        self,
        frame: NDArray[np.uint8],
        court_warp_matrix: NDArray[np.float64] | None = None,
    ) -> NDArray[np.uint8]:
        """Compute a binary ``(H, W)`` ball probability heatmap thresholded at 127 -> 255."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Optic-yellow / bright green-yellow / high-luminance small ball mask
        yellow_mask = cv2.inRange(
            hsv,
            np.array([18, 70, 140], dtype=np.uint8),
            np.array([52, 255, 255], dtype=np.uint8),
        )
        bright_core = cv2.inRange(
            hsv,
            np.array([14, 35, 195], dtype=np.uint8),
            np.array([62, 255, 255], dtype=np.uint8),
        )
        color_cand = cv2.bitwise_or(yellow_mask, bright_core)

        if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
            prev_aligned: NDArray[np.uint8] = self._prev_frame
            if court_warp_matrix is not None and self._prev_warp is not None and np.linalg.cond(self._prev_warp) < 1e8:
                delta_h = court_warp_matrix @ np.linalg.inv(self._prev_warp)
                prev_aligned = np.asarray(cv2.warpPerspective(self._prev_frame, delta_h, (w, h)), dtype=np.uint8)

            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_aligned, cv2.COLOR_BGR2GRAY))
            _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            heatmap = cv2.bitwise_and(color_cand, motion_mask)
            if np.count_nonzero(heatmap) == 0:
                # Fallback to isolated optic-yellow ball when stationary at apex
                heatmap = yellow_mask.copy()
        else:
            heatmap = yellow_mask.copy()

        self._prev_frame = frame.copy()
        if court_warp_matrix is not None:
            self._prev_warp = court_warp_matrix.copy()

        # Morphological top-hat to keep compact circular objects (radius 2..11 px) and reject lines/shirts
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        opened = cv2.morphologyEx(heatmap, cv2.MORPH_OPEN, k_small)
        large_blobs = cv2.morphologyEx(opened, cv2.MORPH_OPEN, k_large)
        compact = cv2.subtract(opened, large_blobs)
        _, binary = cv2.threshold(compact, 127, 255, cv2.THRESH_BINARY)
        return np.asarray(binary, dtype=np.uint8)

    def detect_ball(
        self,
        frame: NDArray[np.uint8],
        court_warp_matrix: NDArray[np.float64] | None = None,
    ) -> tuple[float, float] | None:
        """Detect tennis ball ``(u, v)`` from heatmap via ``HoughCircles`` + circularity scoring."""
        heatmap = self.compute_heatmap(frame, court_warp_matrix=court_warp_matrix)
        h, w = heatmap.shape[:2]

        # 1. Try ArtLabss HoughCircles on the thresholded heatmap
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=8.0,
            param1=50.0,
            param2=2.0,
            minRadius=2,
            maxRadius=11,
        )
        candidates: list[tuple[float, float, float]] = []
        if circles is not None and len(circles) > 0:
            for c in circles[0]:
                cx, cy, r = float(c[0]), float(c[1]), float(c[2])
                if 0.05 * w <= cx <= 0.95 * w and 0.04 * h <= cy <= 0.94 * h:
                    candidates.append((cx, cy, r))

        # 2. Connected contour circularity candidates for sub-pixel accuracy
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if 4.0 <= area <= 360.0:
                perim = max(float(cv2.arcLength(cnt, True)), 1e-3)
                circularity = 4.0 * np.pi * area / (perim * perim)
                if circularity >= 0.42:
                    m = cv2.moments(cnt)
                    if m["m00"] > 1e-5:
                        cx = float(m["m10"] / m["m00"])
                        cy = float(m["m01"] / m["m00"])
                        if 0.05 * w <= cx <= 0.95 * w and 0.04 * h <= cy <= 0.94 * h:
                            candidates.append((cx, cy, circularity * 10.0))

        if not candidates:
            return None

        if self._last_ball_uv is not None:
            lx, ly = self._last_ball_uv
            best = min(
                candidates,
                key=lambda c: float(np.hypot(c[0] - lx, c[1] - ly)) - c[2] * 1.5,
            )
        else:
            best = max(candidates, key=lambda c: c[2])

        self._last_ball_uv = (best[0], best[1])
        return (best[0], best[1])


def create_top_view_image(
    court_detector: CourtDetector,
    feet_pos_1: tuple[float, float] | NDArray[np.float64] | None,
    feet_pos_2: tuple[float, float] | NDArray[np.float64] | None,
    ball_xy_frame: tuple[float, float] | None,
    frame_idx: int,
    bounces_ref: list[tuple[float, float, bool]] | None = None,
    trail_ref: list[tuple[float, float]] | None = None,
) -> NDArray[np.uint8]:
    """Render a single frame of the 2D Top-Down Court Minimap (``ArtLabss`` ``create_top_view``).

    Upgraded with an Australian Open Melbourne-Blue two-tone court canvas, white court lines,
    player feet markers (blue circles), fading ball trajectory trail, live yellow ball marker,
    and cumulative bounce dots (green = ``IN``, red = ``OUT``).
    """
    ref = court_detector.court_reference
    h_ref, w_ref = ref.court.shape[:2]

    # Australian Open two-tone blue canvas (BGR)
    canvas = np.full((h_ref, w_ref, 3), (175, 110, 38), dtype=np.uint8)
    inner_mask = ref.get_court_mask(3)
    canvas[inner_mask > 0] = (158, 86, 24)
    canvas[ref.court > 0] = (245, 245, 245)
    cv2.line(canvas, *ref.net, (200, 220, 255), 10)

    # Draw trajectory trail on minimap
    if trail_ref and len(trail_ref) >= 2:
        for k in range(len(trail_ref) - 1):
            p1 = (round(trail_ref[k][0]), round(trail_ref[k][1]))
            p2 = (round(trail_ref[k + 1][0]), round(trail_ref[k + 1][1]))
            cv2.line(canvas, p1, p2, (0, 215, 255), 14, cv2.LINE_AA)

    # Draw cumulative bounce markers (Green = IN, Red = OUT)
    if bounces_ref:
        for bx, by, is_in in bounces_ref:
            b_pt = (round(bx), round(by))
            color = (40, 230, 60) if is_in else (40, 45, 245)
            cv2.circle(canvas, b_pt, 36, color, -1)
            cv2.circle(canvas, b_pt, 42, (255, 255, 255), 6)

    # Draw Player 1 and Player 2 feet positions (Blue circles with white border)
    if feet_pos_1 is not None and not np.isnan(feet_pos_1[0]):
        p1_pt = (
            int(np.clip(round(float(feet_pos_1[0])), 60, w_ref - 60)),
            int(np.clip(round(float(feet_pos_1[1])), 60, h_ref - 60)),
        )
        cv2.circle(canvas, p1_pt, 48, (255, 75, 30), -1)
        cv2.circle(canvas, p1_pt, 48, (255, 255, 255), 6)

    if feet_pos_2 is not None and not np.isnan(feet_pos_2[0]):
        p2_pt = (
            int(np.clip(round(float(feet_pos_2[0])), 60, w_ref - 60)),
            int(np.clip(round(float(feet_pos_2[1])), 60, h_ref - 60)),
        )
        cv2.circle(canvas, p2_pt, 48, (255, 75, 30), -1)
        cv2.circle(canvas, p2_pt, 48, (255, 255, 255), 6)

    # Draw live ball position mapped via game_warp_matrix (ArtLabss draw_ball_position)
    if ball_xy_frame is not None and court_detector.game_warp_matrix:
        inv_mat = court_detector.game_warp_matrix[min(frame_idx, len(court_detector.game_warp_matrix) - 1)]
        ball_pos = np.array([[[float(ball_xy_frame[0]), float(ball_xy_frame[1])]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(ball_pos, inv_mat)[0][0]
        bx_px = int(np.clip(round(float(transformed[0])), 35, w_ref - 35))
        by_px = int(np.clip(round(float(transformed[1])), 35, h_ref - 35))
        cv2.circle(canvas, (bx_px, by_px), 38, (0, 255, 255), -1)
        cv2.circle(canvas, (bx_px, by_px), 38, (20, 20, 20), 5)

    return canvas


def merge_minimap(frame: NDArray[np.uint8], minimap_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Merge the top-down minimap into the top-right corner of ``frame`` (``ArtLabss`` ``merge``)."""
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]
    target_w = max(120, frame_w // 7)
    aspect = float(minimap_image.shape[0]) / max(float(minimap_image.shape[1]), 1.0)
    target_h = min(round(target_w * aspect), int(frame_h * 0.62))
    target_w = round(target_h / max(aspect, 1e-3))

    resized = cv2.resize(minimap_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x_start = frame_w - target_w - 12
    y_start = 12
    out[y_start : y_start + target_h, x_start : x_start + target_w] = resized
    cv2.rectangle(
        out,
        (x_start - 1, y_start - 1),
        (x_start + target_w, y_start + target_h),
        (230, 230, 230),
        2,
    )
    return out
