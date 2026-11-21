"""Pure-NumPy / SciPy implementation of SORT (Simple, Online and Realtime Tracker) and KalmanFilter.

Adapted from ``ArtLabss/tennis-tracking/sort.py`` and ``aryanhood/tennisball-tracking/ball_tracker.py``,
upgraded to:
1. Eliminate the unmaintained ``filterpy`` dependency via a self-contained, strictly-typed
   linear ``KalmanFilter`` class with identical ``F, H, R, Q, P, x`` matrix semantics.
2. Eliminate global ``KalmanBoxTracker.count`` state leakage across video runs.
3. Include ``BallTracker`` (2D position + velocity Kalman filter with coasting across missed frames).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """Linear discrete-time Kalman Filter compatible with ``filterpy.kalman.KalmanFilter``."""

    def __init__(self, dim_x: int, dim_z: int) -> None:
        self.dim_x: int = dim_x
        self.dim_z: int = dim_z
        self.x: NDArray[np.float64] = np.zeros((dim_x, 1), dtype=np.float64)
        self.P: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)
        self.F: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)
        self.H: NDArray[np.float64] = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.R: NDArray[np.float64] = np.eye(dim_z, dtype=np.float64)
        self.Q: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

    def predict(self) -> NDArray[np.float64]:
        """Advance state ``x = F x`` and covariance ``P = F P F^T + Q``."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update state ``x`` and covariance ``P`` with measurement ``z (dim_z, 1)``."""
        z_col = np.asarray(z, dtype=np.float64).reshape(self.dim_z, 1)
        y = z_col - (self.H @ self.x)
        s_mat = self.H @ self.P @ self.H.T + self.R
        k_gain = self.P @ self.H.T @ np.linalg.inv(s_mat)
        self.x = (self.x + (k_gain @ y)).astype(np.float64)
        eye = np.eye(self.dim_x, dtype=np.float64)
        self.P = ((eye - (k_gain @ self.H)) @ self.P).astype(np.float64)
        return self.x


def linear_assignment(cost_matrix: NDArray[np.float64]) -> NDArray[np.int32]:
    """Solve linear assignment problem using Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.column_stack([row_ind, col_ind]).astype(np.int32)


def iou_batch(bb_test: NDArray[np.float64], bb_gt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute pairwise IoU between two sets of bounding boxes ``[x1, y1, x2, y2]`` (``ArtLabss`` ``sort.py``)."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float64)

    gt_exp = np.expand_dims(bb_gt, 0)
    test_exp = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(test_exp[..., 0], gt_exp[..., 0])
    yy1 = np.maximum(test_exp[..., 1], gt_exp[..., 1])
    xx2 = np.minimum(test_exp[..., 2], gt_exp[..., 2])
    yy2 = np.minimum(test_exp[..., 3], gt_exp[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    area_test = (test_exp[..., 2] - test_exp[..., 0]) * (test_exp[..., 3] - test_exp[..., 1])
    area_gt = (gt_exp[..., 2] - gt_exp[..., 0]) * (gt_exp[..., 3] - gt_exp[..., 1])
    denom = np.maximum(area_test + area_gt - wh, 1e-9)
    return np.asarray(wh / denom, dtype=np.float64)


def convert_bbox_to_z(bbox: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert ``[x1, y1, x2, y2]`` to ``[cx, cy, scale_area, aspect_ratio]^T`` (``4x1``)."""
    w = max(float(bbox[2] - bbox[0]), 1.0)
    h = max(float(bbox[3] - bbox[1]), 1.0)
    x = float(bbox[0]) + w / 2.0
    y = float(bbox[1]) + h / 2.0
    s = w * h
    r = w / h
    return np.array([x, y, s, r], dtype=np.float64).reshape((4, 1))


def convert_x_to_bbox(x: NDArray[np.float64], score: float | None = None) -> NDArray[np.float64]:
    """Convert centre state ``[cx, cy, s, r]`` back to ``[x1, y1, x2, y2]``."""
    x_flat = np.asarray(x, dtype=np.float64).flatten()
    prod = max(float(x_flat[2] * x_flat[3]), 1.0)
    w = float(np.sqrt(prod))
    h = max(float(x_flat[2]) / max(w, 1e-6), 1.0)
    if score is None:
        return np.array(
            [x_flat[0] - w / 2.0, x_flat[1] - h / 2.0, x_flat[0] + w / 2.0, x_flat[1] + h / 2.0],
            dtype=np.float64,
        ).reshape((1, 4))
    return np.array(
        [x_flat[0] - w / 2.0, x_flat[1] - h / 2.0, x_flat[0] + w / 2.0, x_flat[1] + h / 2.0, score],
        dtype=np.float64,
    ).reshape((1, 5))


class KalmanBoxTracker:
    """Internal state of an individual tracked bounding box object (``ArtLabss`` ``sort.py``)."""

    def __init__(self, bbox: NDArray[np.float64], tracker_id: int = 0) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update: int = 0
        self.id: int = tracker_id
        self.history: list[NDArray[np.float64]] = []
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

    def update(self, bbox: NDArray[np.float64]) -> None:
        """Update state vector with observed bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> NDArray[np.float64]:
        """Advance state vector and return predicted bounding box estimate ``(1, 4)``."""
        if float(self.kf.x[6, 0] + self.kf.x[2, 0]) <= 0.0:
            self.kf.x[6, 0] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> NDArray[np.float64]:
        """Return current bounding box estimate ``(1, 4)``."""
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: NDArray[np.float64],
    trackers: NDArray[np.float64],
    iou_threshold: float = 0.3,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Assign detections to tracked boxes via IoU + Hungarian matching."""
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(len(detections), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if int(np.max(a.sum(1))) == 1 and int(np.max(a.sum(0))) == 1:
            matched_indices = np.stack(np.where(a), axis=1).astype(np.int32)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=np.int32)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches: list[NDArray[np.int32]] = []
    for m in matched_indices:
        if float(iou_matrix[m[0], m[1]]) < iou_threshold:
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    matches_arr = np.concatenate(matches, axis=0).astype(np.int32) if matches else np.empty((0, 2), dtype=np.int32)
    return (
        matches_arr,
        np.array(unmatched_detections, dtype=np.int32),
        np.array(unmatched_trackers, dtype=np.int32),
    )


class Sort:
    """Multi-object bounding box tracker from ``ArtLabss/tennis-tracking/sort.py``."""

    def __init__(self, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.05) -> None:
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count: int = 0
        self._next_id: int = 0

    def update(
        self,
        dets: list[NDArray[np.float64]] | NDArray[np.float64] | None,
        scores: list[float] | NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Update tracker with detections ``[[x1, y1, x2, y2], ...]`` and return ``[[x1, y1, x2, y2, id], ...]``."""
        if dets is None or len(dets) == 0:
            dets_arr = np.empty((0, 5), dtype=np.float64)
        else:
            raw = np.asarray(dets, dtype=np.float64)
            if raw.shape[1] == 4 and scores is not None:
                sc = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
                dets_arr = np.column_stack([raw, sc])
            elif raw.shape[1] == 4:
                dets_arr = np.column_stack([raw, np.ones((len(raw), 1), dtype=np.float64)])
            else:
                dets_arr = raw

        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5), dtype=np.float64)
        to_del: list[int] = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0.0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
        if to_del:
            trks = np.delete(trks, to_del, axis=0)

        matched, unmatched_dets, _ = associate_detections_to_trackers(dets_arr, trks, self.iou_threshold)
        for m in matched:
            self.trackers[int(m[1])].update(dets_arr[int(m[0]), :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_arr[int(i), :], tracker_id=self._next_id)
            self._next_id += 1
            self.trackers.append(trk)

        ret: list[NDArray[np.float64]] = []
        idx = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate([d, [float(trk.id + 1)]]).reshape(1, 5))
            idx -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(idx)

        if ret:
            return np.concatenate(ret, axis=0).astype(np.float64)
        return np.empty((0, 5), dtype=np.float64)


class BallKalmanTracker:
    """Upgraded 2D Ball Kalman Filter from ``aryanhood/tennisball-tracking/ball_tracker.py``.

    Tracks ``[u, v, vu, vv]^T`` with Mahalanobis/Euclidean outlier gating, coasting across
    up to ``max_coast_frames`` missed detections, and automatic reset on camera cuts.
    """

    def __init__(self, max_coast_frames: int = 8, gate_distance_px: float = 140.0) -> None:
        self.max_coast_frames: int = max_coast_frames
        self.gate_distance_px: float = gate_distance_px
        self.kf: KalmanFilter = self._init_kf()
        self.frames_without_detection: int = max_coast_frames + 1
        self.initialized: bool = False

    @staticmethod
    def _init_kf() -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        kf.R = np.eye(2, dtype=np.float64) * 6.0
        kf.Q = np.eye(4, dtype=np.float64) * 0.25
        kf.P = np.eye(4, dtype=np.float64) * 250.0
        return kf

    def reset(self) -> None:
        """Reset filter state (called on broadcast camera cuts)."""
        self.kf = self._init_kf()
        self.frames_without_detection = self.max_coast_frames + 1
        self.initialized = False

    def step(self, detection: tuple[float, float] | None) -> tuple[float, float] | None:
        """Advance one frame with optional ``(u, v)`` detection and return filtered ``(u, v)``."""
        if not self.initialized:
            if detection is None:
                return None
            self.kf.x = np.array([[detection[0]], [detection[1]], [0.0], [0.0]], dtype=np.float64)
            self.initialized = True
            self.frames_without_detection = 0
            return (float(detection[0]), float(detection[1]))

        pred = self.kf.predict()
        pred_u, pred_v = float(pred[0, 0]), float(pred[1, 0])

        if detection is not None:
            dist = float(np.hypot(detection[0] - pred_u, detection[1] - pred_v))
            allowed = self.gate_distance_px * (1.0 + 0.35 * self.frames_without_detection)
            if dist <= allowed:
                self.kf.update(np.array([[detection[0]], [detection[1]]], dtype=np.float64))
                self.frames_without_detection = 0
            elif self.frames_without_detection >= 3:
                # Re-anchor on new trajectory segment after several missed frames
                self.kf.x = np.array([[detection[0]], [detection[1]], [0.0], [0.0]], dtype=np.float64)
                self.frames_without_detection = 0
            else:
                self.frames_without_detection += 1
        else:
            self.frames_without_detection += 1

        if self.frames_without_detection > self.max_coast_frames:
            return None

        return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))
