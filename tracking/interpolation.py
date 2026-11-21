"""Outlier removal and trajectory interpolation helpers from ``ArtLabss/tennis-tracking/detection.py``.

Implements:
- ``nan_helper``: logical NaN index mask and non-zero index callable for ``np.interp``.
- ``diff_xy``: consecutive frame-to-frame absolute differences ``(|dx|, |dy|)``.
- ``remove_outliers``: 3-pass spike filter removing points where ``|dx| > threshold`` and ``|dy| > threshold``.
- ``interpolation``: linear interpolation across missing (``None`` / ``NaN``) ball detections.
- ``smooth_trajectory_savgol``: Savitzky-Golay polynomial smoothing for player feet and ball curves.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def nan_helper(y: NDArray[np.float64]) -> tuple[NDArray[np.bool_], Callable[[NDArray[np.bool_]], NDArray[np.intp]]]:
    """Return ``(isnan_mask, nonzero_index_fn)`` for linear NaN interpolation (``ArtLabss`` ``nan_helper``)."""
    return np.isnan(y), lambda z: z.nonzero()[0]


def diff_xy(
    coords: list[tuple[float, float] | list[float] | None],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute consecutive frame-to-frame ``(|dx|, |dy|)`` arrays (``ArtLabss`` ``diff_xy``)."""
    xx_list: list[float] = []
    yy_list: list[float] = []
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        if p1 is not None and p2 is not None:
            xx_list.append(abs(float(p2[0]) - float(p1[0])))
            yy_list.append(abs(float(p2[1]) - float(p1[1])))
        else:
            xx_list.append(float("nan"))
            yy_list.append(float("nan"))
    return np.array(xx_list, dtype=np.float64), np.array(yy_list, dtype=np.float64)


def remove_outliers(
    x_diff: NDArray[np.float64],
    y_diff: NDArray[np.float64],
    coords: list[tuple[float, float] | list[float] | None],
    max_step_px: float = 50.0,
) -> list[tuple[float, float] | list[float] | None]:
    """Remove isolated spike outliers in ``coords`` in-place and return ``coords``."""
    with np.errstate(invalid="ignore"):
        ids_x = set(np.where(x_diff > max_step_px)[0].tolist())
        ids_y = set(np.where(y_diff > max_step_px)[0].tolist())
    ids = sorted(ids_x & ids_y)

    for idx in ids:
        if idx <= 0 or idx + 1 >= len(coords):
            continue
        left = coords[idx - 1]
        middle = coords[idx]
        right = coords[idx + 1]

        candidates: list[tuple[float, int]] = []
        if left is not None:
            candidates.append((float(left[0]) + float(left[1]), idx - 1))
        if middle is not None:
            candidates.append((float(middle[0]) + float(middle[1]), idx))
        if right is not None:
            candidates.append((float(right[0]) + float(right[1]), idx + 1))

        if candidates:
            _, max_pos = max(candidates, key=lambda item: item[0])
            coords[max_pos] = None

    return coords


def interpolation(
    coords: list[tuple[float, float] | list[float] | None],
) -> list[tuple[float, float]]:
    """Interpolate ``None`` gaps in ``coords`` via 1D linear interpolation (``ArtLabss`` ``interpolation``)."""
    if not coords:
        return []

    xxx = np.array([float(c[0]) if c is not None else np.nan for c in coords], dtype=np.float64)
    yyy = np.array([float(c[1]) if c is not None else np.nan for c in coords], dtype=np.float64)

    valid_x = ~np.isnan(xxx)
    valid_y = ~np.isnan(yyy)
    if not np.any(valid_x) or not np.any(valid_y):
        return [(0.0, 0.0) for _ in coords]

    nons, yy = nan_helper(xxx)
    if np.any(nons):
        xxx[nons] = np.interp(yy(nons), yy(~nons), xxx[~nons])

    nans, xx = nan_helper(yyy)
    if np.any(nans):
        yyy[nans] = np.interp(xx(nans), xx(~nans), yyy[~nans])

    return [(float(u), float(v)) for u, v in zip(xxx, yyy, strict=True)]


def smooth_trajectory_savgol(
    positions: NDArray[np.float64],
    window_length: int = 7,
    polyorder: int = 2,
) -> NDArray[np.float64]:
    """Smooth a 2D trajectory ``(N, 2)`` using Savitzky-Golay filter (``ArtLabss`` ``calculate_feet_positions``)."""
    arr = np.asarray(positions, dtype=np.float64)
    if len(arr) < window_length or window_length <= polyorder:
        return arr.copy()
    smoothed = np.zeros_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        smoothed[:, col] = signal.savgol_filter(arr[:, col], window_length, polyorder)
    return smoothed
