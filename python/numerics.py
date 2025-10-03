"""Utility numerical routines required by the Stefan solvers."""
from __future__ import annotations
from typing import Iterable, List, Sequence


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def interp1(x: Sequence[float], y: Sequence[float], x_new: float) -> float:
    if x_new <= x[0]:
        return y[0]
    if x_new >= x[-1]:
        return y[-1]
    # Binary search
    lo, hi = 0, len(x) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= x_new:
            lo = mid
        else:
            hi = mid
    t = (x_new - x[lo]) / (x[lo+1] - x[lo])
    return y[lo] * (1 - t) + y[lo+1] * t


def moving_average(vals: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(vals)
    n = len(vals)
    half = window // 2
    smoothed = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed.append(sum(vals[start:end]) / (end - start))
    return smoothed


def cumulative_sum(vals: Iterable[float]) -> List[float]:
    total = 0.0
    out: List[float] = []
    for v in vals:
        total += v
        out.append(total)
    return out
