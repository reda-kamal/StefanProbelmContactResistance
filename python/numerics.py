"""Utility numerical routines required by the Stefan solvers."""
from __future__ import annotations
import math
from typing import Callable, Iterable, List, Sequence, Tuple

Vector = Tuple[float, float]


def norm_inf(vals: Sequence[float]) -> float:
    return max(abs(v) for v in vals)


def fsolve2(func: Callable[[Vector], Vector], x0: Vector, tol: float = 1e-10,
            max_iter: int = 50, fd_eps: float = 1e-6) -> Vector:
    """Solve a 2x2 nonlinear system with a Newton iteration and finite differences."""
    x = [x0[0], x0[1]]
    for _ in range(max_iter):
        f0 = func((x[0], x[1]))
        if norm_inf(f0) < tol:
            return (x[0], x[1])
        # Finite-difference Jacobian
        J = [[0.0, 0.0], [0.0, 0.0]]
        for j in range(2):
            x_step = x[:]
            step = fd_eps * (abs(x[j]) + 1.0)
            x_step[j] += step
            f_step = func((x_step[0], x_step[1]))
            for i in range(2):
                J[i][j] = (f_step[i] - f0[i]) / step
        # Solve J * dx = -f0 using 2x2 inverse
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        if abs(det) < 1e-18:
            raise RuntimeError("Singular Jacobian in fsolve2")
        invJ = [[ J[1][1] / det, -J[0][1] / det],
                [-J[1][0] / det,  J[0][0] / det]]
        dx0 = -(invJ[0][0] * f0[0] + invJ[0][1] * f0[1])
        dx1 = -(invJ[1][0] * f0[0] + invJ[1][1] * f0[1])
        x[0] += dx0
        x[1] += dx1
        if math.isnan(x[0]) or math.isnan(x[1]):
            raise RuntimeError("NaN encountered in fsolve2")
        if max(abs(dx0), abs(dx1)) < tol:
            return (x[0], x[1])
    raise RuntimeError("fsolve2 failed to converge")


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
