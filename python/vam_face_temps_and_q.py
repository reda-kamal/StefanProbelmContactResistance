"""Evaluate VAM face temperatures and resulting flux history."""
from __future__ import annotations
import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


def vam_face_temps_and_q(params: Dict[str, float], which: str,
                         times: Iterable[float], R_c) -> Tuple[List[float], List[float], List[float]]:
    alpha_w = params['alpha_w']
    alpha_s = params['alpha_s']
    lam = params['lam']
    Tf = params['Tf']
    Ti = params['Ti']
    Tw_inf = params['Tw_inf']
    if which == 'early':
        t0 = params['t0_e']
        S0 = params['S0_e']
        E0 = params['E0_e']
    else:
        t0 = params['t0_l']
        S0 = params['S0_l']
        E0 = params['E0_l']
    rc_eval = _prepare_rc_sampler(R_c)

    out_Tw: List[float] = []
    out_Ts: List[float] = []
    out_q: List[float] = []
    for t in times:
        den_w = 2.0 * math.sqrt(alpha_w * (t + t0))
        den_s = 2.0 * math.sqrt(alpha_s * (t + t0))
        Tw0m = Ti + (Ti - Tw_inf) * math.erf(-E0 / den_w)
        Ts0p = Ti + (Tf - Ti) * math.erf(S0 / den_s) / math.erf(lam)
        rc_val = rc_eval(float(t))
        if abs(rc_val) < 1e-12:
            q_rc = float('inf')
        else:
            q_rc = (Tw0m - Ts0p) / rc_val
        out_Tw.append(Tw0m)
        out_Ts.append(Ts0p)
        out_q.append(q_rc)
    return out_Tw, out_Ts, out_q


def vam_contact_resistance(params: Dict[str, float], which: str,
                           times: Iterable[float]) -> List[float]:
    """Return the VAM effective contact resistance history for a calibration."""

    alpha_w = params['alpha_w']
    alpha_s = params['alpha_s']
    k_w = params['k_w']
    k_s = params['k_s']

    if which == 'early':
        t0 = params['t0_e']
        S0 = params['S0_e']
        E0 = params['E0_e']
    else:
        t0 = params['t0_l']
        S0 = params['S0_l']
        E0 = params['E0_l']

    vals: List[float] = []
    for t in times:
        tp = max(t + t0, 1e-16)
        phi_s = S0 / (2.0 * math.sqrt(alpha_s * tp))
        phi_w = E0 / (2.0 * math.sqrt(alpha_w * tp))
        hs = (k_s / S0) * _gfun(phi_s)
        hw = (k_w / E0) * _gfun(phi_w)
        he = _combine(hw, hs)
        if he <= 0.0:
            vals.append(float('inf'))
        else:
            vals.append(1.0 / he)
    return vals


def _gfun(chi: float) -> float:
    if abs(chi) < 1e-12:
        return 2.0 / math.sqrt(math.pi)
    return (2.0 * chi * math.exp(-chi * chi)) / (math.sqrt(math.pi) * math.erf(chi))


def _combine(hw: float, hs: float) -> float:
    if hw <= 0.0 or hs <= 0.0:
        return 0.0
    return 1.0 / (1.0 / hw + 1.0 / hs)


def _prepare_rc_sampler(R_c) -> Callable[[float], float]:
    if callable(R_c):
        return lambda t: float(R_c(t))

    if isinstance(R_c, dict):
        times = None
        values = None
        for key in ('times', 'time', 't'):
            if key in R_c:
                seq = R_c[key]
                times = [float(v) for v in seq]
                break
        for key in ('values', 'val', 'R_c', 'rc'):
            if key in R_c:
                seq = R_c[key]
                values = [float(v) for v in seq]
                break
        if times is None or values is None:
            raise ValueError('Invalid R_c specification for VAM sampling')
        return lambda t: _interp(times, values, t)

    if isinstance(R_c, Sequence):
        if len(R_c) == 2 and isinstance(R_c[0], Sequence):
            times = [float(v) for v in R_c[0]]
            values = [float(v) for v in R_c[1]]
            return lambda t: _interp(times, values, t)

    rc_const = float(R_c)
    return lambda _t: rc_const


def _interp(times: Sequence[float], values: Sequence[float], t: float) -> float:
    if not times:
        raise ValueError('Empty time series for R_c sampler')
    if len(times) != len(values):
        raise ValueError('Mismatched time/value lengths for R_c sampler')
    if len(times) == 1:
        return float(values[0])
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    lo = 0
    hi = len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    t0 = times[lo]
    t1 = times[lo + 1]
    v0 = values[lo]
    v1 = values[lo + 1]
    if t1 == t0:
        return float(v0)
    w = (t - t0) / (t1 - t0)
    return float(v0 + w * (v1 - v0))
