"""Evaluate VAM face temperatures and resulting flux history."""
from __future__ import annotations
import math
from typing import Dict, Iterable, List, Tuple


def vam_face_temps_and_q(params: Dict[str, float], which: str,
                         times: Iterable[float], R_c: float) -> Tuple[List[float], List[float], List[float]]:
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
    out_Tw: List[float] = []
    out_Ts: List[float] = []
    out_q: List[float] = []
    for t in times:
        den_w = 2.0 * math.sqrt(alpha_w * (t + t0))
        den_s = 2.0 * math.sqrt(alpha_s * (t + t0))
        Tw0m = Ti + (Ti - Tw_inf) * math.erf(-E0 / den_w)
        Ts0p = Ti + (Tf - Ti) * math.erf(S0 / den_s) / math.erf(lam)
        q_rc = (Tw0m - Ts0p) / R_c
        out_Tw.append(Tw0m)
        out_Ts.append(Ts0p)
        out_q.append(q_rc)
    return out_Tw, out_Ts, out_q
