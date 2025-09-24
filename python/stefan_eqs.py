"""Nonlinear residuals for the three-domain Stefan problem with contact resistance."""
from __future__ import annotations
import math
from typing import Tuple


def erfcx(z: float) -> float:
    """Scaled complementary error function."""
    return math.exp(z * z) * math.erfc(z)


def stefan_eqs(x: Tuple[float, float], aw: float, as_: float, al: float,
               kw: float, ks: float, kl: float, L: float, rho_s: float,
               Tw_inf: float, Tl_inf: float, Tf: float) -> Tuple[float, float]:
    """Return the residual vector for lambda and interface temperature."""
    lam, Ti = x
    mu = lam * math.sqrt(as_ / al)

    erf_lam = math.erf(lam)
    if abs(erf_lam) < 1e-12:
        erf_lam = math.copysign(1e-12, erf_lam if erf_lam != 0.0 else 1.0)

    F1 = kw * math.sqrt(as_) * (Ti - Tw_inf) - ks * math.sqrt(aw) / erf_lam * (Tf - Ti)
    termL = kl * (Tf - Tl_inf) / (lam * math.sqrt(math.pi)) * (-mu / erfcx(mu))
    termS = ks * (Tf - Ti) / math.sqrt(math.pi) * math.exp(-lam * lam) / erf_lam
    F2 = rho_s * L * as_ * lam + termL - termS
    return (F1, F2)
