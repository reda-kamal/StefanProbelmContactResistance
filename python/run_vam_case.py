"""Top-level driver mirroring the MATLAB run_vam_case."""
from __future__ import annotations
import math
from typing import Dict, List

try:
    from .numerics import fsolve2
    from .stefan_eqs import stefan_eqs
    from .options import get_opt
    from .explicit_stefan_snapshot import explicit_stefan_snapshot
except ImportError:  # pragma: no cover - allow running as a loose script
    from numerics import fsolve2  # type: ignore
    from stefan_eqs import stefan_eqs  # type: ignore
    from options import get_opt  # type: ignore
    from explicit_stefan_snapshot import explicit_stefan_snapshot  # type: ignore


class CaseResult(dict):
    pass


def run_vam_case(label: str, k_w: float, rho_w: float, c_w: float,
                 M: Dict[str, float], R_c: float, t_phys: float,
                 opts: Dict[str, object] | None = None) -> CaseResult:
    if opts is None:
        opts = {}

    profile_pts_per_seg = get_opt(opts, 'profile_pts_per_seg', 400)
    profile_extent_factor = get_opt(opts, 'profile_extent_factor', 5)
    explicit_opts = get_opt(opts, 'explicit', {})

    k_s = M['k_s']; rho_s = M['rho_s']; c_s = M['c_s']
    k_l = M['k_l']; rho_l = M['rho_l']; c_l = M['c_l']
    L = M['L']; Tf = M['Tf']; Tw_inf = M['Tw_inf']; Tl_inf = M['Tl_inf']

    alpha_w = k_w/(rho_w*c_w)
    alpha_s = k_s/(rho_s*c_s)
    alpha_l = k_l/(rho_l*c_l)

    def fun(x):
        return stefan_eqs(x, alpha_w, alpha_s, alpha_l,
                          k_w, k_s, k_l, L, rho_s, Tw_inf, Tl_inf, Tf)

    lam, Ti = fsolve2(fun, (0.1, 0.5 * (Tf + Tw_inf)))
    mu = lam * math.sqrt(alpha_s / alpha_l)

    h_c = 1.0 / R_c
    hcw = h_c * (Tf - Tw_inf) / max(Ti - Tw_inf, 1e-12)
    hcs = h_c * (Tf - Tw_inf) / max(Tf - Ti, 1e-12)

    S0_e = 2*lam*k_s/(hcs*math.sqrt(math.pi))*math.exp(-lam*lam)/math.erf(lam)
    log_arg = (2*lam*k_w*math.sqrt(alpha_s)) / (hcw*math.sqrt(math.pi*alpha_w)*S0_e)
    log_arg = max(log_arg, 1.0)
    E0_e = (S0_e/lam)*math.sqrt(alpha_w/alpha_s)*math.sqrt(math.log(log_arg))
    t0_e = S0_e**2 /(4*alpha_s*lam*lam)

    S0_l = k_s/hcs
    E0_l = k_w/hcw
    t0_l = S0_l**2 /(4*alpha_s*lam*lam)

    tpe = t_phys + t0_e
    tpl = t_phys + t0_l
    Se = 2*lam*math.sqrt(alpha_s*tpe) - S0_e
    Sl = 2*lam*math.sqrt(alpha_s*tpl) - S0_l

    Lw_e = profile_extent_factor * math.sqrt(alpha_w*tpe)
    Ll_e = profile_extent_factor * math.sqrt(alpha_l*tpe)
    Lw_l = profile_extent_factor * math.sqrt(alpha_w*tpl)
    Ll_l = profile_extent_factor * math.sqrt(alpha_l*tpl)
    x_min = -max(Lw_e, Lw_l)
    x_max = max(Se, Sl) + max(Ll_e, Ll_l)
    knots = sorted([x_min, 0.0, Se, Sl, x_max])
    pts_per_seg = profile_pts_per_seg
    x: List[float] = []
    for j in range(len(knots) - 1):
        seg = linspace(knots[j], knots[j+1], pts_per_seg)
        if j < len(knots) - 2:
            seg = seg[:-1]
        x.extend(seg)

    def erf_safe(z: float) -> float:
        return math.erf(z)

    den_w_e = 2*math.sqrt(alpha_w*tpe)
    den_s_e = 2*math.sqrt(alpha_s*tpe)
    den_l_e = 2*math.sqrt(alpha_l*tpe)
    Te = []
    erf_lam = erf_safe(lam)
    erf_mu = erf_safe(mu)
    for xi in x:
        if xi <= 0:
            Te.append(Ti + (Ti - Tw_inf)*erf_safe((xi - E0_e)/den_w_e))
        elif xi <= Se:
            Te.append(Ti + (Tf - Ti)*erf_safe((xi + S0_e)/den_s_e)/erf_lam)
        else:
            Te.append(Tl_inf + (Tf - Tl_inf)*(erf_safe((xi + S0_e)/den_l_e) - 1)/(erf_mu - 1))

    den_w_l = 2*math.sqrt(alpha_w*tpl)
    den_s_l = 2*math.sqrt(alpha_s*tpl)
    den_l_l = 2*math.sqrt(alpha_l*tpl)
    Tl = []
    for xi in x:
        if xi <= 0:
            Tl.append(Ti + (Ti - Tw_inf)*erf_safe((xi - E0_l)/den_w_l))
        elif xi <= Sl:
            Tl.append(Ti + (Tf - Ti)*erf_safe((xi + S0_l)/den_s_l)/erf_lam)
        else:
            Tl.append(Tl_inf + (Tf - Tl_inf)*(erf_safe((xi + S0_l)/den_l_l) - 1)/(erf_mu - 1))

    Tdiff = [Tl[i] - Te[i] for i in range(len(x))]

    params_struct = {
        'lam': lam, 'mu': mu, 'Ti': Ti, 'Tf': Tf, 'Tw_inf': Tw_inf, 'Tl_inf': Tl_inf,
        'S0_e': S0_e, 'E0_e': E0_e, 't0_e': t0_e,
        'S0_l': S0_l, 'E0_l': E0_l, 't0_l': t0_l,
        'alpha_w': alpha_w, 'alpha_s': alpha_s, 'alpha_l': alpha_l,
        'Se': Se, 'Sl': Sl,
        'k_w': k_w, 'k_s': k_s, 'k_l': k_l,
    }

    snap_explicit = explicit_stefan_snapshot(
        k_w, rho_w, c_w, M, R_c, t_phys, params_struct, explicit_opts
    )

    out = CaseResult()
    out['label'] = label
    out['params'] = params_struct
    out['x'] = x
    out['Te'] = Te
    out['Tl'] = Tl
    out['Tdiff'] = Tdiff
    out['num'] = {'explicit': snap_explicit}
    return out


def linspace(a: float, b: float, n: int) -> List[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]
