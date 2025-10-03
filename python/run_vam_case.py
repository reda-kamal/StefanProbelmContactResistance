"""Top-level driver mirroring the MATLAB run_vam_case."""
from __future__ import annotations
from typing import Dict

try:
    from .options import get_opt
    from .explicit_stefan_snapshot import explicit_stefan_snapshot
except ImportError:  # pragma: no cover - allow running as a loose script
    from options import get_opt  # type: ignore
    from explicit_stefan_snapshot import explicit_stefan_snapshot  # type: ignore


class CaseResult(dict):
    pass


def run_vam_case(label: str, k_w: float, rho_w: float, c_w: float,
                 M: Dict[str, float], R_c: float, t_phys: float,
                 opts: Dict[str, object] | None = None) -> CaseResult:
    if opts is None:
        opts = {}

    explicit_opts = get_opt(opts, 'explicit', {})

    k_s = M['k_s']; rho_s = M['rho_s']; c_s = M['c_s']
    k_l = M['k_l']; rho_l = M['rho_l']; c_l = M['c_l']
    L = M['L']; Tf = M['Tf']; Tw_inf = M['Tw_inf']; Tl_inf = M['Tl_inf']

    alpha_w = k_w/(rho_w*c_w)
    alpha_s = k_s/(rho_s*c_s)
    alpha_l = k_l/(rho_l*c_l)

    params_struct = {
        'Tw_inf': Tw_inf,
        'Tl_inf': Tl_inf,
        'Tf': Tf,
        'alpha_w': alpha_w,
        'alpha_s': alpha_s,
        'alpha_l': alpha_l,
        'k_w': k_w,
        'k_s': k_s,
        'k_l': k_l,
    }

    snap_explicit = explicit_stefan_snapshot(
        k_w, rho_w, c_w, M, R_c, t_phys, params_struct, explicit_opts
    )

    out = CaseResult()
    out['label'] = label
    out['params'] = params_struct
    out['num'] = snap_explicit
    return out
