"""Top-level driver mirroring the MATLAB run_vam_case."""
from __future__ import annotations
import math
import copy
import math
from typing import Dict, List, Tuple

try:
    from .numerics import fsolve2
    from .stefan_eqs import stefan_eqs
    from .options import get_opt
    from .explicit_stefan_snapshot import explicit_stefan_snapshot
    from .vam_face_temps_and_q import vam_face_temps_and_q
except ImportError:  # pragma: no cover - allow running as a loose script
    from numerics import fsolve2  # type: ignore
    from stefan_eqs import stefan_eqs  # type: ignore
    from options import get_opt  # type: ignore
    from explicit_stefan_snapshot import explicit_stefan_snapshot  # type: ignore
    from vam_face_temps_and_q import vam_face_temps_and_q  # type: ignore


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

    snap_explicit, meta_explicit = run_refined_snapshot(
        explicit_stefan_snapshot, 'explicit', explicit_opts,
        k_w, rho_w, c_w, M, R_c, t_phys,
        params_struct, x, Te, Tl,
    )
    snap_explicit['meta'] = meta_explicit

    out = CaseResult()
    out['label'] = label
    out['params'] = params_struct
    out['x'] = x
    out['Te'] = Te
    out['Tl'] = Tl
    out['Tdiff'] = Tdiff
    out['num'] = {'explicit': snap_explicit}
    out['diagnostics'] = {'explicit': meta_explicit}
    return out


def linspace(a: float, b: float, n: int) -> List[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def run_refined_snapshot(solver_fn, method_key: str, base_opts: Dict[str, object],
                         k_w: float, rho_w: float, c_w: float,
                         M: Dict[str, float], R_c: float, t_phys: float,
                         params_struct: Dict[str, float],
                         x_ref: List[float], Te_ref: List[float], Tl_ref: List[float]
                         ) -> Tuple[Dict[str, object], Dict[str, object]]:
    refine_cfg = get_opt(base_opts, 'refine', {})
    cfg = {
        'max_iters': int(get_opt(refine_cfg, 'max_iters', 3)),
        'factor': float(get_opt(refine_cfg, 'factor', 1.5)),
        'cfl_shrink': float(get_opt(refine_cfg, 'cfl_shrink', 0.75)),
        'tol_abs_T': float(get_opt(refine_cfg, 'tol_abs_T', 3.0)),
        'tol_rel_T': float(get_opt(refine_cfg, 'tol_rel_T', 0.01)),
        'tol_abs_q': float(get_opt(refine_cfg, 'tol_abs_q', 200.0)),
        'tol_rel_q': float(get_opt(refine_cfg, 'tol_rel_q', 0.01)),
        'history_shrink': float(get_opt(refine_cfg, 'history_shrink', 0.75)),
        'min_CFL': float(get_opt(refine_cfg, 'min_CFL', 0.05)),
    }

    curr_opts = copy.deepcopy(base_opts)
    adjustments: List[Dict[str, object]] = []
    diag_history: List[Dict[str, object]] = []
    ok = False
    bounds_diag: Dict[str, object] = {}

    max_iters = max(1, cfg['max_iters'])
    for iteration in range(1, max_iters + 1):
        snap = solver_fn(k_w, rho_w, c_w, M, R_c, t_phys, params_struct, curr_opts)
        ok, bounds_diag = check_snapshot_bounds(
            snap, x_ref, Te_ref, Tl_ref, params_struct, R_c, t_phys, cfg,
        )
        bounds_diag['iteration'] = iteration
        diag_history.append(bounds_diag)
        if ok:
            break
        if iteration == max_iters:
            break
        curr_opts, adj = refine_options(curr_opts, cfg)
        adj['iter'] = iteration + 1
        adjustments.append(adj)

    meta = {
        'method': method_key,
        'refinement': {
            'iterations': bounds_diag.get('iteration', 1),
            'success': ok,
            'max_iters': max_iters,
            'adjustments': adjustments,
        },
        'bounds': bounds_diag,
        'bounds_history': diag_history[:bounds_diag.get('iteration', 1)],
        'options': curr_opts,
        'initial_options': copy.deepcopy(base_opts),
        'refine_cfg': cfg,
    }
    return snap, meta


def check_snapshot_bounds(snap: Dict[str, object], x_ref: List[float],
                          Te_ref: List[float], Tl_ref: List[float],
                          params: Dict[str, float], R_c: float, t_phys: float,
                          cfg: Dict[str, float]) -> Tuple[bool, Dict[str, object]]:
    diag: Dict[str, object] = {}

    ok_profile = True
    if isinstance(snap.get('x'), list) and isinstance(snap.get('T'), list):
        snap_x = snap['x']  # type: ignore[assignment]
        snap_T = snap['T']  # type: ignore[assignment]
        Te_interp = linear_interp(x_ref, Te_ref, snap_x)
        Tl_interp = linear_interp(x_ref, Tl_ref, snap_x)
        lower_env = [min(a, b) for a, b in zip(Te_interp, Tl_interp)]
        upper_env = [max(a, b) for a, b in zip(Te_interp, Tl_interp)]
        over_err = [tv - up for tv, up in zip(snap_T, upper_env)]
        under_err = [low - tv for tv, low in zip(snap_T, lower_env)]
        max_over = max_positive(over_err)
        max_under = max_positive(under_err)
        profile_violation = max(max_over, max_under)
        env_span = max_span(upper_env, lower_env)
        tol_profile = max(cfg['tol_abs_T'], cfg['tol_rel_T'] * max(env_span, 1e-6))
        ok_profile = profile_violation <= tol_profile + 1e-12
        diag['profile'] = {
            'max_over': max_over,
            'max_under': max_under,
            'max_violation': profile_violation,
            'tol': tol_profile,
            'ok': ok_profile,
        }
    else:
        diag['profile'] = {
            'max_over': 0.0,
            'max_under': 0.0,
            'max_violation': 0.0,
            'tol': 0.0,
            'ok': True,
        }

    ok_flux = True
    q_struct = snap.get('q') if isinstance(snap, dict) else None
    if isinstance(q_struct, dict) and isinstance(q_struct.get('val'), list):
        q_vals = q_struct['val']  # type: ignore[assignment]
        if q_vals:
            if isinstance(q_struct.get('t_phys'), list):
                t_hist = q_struct['t_phys']  # type: ignore[assignment]
            elif isinstance(q_struct.get('t'), list):
                t_hist = [float(t) for t in q_struct['t']]  # type: ignore
                t_offset = float(snap.get('t_offset', 0.0)) if isinstance(snap, dict) else 0.0
                t_hist = [t + t_offset for t in t_hist]
            else:
                t_hist = linspace(0.0, t_phys, len(q_vals))
            _, _, q_early = vam_face_temps_and_q(params, 'early', t_hist, R_c)
            _, _, q_late = vam_face_temps_and_q(params, 'late', t_hist, R_c)
            lower_q = [min(a, b) for a, b in zip(q_early, q_late)]
            upper_q = [max(a, b) for a, b in zip(q_early, q_late)]
            q_over = [q - up for q, up in zip(q_vals, upper_q)]
            q_under = [low - q for q, low in zip(q_vals, lower_q)]
            max_over_q = max_positive(q_over)
            max_under_q = max_positive(q_under)
            flux_violation = max(max_over_q, max_under_q)
            span_q = max_span(upper_q, lower_q)
            tol_flux = max(cfg['tol_abs_q'], cfg['tol_rel_q'] * max(span_q, 1e-6))
            ok_flux = flux_violation <= tol_flux + 1e-6
            diag['flux'] = {
                'max_over': max_over_q,
                'max_under': max_under_q,
                'max_violation': flux_violation,
                'tol': tol_flux,
                'ok': ok_flux,
            }
        else:
            diag['flux'] = {
                'max_over': 0.0,
                'max_under': 0.0,
                'max_violation': 0.0,
                'tol': 0.0,
                'ok': True,
            }
    else:
        diag['flux'] = {
            'max_over': 0.0,
            'max_under': 0.0,
            'max_violation': 0.0,
            'tol': 0.0,
            'ok': True,
        }

    ok = ok_profile and ok_flux
    diag['ok_profile'] = ok_profile
    diag['ok_flux'] = ok_flux
    diag['ok'] = ok
    return ok, diag


def refine_options(curr_opts: Dict[str, object], cfg: Dict[str, float]) -> Tuple[Dict[str, object], Dict[str, object]]:
    next_opts = copy.deepcopy(curr_opts)

    CFL = next_opts.get('CFL') if isinstance(next_opts, dict) else None
    if isinstance(CFL, (int, float)) and CFL > cfg['min_CFL']:
        next_opts['CFL'] = max(cfg['min_CFL'], CFL * cfg['cfl_shrink'])

    history_dt = next_opts.get('history_dt') if isinstance(next_opts, dict) else None
    if isinstance(history_dt, (int, float)) and history_dt > 0:
        next_opts['history_dt'] = history_dt * cfg['history_shrink']

    wall_opts = next_opts.get('wall') if isinstance(next_opts, dict) else None
    next_opts['wall'] = upscale_cells(wall_opts, cfg['factor'])
    fluid_opts = next_opts.get('fluid') if isinstance(next_opts, dict) else None
    next_opts['fluid'] = upscale_cells(fluid_opts, cfg['factor'])

    adj = {
        'CFL': next_opts.get('CFL'),
        'wall_cells': next_opts['wall'].get('cells') if isinstance(next_opts.get('wall'), dict) else None,
        'fluid_cells': next_opts['fluid'].get('cells') if isinstance(next_opts.get('fluid'), dict) else None,
    }
    return next_opts, adj


def upscale_cells(sub_opts, factor: float) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(sub_opts, dict):
        out.update(sub_opts)
    curr_cells = out.get('cells')
    if not isinstance(curr_cells, (int, float)):
        curr_cells = out.get('min_cells')
        if not isinstance(curr_cells, (int, float)):
            curr_cells = 400
    new_cells = max(int(round(curr_cells)) + 2, int(math.ceil(curr_cells * factor)))
    out['cells'] = new_cells
    prev_min = out.get('min_cells')
    if isinstance(prev_min, (int, float)):
        out['min_cells'] = max(int(prev_min), new_cells)
    else:
        out['min_cells'] = new_cells
    return out


def linear_interp(xs: List[float], ys: List[float], xq: List[float]) -> List[float]:
    if not xs:
        return [float('nan')] * len(xq)
    if len(xs) == 1:
        return [ys[0]] * len(xq)
    result: List[float] = []
    for x in xq:
        if x <= xs[0]:
            y = ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0])
            result.append(y)
            continue
        if x >= xs[-1]:
            y = ys[-2] + (ys[-1] - ys[-2]) * (x - xs[-2]) / (xs[-1] - xs[-2])
            result.append(y)
            continue
        lo = 0
        hi = len(xs) - 1
        while hi - lo > 1:
            mid = (hi + lo) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0 = xs[lo]
        x1 = xs[lo + 1]
        y0 = ys[lo]
        y1 = ys[lo + 1]
        if x1 == x0:
            result.append(y0)
        else:
            result.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return result


def max_positive(values: List[float]) -> float:
    best = 0.0
    for val in values:
        if isinstance(val, (int, float)) and val > best:
            best = float(val)
    return best


def max_span(upper: List[float], lower: List[float]) -> float:
    span = 0.0
    for up, lo in zip(upper, lower):
        if isinstance(up, (int, float)) and isinstance(lo, (int, float)):
            span = max(span, abs(up - lo))
    return span
