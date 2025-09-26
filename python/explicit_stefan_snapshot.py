"""Explicit finite-difference snapshot for the three-domain Stefan problem."""
from __future__ import annotations
import math
from typing import Callable, Dict, List, Sequence

try:
    from .options import get_opt, get_struct
    from .numerics import moving_average
except ImportError:  # pragma: no cover - allow running as a loose script
    from options import get_opt, get_struct  # type: ignore
    from numerics import moving_average  # type: ignore


class Snapshot(dict):
    """Simple dictionary-based container."""


def explicit_stefan_snapshot(k_w: float, rho_w: float, c_w: float,
                             M: Dict[str, float], R_c,
                             t_end: float, params: Dict[str, float],
                             opts: Dict[str, object] | None = None) -> Snapshot:
    if opts is None:
        opts = {}

    k_s = M['k_s']; rho_s = M['rho_s']; c_s = M['c_s']
    k_l = M['k_l']; rho_l = M['rho_l']; c_l = M['c_l']
    L = M['L']; Tf = M['Tf']; Tw_inf = M['Tw_inf']; Tl_inf = M['Tl_inf']

    aw = k_w/(rho_w*c_w)
    as_ = k_s/(rho_s*c_s)
    al = k_l/(rho_l*c_l)

    lam = params['lam']; Ti = params['Ti']; S0_e = params['S0_e']
    E0_e = params['E0_e']; t0_e = params['t0_e']; mu = params['mu']

    nodes_per_diff = get_opt(opts, 'nodes_per_diff', None)
    min_cells_legacy = get_opt(opts, 'min_cells', 400)
    domain_factor = get_opt(opts, 'domain_factor', 5)
    min_length_legacy = get_opt(opts, 'min_length', 2e-3)

    wall_opts = get_struct(opts, 'wall')
    fluid_opts = get_struct(opts, 'fluid')

    min_seed_cells = get_opt(opts, 'min_seed_cells', get_opt(fluid_opts, 'min_seed_cells', 1))
    nsave = get_opt(opts, 'nsave', 2000)
    history_dt = get_opt(opts, 'history_dt', 0.0)
    flux_window_opt = get_opt(opts, 'flux_smoothing', 0)
    flux_window = 0
    if flux_window_opt and flux_window_opt > 1:
        flux_window = int(math.floor(flux_window_opt))
        if flux_window % 2 == 0:
            flux_window += 1

    wall_extent_factor = get_opt(wall_opts, 'extent_factor', domain_factor)
    fluid_extent_factor = get_opt(fluid_opts, 'extent_factor', domain_factor)
    wall_min_length = get_opt(wall_opts, 'min_length', min_length_legacy)
    fluid_min_length = get_opt(fluid_opts, 'min_length', min_length_legacy)

    wall_length_user = get_opt(wall_opts, 'length', get_opt(wall_opts, 'extent', None))
    wall_length_fixed = wall_length_user is not None
    fluid_length_user = get_opt(fluid_opts, 'length', get_opt(fluid_opts, 'extent', None))
    fluid_length_fixed = fluid_length_user is not None

    wall_min_cells = get_opt(wall_opts, 'min_cells', min_cells_legacy)
    fluid_min_cells = get_opt(fluid_opts, 'min_cells', min_cells_legacy)

    wall_cells = get_opt(wall_opts, 'cells', None)
    if wall_cells is None:
        wall_dx = get_opt(wall_opts, 'dx', None)
        if wall_dx:
            length = wall_length_user if wall_length_user else wall_min_length
            wall_cells = max(3, round(length / wall_dx))
        else:
            wall_cells = wall_min_cells
            if nodes_per_diff:
                wall_cells = max(wall_cells, math.ceil(nodes_per_diff * wall_extent_factor))
    else:
        wall_cells = max(3, round(wall_cells))
    wall_cells = max(wall_cells, wall_min_cells)

    fluid_cells = get_opt(fluid_opts, 'cells', None)
    if fluid_cells is None:
        fluid_dx = get_opt(fluid_opts, 'dx', None)
        if fluid_dx:
            length = fluid_length_user if fluid_length_user else fluid_min_length
            fluid_cells = max(3, round(length / fluid_dx))
        else:
            fluid_cells = fluid_min_cells
            if nodes_per_diff:
                fluid_cells = max(fluid_cells, math.ceil(nodes_per_diff * fluid_extent_factor))
    else:
        fluid_cells = max(3, round(fluid_cells))
    fluid_cells = max(fluid_cells, fluid_min_cells)

    Nw = wall_cells
    Nf = fluid_cells

    def compute_lengths(t_phys: float) -> tuple[float, float]:
        if wall_length_fixed:
            Lw = max(wall_length_user, wall_min_length)
        else:
            Lw = max(wall_extent_factor * math.sqrt(aw * max(t_phys, 1e-16)), wall_min_length)
        if fluid_length_fixed:
            Lf = max(fluid_length_user, fluid_min_length)
        else:
            Lf = max(fluid_extent_factor * math.sqrt(max(as_, al) * max(t_phys, 1e-16)), fluid_min_length)
        return Lw, Lf

    seed_time = 0.0
    t_final_phys = t_end
    for _ in range(5):
        Lw, Lf = compute_lengths(t_final_phys)
        dxw = Lw / Nw
        dxf = Lf / Nf
        seed_thickness = min_seed_cells * dxf
        seed_time_new = ((seed_thickness + S0_e)**2) / (4 * lam * lam * as_) - t0_e
        seed_time_new = max(0.0, seed_time_new)
        t_final_phys = max(t_end, seed_time_new)
        if abs(seed_time_new - seed_time) < 1e-12:
            seed_time = seed_time_new
            break
        seed_time = seed_time_new

    seed_time = max(0.0, seed_time)
    t_final_phys = max(t_end, seed_time)
    Lw, Lf = compute_lengths(t_final_phys)
    dxw = Lw / Nw
    dxf = Lf / Nf

    xw = [-(i + 0.5) * dxw for i in range(Nw)]
    xf = [(i + 0.5) * dxf for i in range(Nf)]

    seed_thickness = min_seed_cells * dxf
    seed_time = ((seed_thickness + S0_e)**2) / (4 * lam * lam * as_) - t0_e
    seed_time = max(0.0, seed_time)
    t_final_phys = max(t_end, seed_time)

    tpe_seed = seed_time + t0_e
    den_w_e = 2 * math.sqrt(aw * tpe_seed)
    den_s_e = 2 * math.sqrt(as_ * tpe_seed)
    den_l_e = 2 * math.sqrt(al * tpe_seed)
    erf_lam = math.erf(lam)
    if abs(erf_lam) < 1e-12:
        erf_lam = math.copysign(1e-12, erf_lam if erf_lam != 0 else 1.0)
    erf_mu = math.erf(mu)
    if abs(erf_mu - 1.0) < 1e-12:
        erf_mu = 1.0 - math.copysign(1e-12, 1.0)

    Se_seed = 2 * lam * math.sqrt(as_ * tpe_seed) - S0_e

    Tw = [Ti + (Ti - Tw_inf) * math.erf((x - E0_e) / den_w_e) for x in xw]
    Tfld: List[float] = []
    for x in xf:
        if x <= Se_seed:
            Tfld.append(Ti + (Tf - Ti) * math.erf((x + S0_e) / den_s_e) / erf_lam)
        else:
            Tfld.append(Tl_inf + (Tf - Tl_inf) * (math.erf((x + S0_e) / den_l_e) - 1.0) / (erf_mu - 1.0))

    S_real = max(dxf, min((Nf - 1) * dxf, Se_seed))

    seed_info = {
        'Se_vam': Se_seed,
        'time': seed_time,
        'thickness': S_real,
        'cell_width': dxf,
    }

    t_phys = seed_time
    t_rel = 0.0
    sim_duration = max(t_end - seed_time, 0.0)

    CFL = get_opt(opts, 'CFL', 0.3)
    dt_base = CFL * min(dxw * dxw / (2 * aw), dxf * dxf / (2 * max(as_, al)))
    if sim_duration <= 0:
        nsteps = 0
        dt_base = 0.0
    else:
        nsteps = max(1, math.ceil(sim_duration / dt_base))
        dt_base = min(dt_base, sim_duration / nsteps)
    t_elapsed = 0.0

    Rw = dxw / (2 * k_w)
    Rs = dxf / (2 * k_s)

    rc_eval, rc_meta = _prepare_rc_evaluator(R_c)

    coeff = _local_coeffs(dt_base, aw, as_, al, dxw, dxf)
    curr_dt = dt_base

    if history_dt and history_dt > 0:
        n_hist_max = max(2, math.ceil(sim_duration / history_dt) + 2)
        stride = 1
    else:
        stride = max(1, math.floor(max(nsteps, 1) / max(nsave, 1))) if nsave else 1
        n_hist_max = max(2, math.ceil(max(nsteps, 1) / stride) + 1)
    t_hist = [0.0] * n_hist_max
    q_hist = [0.0] * n_hist_max
    rc_hist = [0.0] * n_hist_max
    Tw_face_hist = [0.0] * n_hist_max
    Ts_face_hist = [0.0] * n_hist_max
    front_hist = [0.0] * n_hist_max
    front_err_s = [0.0] * n_hist_max
    front_err_l = [0.0] * n_hist_max
    ksave = 0
    last_save_time = -1e30

    m_seed = max(0, min(Nf - 1, int(math.floor(S_real / dxf))))
    solid_end_seed = m_seed - 1
    liquid_start_seed = m_seed
    solid_cells_seed = max(1, m_seed)

    rc_seed = rc_eval(t_phys)
    q_seed = (Tw[0] - Tfld[0]) / (Rw + rc_seed + Rs)
    Tw_face_seed = Tw[0] - Rw * q_seed
    Ts_face_seed = Tfld[0] + Rs * q_seed
    Tw_face_seed_lin = _face_temperature(Tw, Tw_face_seed, 'wall')
    Ts_face_seed_lin = _face_temperature(Tfld, Ts_face_seed, 'solid', solid_cells_seed)
    grad_s_seed, grad_l_seed = _stefan_gradients(
        Tfld, dxf, Tf, S_real, solid_end_seed, liquid_start_seed,
    )
    Ts_int_seed, Tl_int_seed = _interface_temps_from_gradients(
        Tfld, dxf, S_real, grad_s_seed, grad_l_seed,
        solid_end_seed, liquid_start_seed,
    )
    ksave += 1
    t_hist[ksave-1] = t_rel
    q_hist[ksave-1] = _contact_flux(Tw_face_seed_lin, Ts_face_seed_lin, q_seed, rc_seed)
    rc_hist[ksave-1] = rc_seed
    Tw_face_hist[ksave-1] = Tw_face_seed_lin
    Ts_face_hist[ksave-1] = Ts_face_seed_lin
    front_hist[ksave-1] = S_real
    front_err_s[ksave-1] = Ts_int_seed - Tf if Ts_int_seed == Ts_int_seed else float('nan')
    front_err_l[ksave-1] = Tl_int_seed - Tf if Tl_int_seed == Tl_int_seed else float('nan')
    last_save_time = t_rel

    for step in range(1, nsteps + 1):
        dt_step = 0.0 if sim_duration == 0 else min(curr_dt, sim_duration - t_elapsed)
        if dt_step <= 0:
            break
        if abs(dt_step - curr_dt) > 1e-15:
            coeff = _local_coeffs(dt_step, aw, as_, al, dxw, dxf)
            curr_dt = dt_step

        m = max(0, min(Nf - 1, int(math.floor(S_real / dxf))))
        solid_end = m - 1  # python index of last solid cell
        liquid_start = m   # python index of first liquid cell

        rc_curr = rc_eval(t_phys)
        q0 = (Tw[0] - Tfld[0]) / (Rw + rc_curr + Rs)
        Tw_face = Tw[0] - Rw * q0
        Ts_face = Tfld[0] + Rs * q0

        Tw_new = Tw[:]
        if Nw >= 2:
            Tw_new[0] = Tw[0] + coeff['wall_edge'] * (Tw_face - 2 * Tw[0] + Tw[1])
        if Nw > 2:
            for j in range(1, Nw - 1):
                Tw_new[j] = Tw[j] + coeff['wall_bulk'] * (Tw[j+1] - 2 * Tw[j] + Tw[j-1])
        Tw_new[-1] = Tw_inf
        Tw = Tw_new

        Tn = Tfld[:]
        if Nf >= 2:
            Tn[0] = Tfld[0] + coeff['solid_edge'] * (Tfld[1] - 2 * Tfld[0] + Ts_face)
        if solid_end >= 1:
            for j in range(1, solid_end):
                Tn[j] = Tfld[j] + coeff['solid_bulk'] * (Tfld[j+1] - 2*Tfld[j] + Tfld[j-1])
        if solid_end >= 0:
            neighbor = Ts_face if solid_end == 0 else Tfld[solid_end - 1]
            lap_m = coeff['near_face_coeff'] * (neighbor - 3 * Tfld[solid_end] + 2 * Tf)
            Tn[solid_end] = Tfld[solid_end] + coeff['as_dt'] * lap_m
        if liquid_start < Nf:
            if liquid_start + 1 < Nf:
                lap_mp1 = coeff['near_face_coeff'] * (Tfld[liquid_start + 1] - 3 * Tfld[liquid_start] + 2 * Tf)
                Tn[liquid_start] = Tfld[liquid_start] + coeff['al_dt'] * lap_mp1
            else:
                Tn[liquid_start] = Tf
        if liquid_start + 1 < Nf - 0:
            for j in range(liquid_start + 1, Nf - 1):
                Tn[j] = Tfld[j] + coeff['liquid_bulk'] * (Tfld[j+1] - 2*Tfld[j] + Tfld[j-1])
        Tn[-1] = Tl_inf
        Tfld = Tn

        grad_s, grad_l = _stefan_gradients(Tfld, dxf, Tf, S_real, solid_end, liquid_start)
        Ts_int, Tl_int = _interface_temps_from_gradients(
            Tfld, dxf, S_real, grad_s, grad_l, solid_end, liquid_start,
        )

        S_real = S_real + dt_step * (k_s * grad_s - k_l * grad_l) / (rho_s * L)
        S_real = min((Nf - 1) * dxf, max(1e-9, S_real))

        m_next = max(0, min(Nf - 1, int(math.floor(S_real / dxf))))

        t_elapsed += dt_step
        t_phys += dt_step
        t_rel = t_phys - seed_time

        rc_next = rc_eval(t_phys)
        q1 = (Tw[0] - Tfld[0]) / (Rw + rc_next + Rs)
        Tw_face_new = Tw[0] - Rw * q1
        Ts_face_new = Tfld[0] + Rs * q1
        solid_cells = max(1, m_next)
        Tw_face_lin = _face_temperature(Tw, Tw_face_new, 'wall')
        Ts_face_lin = _face_temperature(Tfld, Ts_face_new, 'solid', solid_cells)
        q_contact = _contact_flux(Tw_face_lin, Ts_face_lin, q1, rc_next)

        should_save = False
        if history_dt and history_dt > 0:
            if (t_rel - last_save_time) >= history_dt * (1 - 1e-8) or step == nsteps:
                should_save = True
        else:
            if (step % stride) == 0 or step == nsteps:
                should_save = True
        if should_save:
            ksave += 1
            t_hist[ksave-1] = t_rel
            q_hist[ksave-1] = q_contact
            rc_hist[ksave-1] = rc_next
            Tw_face_hist[ksave-1] = Tw_face_lin
            Ts_face_hist[ksave-1] = Ts_face_lin
            front_hist[ksave-1] = S_real
            front_err_s[ksave-1] = Ts_int - Tf if Ts_int == Ts_int else float('nan')
            front_err_l[ksave-1] = Tl_int - Tf if Tl_int == Tl_int else float('nan')
            last_save_time = t_rel

    m_final = max(0, min(Nf - 1, int(math.floor(S_real / dxf)))) if Nf > 0 else 0
    solid_end_final = m_final - 1
    liquid_start_final = m_final

    grad_s_final, grad_l_final = _stefan_gradients(
        Tfld, dxf, Tf, S_real, solid_end_final, liquid_start_final,
    )
    Ts_int_final, Tl_int_final = _interface_temps_from_gradients(
        Tfld, dxf, S_real, grad_s_final, grad_l_final,
        solid_end_final, liquid_start_final,
    )

    rc_final = rc_eval(t_phys)
    denom_final = Rw + rc_final + Rs
    if abs(denom_final) < 1e-18:
        q_half_final = 0.0
    else:
        q_half_final = (Tw[0] - Tfld[0]) / denom_final
    Tw_face_final_raw = Tw[0] - Rw * q_half_final
    Ts_face_final_raw = Tfld[0] + Rs * q_half_final
    solid_cells_final = max(1, m_final)
    Tw_face_final = _face_temperature(Tw, Tw_face_final_raw, 'wall')
    Ts_face_final = _face_temperature(Tfld, Ts_face_final_raw, 'solid', solid_cells_final)
    q_contact_final = _contact_flux(Tw_face_final, Ts_face_final, q_half_final, rc_final)

    snap = Snapshot()
    snap['x'] = xw + xf
    snap['T'] = Tw + Tfld
    snap['S'] = S_real
    snap['t'] = t_end
    snap['t_rel'] = t_rel
    snap['t_offset'] = seed_time
    snap['seed'] = seed_info
    snap['grid'] = {
        'dx_wall': dxw,
        'dx_fluid': dxf,
        'N_wall': Nw,
        'N_fluid': Nf,
        'L_wall': Lw,
        'L_fluid': Lf,
    }
    snap['history'] = {'history_dt': history_dt, 'flux_window': flux_window, 'nsave': nsave}

    t_hist = t_hist[:ksave]
    q_hist = q_hist[:ksave]
    rc_hist = rc_hist[:ksave]
    Tw_face_hist = Tw_face_hist[:ksave]
    Ts_face_hist = Ts_face_hist[:ksave]
    front_hist = front_hist[:ksave]
    front_err_s = front_err_s[:ksave]
    front_err_l = front_err_l[:ksave]

    t_phys_hist = [tv + seed_time for tv in t_hist]

    if flux_window > 1:
        q_hist = moving_average(q_hist, flux_window)

    snap['q'] = {
        't': t_hist,
        'val': q_hist,
        'R_c': rc_hist,
        'Tw_face': Tw_face_hist,
        'Ts_face': Ts_face_hist,
        't_phys': t_phys_hist,
    }
    snap['front'] = {
        't': t_hist,
        't_phys': t_phys_hist,
        'S': front_hist,
        'solid_delta': front_err_s,
        'liquid_delta': front_err_l,
    }
    snap['cells'] = {
        'wall': {'x': list(xw), 'T': list(Tw)},
        'fluid': {'x': list(xf), 'T': list(Tfld), 'solid_count': m_final},
    }
    snap['faces'] = {
        'wall': {
            'x': 0.0,
            'Tw': Tw_face_final,
            'Ts': Ts_face_final,
            'q': q_contact_final,
            'R_c': rc_final,
            'Rw': Rw,
            'Rs': Rs,
        },
        'interface': {
            'x': S_real,
            'Tf': Tf,
            'Ts': Ts_int_final,
            'Tl': Tl_int_final,
            'solid_grad': grad_s_final,
            'liquid_grad': grad_l_final,
            'index': m_final,
        },
    }
    snap['interface'] = {
        'x': S_real,
        'Tf': Tf,
        'Ts': Ts_int_final,
        'Tl': Tl_int_final,
        'solid_grad': grad_s_final,
        'liquid_grad': grad_l_final,
        'index': m_final,
    }
    if rc_meta is not None:
        snap['q_meta'] = rc_meta
    return snap


def _local_coeffs(dt: float, aw: float, as_: float, al: float, dxw: float, dxf: float) -> Dict[str, float]:
    return {
        'wall_bulk': aw * dt / (dxw * dxw) if dxw else 0.0,
        'wall_edge': 2 * aw * dt / (dxw * dxw) if dxw else 0.0,
        'solid_bulk': as_ * dt / (dxf * dxf) if dxf else 0.0,
        'solid_edge': 2 * as_ * dt / (dxf * dxf) if dxf else 0.0,
        'liquid_bulk': al * dt / (dxf * dxf) if dxf else 0.0,
        'near_face_coeff': 4 / (3 * dxf * dxf) if dxf else 0.0,
        'as_dt': as_ * dt,
        'al_dt': al * dt,
        'two_over_dx': 2 / dxf if dxf else 0.0,
        'grad_upwind': 1 / (3 * dxf) if dxf else 0.0,
    }


def _face_temperature(values: List[float], fallback: float, side: str,
                      solid_cells: int | None = None) -> float:
    if side == 'wall':
        count = len(values)
    else:
        count = solid_cells if solid_cells is not None else len(values)
        count = min(count, len(values))

    if count >= 3:
        # Quadratic extrapolation through the first three cell centres.
        return (15 * values[0] - 10 * values[1] + 3 * values[2]) / 8.0
    if count >= 2:
        return values[0] + 0.5 * (values[0] - values[1])
    return fallback


def _contact_flux(Tw_face: float, Ts_face: float, q_halfcell: float, R_c: float) -> float:
    if abs(R_c) < 1e-12:
        return q_halfcell
    return (Tw_face - Ts_face) / R_c


def _interface_temps_from_gradients(Tfld: List[float], dxf: float, S_real: float,
                                    grad_s: float, grad_l: float,
                                    solid_end: int, liquid_start: int
                                    ) -> tuple[float, float]:
    Ts_int = float('nan')
    Tl_int = float('nan')

    if solid_end >= 0 and solid_end < len(Tfld):
        center = (solid_end + 0.5) * dxf
        Ts_int = Tfld[solid_end] + grad_s * (S_real - center)

    if liquid_start >= 0 and liquid_start < len(Tfld):
        center = (liquid_start + 0.5) * dxf
        Tl_int = Tfld[liquid_start] - grad_l * (center - S_real)

    return Ts_int, Tl_int


def _stefan_gradients(Tfld: List[float], dxf: float, Tf: float, S_real: float,
                      solid_end: int, liquid_start: int) -> tuple[float, float]:
    grad_s = 0.0
    if solid_end >= 0:
        center = (solid_end + 0.5) * dxf
        dist = max(1e-12, S_real - center)
        if solid_end >= 1 and len(Tfld) >= solid_end + 1:
            z1 = dist
            z2 = max(z1 + dxf, 1e-12)
            T0 = Tfld[solid_end]
            T1 = Tfld[solid_end - 1]
            grad_s = _dirichlet_gradient(T0, T1, Tf, z1, z2 - z1, orientation='solid')
        else:
            grad_s = (Tf - Tfld[solid_end]) / dist

    grad_l = 0.0
    if liquid_start < len(Tfld):
        center = (liquid_start + 0.5) * dxf
        dist = max(1e-12, center - S_real)
        if liquid_start + 1 < len(Tfld):
            z1 = dist
            z2 = max(z1 + dxf, 1e-12)
            T0 = Tfld[liquid_start]
            T1 = Tfld[liquid_start + 1]
            grad_l = _dirichlet_gradient(T0, T1, Tf, z1, z2 - z1, orientation='liquid')
        else:
            grad_l = (Tfld[liquid_start] - Tf) / dist

    return grad_s, grad_l


def _dirichlet_gradient(T0: float, T1: float, Tf: float, z1: float, dz: float,
                        orientation: str) -> float:
    z1 = max(z1, 1e-12)
    z2 = z1 + max(dz, 1e-12)
    denom = z2 * (z2 - z1)
    if abs(denom) < 1e-18:
        b = (T0 - Tf) / z1
    else:
        c = (T1 - Tf - (T0 - Tf) * (z2 / z1)) / denom
        b = (T0 - Tf - c * z1 * z1) / z1
    if orientation == 'solid':
        return -b
    return b


def _prepare_rc_evaluator(R_c) -> tuple[Callable[[float], float], Dict[str, object] | None]:
    if callable(R_c):
        return R_c, {'type': 'callable'}

    if isinstance(R_c, dict):
        times = None
        values = None
        for key in ('times', 'time', 't'):
            if key in R_c:
                times = [float(v) for v in R_c[key]]
                break
        for key in ('values', 'val', 'R_c', 'rc'):
            if key in R_c:
                values = [float(v) for v in R_c[key]]
                break
        if times is None or values is None:
            raise ValueError('Invalid R_c specification dictionary')
        eval_fn = lambda tt: _interp1(times, values, float(tt))
        return eval_fn, {'type': 'timeseries', 'times': times, 'values': values}

    if isinstance(R_c, Sequence) and len(R_c) == 2:
        times = [float(v) for v in R_c[0]]
        values = [float(v) for v in R_c[1]]
        eval_fn = lambda tt: _interp1(times, values, float(tt))
        return eval_fn, {'type': 'timeseries', 'times': times, 'values': values}

    rc_const = float(R_c)
    return lambda _tt: rc_const, {'type': 'constant', 'value': rc_const}


def _interp1(times: Sequence[float], values: Sequence[float], t: float) -> float:
    if not times:
        raise ValueError('Empty time series for R_c specification')
    if len(times) != len(values):
        raise ValueError('Mismatched time/value lengths for R_c specification')
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
