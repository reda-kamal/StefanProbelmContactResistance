"""Explicit finite-difference snapshot for the three-domain Stefan problem."""
from __future__ import annotations
import math
from typing import Dict, List

try:
    from .options import get_opt, get_struct
    from .numerics import moving_average
except ImportError:  # pragma: no cover - allow running as a loose script
    from options import get_opt, get_struct  # type: ignore
    from numerics import moving_average  # type: ignore


class Snapshot(dict):
    """Simple dictionary-based container."""


def update_interface_from_T_gradient(T: List[float], y: List[float], i_s: int,
                                     k_s: float, k_l: float,
                                     rho_s: float, L_latent: float,
                                     dt: float) -> float:
    """Advance the interface using one-sided temperature gradients."""
    i_l = i_s + 1
    if i_s - 1 < 0 or i_l + 1 >= len(T):
        raise IndexError("Interface too close to boundary for one-sided gradients.")

    dy_s = y[i_s] - y[i_s - 1]
    dy_l = y[i_l + 1] - y[i_l]

    dTdy_s = (T[i_s] - T[i_s - 1]) / dy_s
    dTdy_l = (T[i_l + 1] - T[i_l]) / dy_l

    ds_dt = (k_s * dTdy_s - k_l * dTdy_l) / (rho_s * L_latent)
    return dt * ds_dt


def explicit_stefan_snapshot(k_w: float, rho_w: float, c_w: float,
                             M: Dict[str, float], R_c: float,
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
    Tw_face_hist = [0.0] * n_hist_max
    Ts_face_hist = [0.0] * n_hist_max
    ksave = 0
    last_save_time = -1e30

    q_seed = (Tw[0] - Tfld[0]) / (Rw + R_c + Rs)
    Tw_face_seed = Tw[0] - Rw * q_seed
    Ts_face_seed = Tfld[0] + Rs * q_seed
    ksave += 1
    t_hist[ksave-1] = t_rel
    q_hist[ksave-1] = (Tw_face_seed - Ts_face_seed) / R_c
    Tw_face_hist[ksave-1] = Tw_face_seed
    Ts_face_hist[ksave-1] = Ts_face_seed
    last_save_time = t_rel

    for step in range(1, nsteps + 1):
        dt_step = 0.0 if sim_duration == 0 else min(curr_dt, sim_duration - t_elapsed)
        if dt_step <= 0:
            break
        if abs(dt_step - curr_dt) > 1e-15:
            coeff = _local_coeffs(dt_step, aw, as_, al, dxw, dxf)
            curr_dt = dt_step

        m = max(1, min(Nf - 1, int(math.floor(S_real / dxf))))
        solid_end = m - 1  # python index of last solid cell
        liquid_start = m   # python index of first liquid cell

        q0 = (Tw[0] - Tfld[0]) / (Rw + R_c + Rs)
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

        try:
            ds = update_interface_from_T_gradient(
                T=Tfld,
                y=xf,
                i_s=solid_end,
                k_s=k_s,
                k_l=k_l,
                rho_s=rho_s,
                L_latent=L,
                dt=dt_step,
            )
        except IndexError:
            if solid_end >= 1:
                As_ = Tfld[solid_end] - Tf
                Bs_ = Tfld[solid_end - 1] - Tf
                grad_s = (Bs_ - 9 * As_) / (3 * dxf)
            elif solid_end >= 0:
                grad_s = coeff['two_over_dx'] * (Tf - Tfld[solid_end])
            else:
                grad_s = 0.0

            if liquid_start + 1 < Nf:
                Al_ = Tfld[liquid_start] - Tf
                Bl_ = Tfld[liquid_start + 1] - Tf
                grad_l = coeff['grad_upwind'] * (9 * Al_ - Bl_)
            elif liquid_start < Nf:
                grad_l = coeff['two_over_dx'] * (Tfld[liquid_start] - Tf)
            else:
                grad_l = 0.0

            ds = dt_step * (k_s * grad_s - k_l * grad_l) / (rho_s * L)

        S_real = S_real + ds
        S_real = min((Nf - 1) * dxf, max(dxf, S_real))

        t_elapsed += dt_step
        t_phys += dt_step
        t_rel = t_phys - seed_time

        q1 = (Tw[0] - Tfld[0]) / (Rw + R_c + Rs)
        Tw_face_new = Tw[0] - Rw * q1
        Ts_face_new = Tfld[0] + Rs * q1
        q_contact = (Tw_face_new - Ts_face_new) / R_c

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
            Tw_face_hist[ksave-1] = Tw_face_new
            Ts_face_hist[ksave-1] = Ts_face_new
            last_save_time = t_rel

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
    Tw_face_hist = Tw_face_hist[:ksave]
    Ts_face_hist = Ts_face_hist[:ksave]

    if flux_window > 1:
        q_hist = moving_average(q_hist, flux_window)

    snap['q'] = {
        't': t_hist,
        'val': q_hist,
        'Tw_face': Tw_face_hist,
        'Ts_face': Ts_face_hist,
        't_phys': [tv + seed_time for tv in t_hist],
    }
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
