"""Enthalpy-based solver mirroring the MATLAB implementation."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

from .options import get_opt, get_struct
from .numerics import moving_average


class Snapshot(dict):
    pass


def enthalpy_stefan_snapshot(k_w: float, rho_w: float, c_w: float,
                             M: Dict[str, float], R_c: float,
                             t_end: float, params: Dict[str, float],
                             opts: Dict[str, object] | None = None) -> Snapshot:
    if opts is None:
        opts = {}

    k_s = M['k_s']; rho_s = M['rho_s']; c_s = M['c_s']
    k_l = M['k_l']; rho_l = M['rho_l']; c_l = M['c_l']
    Tf = M['Tf']; Tw_inf = M['Tw_inf']; Tl_inf = M['Tl_inf']

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

    S_seed = max(dxf, min((Nf - 1) * dxf, Se_seed))

    rhoLcpL = rho_l * c_l
    rhoScpS = rho_s * c_s
    L_vol = rho_s * M['L']

    H = []
    for j in range(Nf):
        if xf[j] <= Se_seed:
            H.append(-L_vol + rhoScpS * (Tfld[j] - Tf))
        else:
            H.append(rhoLcpL * (Tfld[j] - Tl_inf))

    CFL = get_opt(opts, 'CFL', 0.3)
    alpha_max = max(as_, al)
    dt_base = CFL * min(dxw * dxw / (2 * aw), dxf * dxf / (2 * alpha_max))
    sim_duration = max(t_end - seed_time, 0.0)
    if sim_duration <= 0:
        nsteps = 0
        dt_base = 0.0
    else:
        nsteps = max(1, math.ceil(sim_duration / dt_base))
        dt_base = min(dt_base, sim_duration / nsteps)

    t_elapsed = 0.0
    t_phys = seed_time
    t_rel = 0.0

    coeff = {'wall_bulk': aw * dt_base / (dxw * dxw) if dxw else 0.0,
             'wall_edge': 2 * aw * dt_base / (dxw * dxw) if dxw else 0.0}
    curr_dt = dt_base

    Rw = dxw / (2 * k_w)

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

    Tfluid, phi_liq, k_cell = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol)
    q_seed, Tw_face_seed, Ts_face_seed = contact_flux(Tw[0], Tfluid[0], phi_liq[0], Tf, Rw, R_c, dxf, k_cell[0])
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
            coeff = {'wall_bulk': aw * dt_step / (dxw * dxw) if dxw else 0.0,
                     'wall_edge': 2 * aw * dt_step / (dxw * dxw) if dxw else 0.0}
            curr_dt = dt_step

        Tfluid, phi_liq, k_cell = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol)
        q0, Tw_face, Ts_face = contact_flux(Tw[0], Tfluid[0], phi_liq[0], Tf, Rw, R_c, dxf, k_cell[0])

        Tw_new = Tw[:]
        if Nw >= 2:
            Tw_new[0] = Tw[0] + coeff['wall_edge'] * (Tw_face - 2 * Tw[0] + Tw[1])
        if Nw > 2:
            for j in range(1, Nw - 1):
                Tw_new[j] = Tw[j] + coeff['wall_bulk'] * (Tw[j+1] - 2*Tw[j] + Tw[j-1])
        Tw_new[-1] = Tw_inf
        Tw = Tw_new

        q_faces = [0.0] * (Nf + 1)
        q_faces[0] = q0
        for j in range(Nf - 1):
            k_face = harmonic_mean(k_cell[j], k_cell[j+1])
            q_faces[j+1] = -k_face * (Tfluid[j+1] - Tfluid[j]) / dxf
        k_last = max(k_cell[-1], 1e-12)
        q_faces[Nf] = -k_last * (Tl_inf - Tfluid[-1]) / (0.5 * dxf)

        for j in range(Nf):
            qL = q_faces[j]
            qR = q_faces[j+1]
            H[j] = H[j] + dt_step * (qL - qR) / dxf

        t_elapsed += dt_step
        t_phys += dt_step
        t_rel = t_phys - seed_time

        Tfluid, phi_liq, k_cell = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol)
        q_contact, Tw_face_new, Ts_face_new = contact_flux(Tw[0], Tfluid[0], phi_liq[0], Tf, Rw, R_c, dxf, k_cell[0])

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

    Tfluid, phi_liq, k_cell = enthalpy_state(H, Tf, Tl_inf, rhoLcpL, rhoScpS, k_l, k_s, L_vol)

    S_est = estimate_front(phi_liq, dxf)

    snap = Snapshot()
    snap['x'] = xw + xf
    snap['T'] = Tw + Tfluid
    snap['Tw'] = Tw
    snap['Tf'] = Tfluid
    snap['H'] = H
    snap['phi'] = phi_liq
    snap['S'] = S_est
    snap['t'] = t_end
    snap['t_rel'] = t_rel
    snap['t_offset'] = seed_time
    snap['seed'] = {'time': seed_time, 'thickness': S_seed, 'cell_width': dxf, 'Se_vam': Se_seed}
    snap['grid'] = {'dx_wall': dxw, 'dx_fluid': dxf, 'N_wall': Nw, 'N_fluid': Nf, 'L_wall': Lw, 'L_fluid': Lf}
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


def enthalpy_state(H: List[float], Tf: float, Tl_inf: float, rhoLcpL: float, rhoScpS: float,
                   k_l: float, k_s: float, L_vol: float) -> Tuple[List[float], List[float], List[float]]:
    H_tf = rhoLcpL * (Tf - Tl_inf)
    N = len(H)
    T = [0.0] * N
    phi = [0.0] * N
    k_cell = [0.0] * N
    for j, Hj in enumerate(H):
        if Hj >= H_tf:
            T[j] = Tf + (Hj - H_tf) / rhoLcpL
            phi[j] = 1.0
            k_cell[j] = k_l
        elif Hj >= 0.0:
            T[j] = Tl_inf + Hj / rhoLcpL
            phi[j] = 1.0
            k_cell[j] = k_l
        elif Hj >= -L_vol:
            phi_j = max(0.0, min(1.0, 1.0 + Hj / L_vol))
            phi[j] = phi_j
            T[j] = Tf
            k_cell[j] = phi_j * k_l + (1.0 - phi_j) * k_s
        else:
            T[j] = Tf + (Hj + L_vol) / rhoScpS
            phi[j] = 0.0
            k_cell[j] = k_s
    return T, phi, k_cell


def harmonic_mean(kL: float, kR: float) -> float:
    if kL <= 0.0 and kR <= 0.0:
        return 0.0
    if kL <= 0.0:
        return kR
    if kR <= 0.0:
        return kL
    return 2 * kL * kR / (kL + kR)


def contact_flux(Tw_cell: float, T_cell: float, phi_cell: float, Tf: float,
                 Rw: float, R_c: float, dxf: float, k_cell: float) -> Tuple[float, float, float]:
    phi_cell = max(0.0, min(1.0, phi_cell))
    tol = 1e-8
    if tol < phi_cell < 1.0 - tol:
        q0 = (Tw_cell - Tf) / (Rw + R_c)
        Tw_face = Tw_cell - Rw * q0
        Ts_face = Tf
    else:
        k_eff = max(k_cell, 1e-12)
        Rs_eff = dxf / (2 * k_eff)
        q0 = (Tw_cell - T_cell) / (Rw + R_c + Rs_eff)
        Tw_face = Tw_cell - Rw * q0
        Ts_face = T_cell + Rs_eff * q0
    return q0, Tw_face, Ts_face


def estimate_front(phi_liq: List[float], dxf: float) -> float:
    for idx, phi in enumerate(phi_liq):
        if phi < 0.5:
            if idx == 0:
                return 0.5 * dxf
            prev = phi_liq[idx - 1]
            denom = max(prev - phi, 1e-12)
            frac = (0.5 - prev) / denom
            frac = max(0.0, min(1.0, frac))
            pos = (idx - 0.5 + frac) * dxf
            return max(dxf, min((len(phi_liq) - 1) * dxf, pos))
    return max(dxf, (len(phi_liq) - 1) * dxf)
