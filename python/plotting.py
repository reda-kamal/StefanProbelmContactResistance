"""Plotting helpers mirroring the MATLAB visualisations."""
from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency at runtime
    plt = None  # type: ignore[assignment]

try:
    from .vam_face_temps_and_q import vam_face_temps_and_q
except ImportError:  # pragma: no cover - support running as a script
    from vam_face_temps_and_q import vam_face_temps_and_q  # type: ignore


def plot_profiles(case: Dict[str, object]) -> None:
    """Plot early/late VAM temperature profiles with numerical overlays."""

    if not _ensure_matplotlib('temperature profile'):
        return

    params: Dict[str, float] = case['params']  # type: ignore[assignment]
    x: Iterable[float] = case['x']  # type: ignore[assignment]
    Te: Iterable[float] = case['Te']  # type: ignore[assignment]
    Tl: Iterable[float] = case['Tl']  # type: ignore[assignment]

    x_arr = list(x)
    Te_arr = list(Te)
    Tl_arr = list(Tl)

    fig, ax = plt.subplots(num=f"Profiles @ t_phys — {case['label']}")
    ax.grid(True)
    ax.plot(x_arr, Te_arr, 'k--', linewidth=1.6, label='VAM (0-time)')
    ax.plot(x_arr, Tl_arr, 'b-', linewidth=1.7, label='VAM (∞-time)')

    Se = params['Se']
    Sl = params['Sl']
    ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0, label='Wall–solid')
    ax.axvline(Se, color='k', linestyle='--', linewidth=1.0, label=r'$S^{(0)}$')
    ax.axvline(Sl, color='b', linestyle='--', linewidth=1.0, label=r'$S^{(\infty)}$')

    num_struct = case.get('num')  # type: ignore[assignment]
    explicit = None
    if isinstance(num_struct, dict) and num_struct:
        if 'x' in num_struct and 'T' in num_struct:
            explicit = num_struct  # type: ignore[assignment]
        else:
            candidate = num_struct.get('explicit') if isinstance(num_struct, dict) else None
            if isinstance(candidate, dict):
                explicit = candidate  # type: ignore[assignment]
    if isinstance(explicit, dict):
        ax.plot(list(explicit['x']), list(explicit['T']), '.', markersize=6,
                label='Explicit numeric')
        if 'S' in explicit:
            ax.axvline(explicit['S'], color='m', linestyle='--', linewidth=1.2,
                       label=r'$S^{num}_{exp}$')
        _warn_if_unbounded(explicit, 'explicit profile')

    Tw = params['Tw_inf']
    Tf = params['Tf']
    Tl_inf = params['Tl_inf']

    ax.set_xlabel('Physical coordinate  x  [m]')
    ax.set_ylabel('Temperature  [°C]')
    ax.set_title(f"Two VAM vs explicit @ t = t_phys — {case['label']}")
    ax.legend(loc='lower right')
    ax.set_xlim(min(x_arr), max(x_arr))
    ax.set_ylim(min(Tw, Tf, Tl_inf) - 5.0, max(Tw, Tf, Tl_inf) + 5.0)


def plot_diff_profile(case: Dict[str, object]) -> None:
    """Plot the late/early profile difference."""

    if not _ensure_matplotlib('profile difference'):
        return

    params: Dict[str, float] = case['params']  # type: ignore[assignment]
    x = list(case['x'])  # type: ignore[arg-type]
    Tdiff = list(case['Tdiff'])  # type: ignore[arg-type]

    fig, ax = plt.subplots(num=f"Profile difference @ t_phys — {case['label']}")
    ax.grid(True)
    ax.plot(x, Tdiff, 'm-', linewidth=1.7, label=r'$\Delta T(x)$')
    ax.axhline(0.0, color='k', linestyle=':')
    ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0)
    ax.axvline(params['Se'], color='k', linestyle='--', linewidth=1.0, label=r'$S^{(0)}$')
    ax.axvline(params['Sl'], color='b', linestyle='--', linewidth=1.0, label=r'$S^{(\infty)}$')

    ax.set_xlabel('Physical coordinate  x  [m]')
    ax.set_ylabel(r'$\Delta T(x) = T^{(\infty)} - T^{(0)}  [°C]$')
    ax.set_title(f"Difference of VAM profiles @ t = t_phys — {case['label']}")
    ax.legend(loc='lower right')
    ax.set_xlim(min(x), max(x))


def plot_conductance(case: Dict[str, object], R_c: float, t_max: float = 0.1) -> None:
    """Plot effective conductance envelopes for the contact."""

    if not _ensure_matplotlib('effective conductance'):
        return

    params: Dict[str, float] = case['params']  # type: ignore[assignment]
    hc = 1.0 / R_c if R_c else math.inf

    alpha_w = params['alpha_w']
    alpha_s = params['alpha_s']
    k_w = params['k_w']
    k_s = params['k_s']
    S0_e = params['S0_e']
    E0_e = params['E0_e']
    t0_e = params['t0_e']
    S0_l = params['S0_l']
    E0_l = params['E0_l']
    t0_l = params['t0_l']

    t = _linspace(0.0, max(t_max, 1e-9), 1000)

    def gfun_scalar(chi: float) -> float:
        denom = math.erf(chi)
        if abs(denom) < 1e-12:
            denom = math.copysign(1e-12, denom if denom != 0.0 else 1.0)
        return (2.0 * chi * math.exp(-chi * chi)) / (math.sqrt(math.pi) * denom)

    tpe = [ti + t0_e for ti in t]
    tpl = [ti + t0_l for ti in t]

    phi_s = [S0_e / (2.0 * math.sqrt(alpha_s * max(tp, 1e-16))) for tp in tpe]
    phi_w = [E0_e / (2.0 * math.sqrt(alpha_w * max(tp, 1e-16))) for tp in tpe]
    hs_e = [(k_s / S0_e) * gfun_scalar(ps) for ps in phi_s]
    hw_e = [(k_w / E0_e) * gfun_scalar(pw) for pw in phi_w]
    he_e = [_combine_series(hw, hs) for hw, hs in zip(hw_e, hs_e)]
    if he_e:
        he_e[0] = hc

    phi_sL = [S0_l / (2.0 * math.sqrt(alpha_s * max(tp, 1e-16))) for tp in tpl]
    phi_wL = [E0_l / (2.0 * math.sqrt(alpha_w * max(tp, 1e-16))) for tp in tpl]
    hs_l = [(k_s / S0_l) * gfun_scalar(ps) for ps in phi_sL]
    hw_l = [(k_w / E0_l) * gfun_scalar(pw) for pw in phi_wL]
    he_l = [_combine_series(hw, hs) for hw, hs in zip(hw_l, hs_l)]

    fig, ax = plt.subplots(num=f"h_eff(t) — {case['label']}")
    ax.grid(True)
    ax.plot(t, he_e, 'k--', linewidth=1.6, label='VAM (t=0 calibration)')
    ax.plot(t, he_l, 'b-', linewidth=1.7, label='VAM (t=∞ calibration)')
    ax.axhline(hc, color='r', linestyle='-.', linewidth=1.4, label='true h_c (=1/R_c)')
    ax.plot([0.0], [hc], 'ko', markerfacecolor='k', label=r'$h_e^{(0)}(0)=h_c$')

    ax.set_xlabel('time t [s]')
    ax.set_ylabel(r'effective conductance $h_{eff}(t)$ [W m$^{-2}$ K$^{-1}$]')
    ax.set_title(f"Effective contact conductance vs time — {case['label']}")
    ax.legend(loc='lower right')
    ax.set_xlim(0.0, t_max)


def plot_flux(case: Dict[str, object], R_c: float, t_max: float | None = None) -> None:
    """Plot interface heat flux histories for VAM and numerical solutions."""

    if not _ensure_matplotlib('interface flux'):
        return

    params: Dict[str, float] = case['params']  # type: ignore[assignment]

    if t_max is None:
        t_max = _infer_tmax(case)
    if t_max is None:
        t_max = 0.1

    Nt = 600
    t = _linspace(0.0, t_max, Nt)
    _, _, q0 = vam_face_temps_and_q(params, 'early', t, R_c)
    _, _, qI = vam_face_temps_and_q(params, 'late', t, R_c)

    fig, ax = plt.subplots(num=f"Interface flux vs time — {case['label']}")
    ax.grid(True)
    ax.plot(t, q0, 'k--', linewidth=1.6, label=r'VAM$^{(0)}$: q = ΔT/R_c')
    ax.plot(t, qI, 'b-', linewidth=1.7, label=r'VAM$^{(∞)}$: q = ΔT/R_c')

    num_struct = case.get('num')  # type: ignore[assignment]
    explicit = None
    if isinstance(num_struct, dict) and num_struct:
        if 'q' in num_struct:
            explicit = num_struct  # type: ignore[assignment]
        else:
            candidate = num_struct.get('explicit') if isinstance(num_struct, dict) else None
            if isinstance(candidate, dict):
                explicit = candidate  # type: ignore[assignment]
    if isinstance(explicit, dict):
        th, qh, seed_t, label = _extract_flux(explicit, 'Explicit (const R_c)')
        ax.plot(th, qh, '-', linewidth=1.6, color=(0.95, 0.65, 0.2), label=label)
        if seed_t > 0:
            ax.axvline(seed_t, color=(0.4, 0.4, 0.4), linestyle=':', linewidth=1.0,
                       label='Seed time (explicit)')
        _warn_if_flux_unbounded(explicit, 'explicit flux')

    ax.set_xlabel('time t [s]')
    ax.set_ylabel(r'interface heat flux q(0,t) [W m$^{-2}$]')
    ax.set_title(f"Interface flux vs time — {case['label']}")
    ax.legend(loc='lower right')
    ax.set_xlim(0.0, t_max)


def plot_variable_contact_flux(case: Dict[str, object], t_max: float | None = None) -> None:
    """Compare VAM flux envelopes with numerical runs using variable R_c(t)."""

    if not _ensure_matplotlib('variable-contact flux'):
        return

    num_struct = case.get('num')
    if not isinstance(num_struct, dict):
        warnings.warn('No numerical data available for variable-contact flux plot.', RuntimeWarning)
        return

    var_struct = num_struct.get('variable') if isinstance(num_struct, dict) else None
    if not isinstance(var_struct, dict):
        warnings.warn('Variable-contact snapshots not present in case data.', RuntimeWarning)
        return

    params: Dict[str, float] = case['params']  # type: ignore[assignment]

    if t_max is None:
        t_max = _infer_tmax(case)
    if t_max is None:
        t_max = 0.1

    fig, ax = plt.subplots(num=f"Variable-contact flux — {case['label']}")
    ax.grid(True)

    entries = [
        ('early', 'VAM$^{(0)}$', 'Explicit (VAM $R_c^{(0)}$)', 'k'),
        ('late', 'VAM$^{(\infty)}$', 'Explicit (VAM $R_c^{(\infty)}$)', 'b'),
    ]

    for which, label_analytic, label_numeric, color in entries:
        snap = var_struct.get(which)
        if not isinstance(snap, dict):
            continue
        q_struct = snap.get('q') if isinstance(snap, dict) else None
        if not isinstance(q_struct, dict):
            continue
        t_hist = q_struct.get('t_phys') or q_struct.get('t')
        q_hist = q_struct.get('val')
        rc_hist = q_struct.get('R_c')
        if not (isinstance(t_hist, list) and isinstance(q_hist, list) and isinstance(rc_hist, list)):
            continue
        t_vals = [float(v) for v in t_hist]
        if not t_vals:
            continue
        q_vals = [float(v) for v in q_hist]
        rc_vals = [float(v) for v in rc_hist]
        rc_spec = {'times': t_vals, 'values': rc_vals}
        _, _, q_vam = vam_face_temps_and_q(params, which, t_vals, rc_spec)
        ax.plot(t_vals, q_vam, linestyle='--', linewidth=1.4, color=color, label=label_analytic)
        ax.plot(t_vals, q_vals, linestyle='-', linewidth=1.6, color=color, alpha=0.65,
                label=label_numeric)

    ax.set_xlabel('time t [s]')
    ax.set_ylabel(r'interface heat flux q(0,t) [W m$^{-2}$]')
    ax.set_xlim(0.0, t_max)
    ax.set_title(f"Variable-contact comparison — {case['label']}")
    ax.legend(loc='lower right')


def _infer_tmax(case: Dict[str, object]) -> float | None:
    num_struct = case.get('num')
    if not isinstance(num_struct, dict):
        return None
    if 't' in num_struct:
        return max(0.1, float(num_struct['t']))
    explicit = num_struct.get('explicit')
    if isinstance(explicit, dict) and 't' in explicit:
        return max(0.1, float(explicit['t']))
    return None


def _extract_flux(snap: Dict[str, object], base_label: str) -> Tuple[List[float], List[float], float, str]:
    q_struct = snap.get('q') if isinstance(snap, dict) else None
    if isinstance(q_struct, dict):
        if 't_phys' in q_struct:
            th = [float(val) for val in q_struct['t_phys']]
        else:
            th = [float(val) for val in q_struct.get('t', [])]
            t_offset = float(snap.get('t_offset', 0.0))
            th = [val + t_offset for val in th]
        qh = [float(val) for val in q_struct.get('val', [])]
    else:
        th = []
        qh = []

    seed = snap.get('seed') if isinstance(snap, dict) else None
    seed_t = float(seed.get('time', 0.0)) if isinstance(seed, dict) and 'time' in seed else 0.0

    label = base_label
    history = snap.get('history') if isinstance(snap, dict) else None
    if isinstance(history, dict) and history.get('flux_window', 0) and history['flux_window'] > 1:
        label = f"{base_label} ({history['flux_window']}-pt mov. avg.)"

    return th, qh, seed_t, label


def _warn_if_unbounded(snap: Dict[str, object], label: str) -> None:
    bounds = _extract_bounds(snap)
    if not bounds or bounds.get('ok', True):
        return
    profile_violation = float('nan')
    flux_violation = float('nan')
    profile = bounds.get('profile') if isinstance(bounds, dict) else None
    if isinstance(profile, dict):
        profile_violation = profile.get('max_violation', float('nan'))
    flux = bounds.get('flux') if isinstance(bounds, dict) else None
    if isinstance(flux, dict):
        flux_violation = flux.get('max_violation', float('nan'))
    warnings.warn(
        f"Numerical {label} exceeds VAM envelope (ΔT={profile_violation} °C, Δq={flux_violation} W/m²).",
        RuntimeWarning,
        stacklevel=2,
    )


def _warn_if_flux_unbounded(snap: Dict[str, object], label: str) -> None:
    bounds = _extract_bounds(snap)
    if not bounds or bounds.get('ok', True):
        return
    flux = bounds.get('flux') if isinstance(bounds, dict) else None
    flux_violation = float('nan')
    if isinstance(flux, dict):
        flux_violation = flux.get('max_violation', float('nan'))
    warnings.warn(
        f"Numerical {label} exceeds VAM flux envelope (Δq={flux_violation} W/m²).",
        RuntimeWarning,
        stacklevel=2,
    )


def _extract_bounds(snap: Dict[str, object]) -> Dict[str, object] | None:
    if not isinstance(snap, dict):
        return None
    meta = snap.get('meta')
    if isinstance(meta, dict):
        bounds = meta.get('bounds')
        if isinstance(bounds, dict):
            return bounds
    return None


def _ensure_matplotlib(purpose: str) -> bool:
    if plt is not None:
        return True
    warnings.warn(
        f"matplotlib is required to render the {purpose} plot; skipping graphical output.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _combine_series(hw: float, hs: float) -> float:
    if hw == 0.0 or hs == 0.0:
        return 0.0
    return 1.0 / (1.0 / hw + 1.0 / hs)

