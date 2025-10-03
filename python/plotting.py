import math
import warnings
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency at runtime
    plt = None  # type: ignore[assignment]


def plot_profiles(case: Dict[str, object]) -> None:
    """Plot the explicit temperature profile produced by the numerical solver."""

    if not _ensure_matplotlib('temperature profile'):
        return

    snap = _get_numeric_snapshot(case)
    if snap is None:
        warnings.warn('Case missing numerical snapshot; skipping profile plot.', RuntimeWarning, stacklevel=2)
        return

    x_vals = [float(val) for val in snap.get('x', [])]
    T_vals = [float(val) for val in snap.get('T', [])]
    if not x_vals or not T_vals:
        warnings.warn('Numerical snapshot lacks temperature profile data.', RuntimeWarning, stacklevel=2)
        return

    params = _get_params(case)
    Tf = params.get('Tf', 0.0)
    Tw_inf = params.get('Tw_inf', min(T_vals))
    Tl_inf = params.get('Tl_inf', max(T_vals))

    fig, ax = plt.subplots(num=f"Explicit profile @ t_phys — {case['label']}")
    ax.grid(True)
    ax.plot(x_vals, T_vals, '-', linewidth=1.8, color=(0.2, 0.4, 0.8), label='Explicit numeric')

    if 'S' in snap:
        ax.axvline(float(snap['S']), color='m', linestyle='--', linewidth=1.2, label='Interface $S(t)$')
    ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0, label='Wall/solid boundary')

    ax.set_xlabel('Physical coordinate  x  [m]')
    ax.set_ylabel('Temperature  [°C]')
    ax.set_title(f"Explicit profile at t = t_phys — {case['label']}")
    ax.legend(loc='best')
    ax.set_xlim(min(x_vals), max(x_vals))
    ax.set_ylim(min(Tw_inf, Tf, Tl_inf) - 5.0, max(Tw_inf, Tf, Tl_inf) + 5.0)


def plot_diff_profile(case: Dict[str, object]) -> None:
    """Plot the explicit temperature relative to the fusion temperature."""

    if not _ensure_matplotlib('profile difference'):
        return

    snap = _get_numeric_snapshot(case)
    if snap is None:
        warnings.warn('Case missing numerical snapshot; skipping profile-difference plot.', RuntimeWarning, stacklevel=2)
        return

    x_vals = [float(val) for val in snap.get('x', [])]
    T_vals = [float(val) for val in snap.get('T', [])]
    if not x_vals or not T_vals:
        warnings.warn('Numerical snapshot lacks profile data for difference plot.', RuntimeWarning, stacklevel=2)
        return

    params = _get_params(case)
    Tf = params.get('Tf', 0.0)
    diff = [temp - Tf for temp in T_vals]

    fig, ax = plt.subplots(num=f"Explicit profile − Tf @ t_phys — {case['label']}")
    ax.grid(True)
    ax.plot(x_vals, diff, 'm-', linewidth=1.8, label=r'$T(x) - T_f$ (explicit)')
    ax.axhline(0.0, color='k', linestyle=':')
    if 'S' in snap:
        ax.axvline(float(snap['S']), color='m', linestyle='--', linewidth=1.2, label='Interface $S(t)$')
    ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0)

    ax.set_xlabel('Physical coordinate  x  [m]')
    ax.set_ylabel(r'Temperature offset  [°C]')
    ax.set_title(f"Explicit deviation from $T_f$ — {case['label']}")
    ax.legend(loc='best')
    ax.set_xlim(min(x_vals), max(x_vals))


def plot_conductance(case: Dict[str, object], R_c: float, t_max: float = 0.1) -> None:
    """Plot the effective contact conductance computed from the explicit flux history."""

    if not _ensure_matplotlib('effective conductance'):
        return

    snap = _get_numeric_snapshot(case)
    if snap is None:
        warnings.warn('Case missing numerical snapshot; skipping conductance plot.', RuntimeWarning, stacklevel=2)
        return

    times, flux, Tw_face, Ts_face, seed_time = _extract_history(snap)
    if not flux:
        warnings.warn('Numerical snapshot has no flux history; skipping conductance plot.', RuntimeWarning, stacklevel=2)
        return

    heff: List[float] = []
    if Tw_face and Ts_face and len(Tw_face) == len(flux):
        for q, Tw, Ts in zip(flux, Tw_face, Ts_face):
            delta = Tw - Ts
            if abs(delta) > 1e-12:
                heff.append(q / delta)
            else:
                heff.append(float('nan'))
    else:
        hc = 1.0 / R_c if R_c else math.inf
        heff = [hc for _ in flux]

    fig, ax = plt.subplots(num=f"h_eff(t) — {case['label']}")
    ax.grid(True)
    ax.plot(times, heff, '-', linewidth=1.6, color=(0.2, 0.6, 0.2), label='Explicit $h_{eff}$')
    hc_true = 1.0 / R_c if R_c else math.inf
    ax.axhline(hc_true, color='r', linestyle='--', linewidth=1.2, label='Nominal $1/R_c$')
    if seed_time > 0.0:
        ax.axvline(seed_time, color='k', linestyle=':', linewidth=1.0, label='Seed time')

    ax.set_xlabel('time  t  [s]')
    ax.set_ylabel(r'effective conductance $h_{eff}$ [W m$^{-2}$ K$^{-1}$]')
    ax.set_title(f"Explicit contact conductance — {case['label']}")
    ax.legend(loc='best')
    ax.set_xlim(0.0, max(t_max, times[-1] if times else t_max))


def plot_flux(case: Dict[str, object], _R_c: float, t_max: float | None = None) -> None:
    """Plot the explicit interface heat flux history."""

    if not _ensure_matplotlib('interface flux'):
        return

    snap = _get_numeric_snapshot(case)
    if snap is None:
        warnings.warn('Case missing numerical snapshot; skipping flux plot.', RuntimeWarning, stacklevel=2)
        return

    times, flux, _Tw_face, _Ts_face, seed_time = _extract_history(snap)
    if not flux:
        warnings.warn('Numerical snapshot has no flux history; skipping flux plot.', RuntimeWarning, stacklevel=2)
        return

    if t_max is None:
        t_max = max(times) if times else 0.1
    t_max = max(t_max, max(times) if times else 0.1)

    fig, ax = plt.subplots(num=f"Interface flux vs time — {case['label']}")
    ax.grid(True)
    ax.plot(times, flux, '-', linewidth=1.6, color=(0.95, 0.65, 0.2), label='Explicit flux')
    if seed_time > 0.0:
        ax.axvline(seed_time, color='k', linestyle=':', linewidth=1.0, label='Seed time')

    ax.set_xlabel('time  t  [s]')
    ax.set_ylabel(r'interface heat flux  q  [W m$^{-2}$]')
    ax.set_title(f"Explicit interface flux — {case['label']}")
    ax.legend(loc='best')
    ax.set_xlim(0.0, t_max)


def _get_numeric_snapshot(case: Dict[str, object]) -> Dict[str, object] | None:
    snap = case.get('num')
    if isinstance(snap, dict) and snap:
        return snap
    return None


def _extract_history(snap: Dict[str, object]) -> Tuple[List[float], List[float], List[float], List[float], float]:
    q_struct = snap.get('q') if isinstance(snap, dict) else None
    if isinstance(q_struct, dict):
        if 't_phys' in q_struct:
            times = [float(val) for val in q_struct['t_phys']]
        else:
            times = [float(val) for val in q_struct.get('t', [])]
            t_offset = float(snap.get('t_offset', 0.0))
            times = [val + t_offset for val in times]
        flux = [float(val) for val in q_struct.get('val', [])]
        Tw_face = [float(val) for val in q_struct.get('Tw_face', [])]
        Ts_face = [float(val) for val in q_struct.get('Ts_face', [])]
    else:
        times = []
        flux = []
        Tw_face = []
        Ts_face = []
    seed_time = 0.0
    seed = snap.get('seed') if isinstance(snap, dict) else None
    if isinstance(seed, dict) and 'time' in seed:
        seed_time = float(seed['time'])
    return times, flux, Tw_face, Ts_face, seed_time


def _get_params(case: Dict[str, object]) -> Dict[str, float]:
    params = case.get('params')
    if isinstance(params, dict):
        return {key: float(params[key]) for key in params if isinstance(params[key], (int, float))}
    return {}


def _ensure_matplotlib(purpose: str) -> bool:
    if plt is not None:
        return True
    warnings.warn(
        f"matplotlib is required to render the {purpose} plot; skipping graphical output.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False
