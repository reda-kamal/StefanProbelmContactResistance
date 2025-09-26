"""Python driver for the three-domain Stefan problem with contact resistance."""
from __future__ import annotations

import pathlib
import sys

try:
    from .run_vam_case import run_vam_case
    from .plotting import (
        plot_conductance,
        plot_diff_profile,
        plot_flux,
        plot_profiles,
        plot_front_history,
        plot_variable_contact_flux,
        plot_variable_contact_resistance,
    )
except ImportError:  # pragma: no cover - support running directly via PyCharm/CLI
    if __package__ is None or __package__ == "":
        package_dir = pathlib.Path(__file__).resolve().parent
        if str(package_dir) not in sys.path:
            sys.path.append(str(package_dir))
    from run_vam_case import run_vam_case  # type: ignore[no-redef]
    from plotting import (  # type: ignore[no-redef]
        plot_conductance,
        plot_diff_profile,
        plot_flux,
        plot_profiles,
        plot_front_history,
        plot_variable_contact_flux,
        plot_variable_contact_resistance,
    )


def main(show_plots: bool = True) -> None:
    t_phys = 0.02
    R_c = 2e-5

    explicit_opts = {
        'CFL': 0.30,
        'wall': {'length': 7.5e-3, 'cells': 220},
        'fluid': {'length': 8.0e-3, 'cells': 320, 'min_cells': 220},
        'min_seed_cells': 1,
        'history_dt': 5.0e-4,
        'flux_smoothing': 5,
        'nsave': 4000,
        'refine': {
            'max_iters': 3,
            'factor': 1.5,
            'cfl_shrink': 0.75,
            'tol_abs_T': 3.0,
            'tol_rel_T': 0.01,
            'tol_abs_q': 200.0,
            'tol_rel_q': 0.01,
            'history_shrink': 0.75,
            'min_CFL': 0.05,
        },
    }
    sim_opts_water = {
        'explicit': {**explicit_opts},
        'profile_pts_per_seg': 400,
    }
    sim_opts_tin = {
        'explicit': {**explicit_opts, 'variable_contact': False},
        'profile_pts_per_seg': 400,
    }

    k_w = 40.0
    rho_w = 3980.0
    c_w = 750.0

    A = {
        'k_s': 2.22, 'rho_s': 917.0, 'c_s': 2100.0,
        'k_l': 0.6,  'rho_l': 998.0, 'c_l': 4180.0,
        'L': 333.5e3,
        'Tf': 0.0, 'Tw_inf': -15.0, 'Tl_inf': -15.0,
    }

    B = {
        'k_s': 66.0, 'rho_s': 7310.0, 'c_s': 230.0,
        'k_l': 31.0, 'rho_l': 6980.0, 'c_l': 300.0,
        'L': 59.2e3,
        'Tf': 231.93, 'Tw_inf': 50.0, 'Tl_inf': 226.93,
    }

    caseA = run_vam_case('Water/Ice + Sapphire', k_w, rho_w, c_w, A, R_c, t_phys, sim_opts_water)
    caseB = run_vam_case('Tin (liq/sol) + Sapphire', k_w, rho_w, c_w, B, R_c, t_phys, sim_opts_tin)

    cases = (caseA, caseB)
    for case in cases:
        print(f"=== {case['label']} ===")
        params = case['params']
        print(f"lambda = {params['lam']:.5f}, Ti = {params['Ti']:.3f} C")
        snap = case['num']['explicit']
        q_hist = snap['q']['val']
        t_hist = snap['q']['t']
        final_q = q_hist[-1] if q_hist else float('nan')
        final_t = t_hist[-1] if t_hist else float('nan')
        print(f"  explicit  : S = {snap['S']:.6f} m at t={final_t:.4f} s, q={final_q:.2f} W/m^2")
        print()

    for case in cases:
        plot_profiles(case)
        plot_diff_profile(case)
        plot_conductance(case, R_c, 0.1)
        plot_flux(case, R_c, 0.1)
        plot_front_history(case)
        plot_variable_contact_flux(case, 0.1)
        plot_variable_contact_resistance(case)

    if show_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("matplotlib not available; skipping interactive figures.")
        else:
            plt.show()

    print("Simulation complete. Inspect case dictionaries for detailed data.")


if __name__ == "__main__":
    main()
