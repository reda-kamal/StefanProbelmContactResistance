# Python Stefan Solver

This directory provides a pure-Python translation of the MATLAB tools used in the
three-domain Stefan problem with contact resistance.  The modules avoid third-party
packages so they can run in restricted environments.

## Running the demo

From the repository root run:

```bash
python -m python.main
```

The script executes the water/ice and tin/sapphire scenarios with a coarse grid so
it finishes quickly and prints the interface locations and contact heat flux for the
explicit and enthalpy solvers.  The dictionaries returned by `run_vam_case` contain
all intermediate fields if deeper analysis is required.
