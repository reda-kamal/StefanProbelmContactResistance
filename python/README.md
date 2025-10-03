# Python Stefan Solver

This directory provides a pure-Python translation of the MATLAB tools used in the
three-domain Stefan problem with contact resistance.  The modules avoid third-party
packages so they can run in restricted environments.

## Running the demo

The plotting helpers rely on `matplotlib` for figure creation.  If it is not
available the solver still runs, but it will emit a warning and skip the figures.

From the repository root you can either execute the package module:

```bash
python -m python.main
```

or run the script directly (useful for IDEs such as PyCharm):

```bash
python python/main.py
```

Both entry points execute the water/ice and tin/sapphire scenarios, emit a short
text summary, and open four matplotlib figures per case (explicit temperature
profiles, deviation from the fusion temperature, effective conductance inferred
from the flux history, and interface heat flux).  The dictionaries returned by
`run_vam_case` contain the explicit snapshot and basic material properties for
further analysis.
