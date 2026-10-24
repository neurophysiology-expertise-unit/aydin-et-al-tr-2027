# Event-gated output coupling in excitable lattices: simulator, pre-registered analyses and figure

This repository regenerates the results and Figure 1 of the manuscript on front propagation in a lattice of
FitzHugh–Nagumo cells with event-gated output coupling (a cell's output gain is set when it fires; refractory clamp;
decremental release), and runs the model. It contains the simulator (numba), the eight pre-registered analysis
scripts, their stored results (`processed_data/*.npz`) and the figure script.

## Install and self-test

    conda env create -f environment.yml && conda activate excitable-lattice
    pytest -q tests                        # about 1 min: integrators bit for bit + one operating-window check
    python -s paperC_model.py              # run_gen reproduces stored results in processed_data/

`requirements.txt` lists the minimum versions for a pip install; `environment.yml` pins the versions used for every
stored result.

## The simulator (all functions return per-cell first-firing times `t_rise`)

| function (module) | lattice | what it adds to `run_gen` |
|---|---|---|
| `run_gen` (`paperC_model`) | 2-D, per-cell parameters | the reference integrator |
| `run_chain` (`paperC_continuum`) | 1-D row reduction of a planar front, any symmetric offset stencil | time step argument |
| `run_nbr` (`paperC_edges`) | 1-D row reduction with a per-row neighbour table (e.g. honeycomb) | neighbour table |
| `run_pde` (`paperC_pde`) | 1-D continuum equation, FD2/FD4/FD6 central stencils, grid dx | settling release rule (`recompute`) |
| `run_gx` (`paperC_design`) | as `run_pde` | pre-firing gain `g0`, binary output (`mode = 1`) |
| `run_gx8` (`paperC_template`) | as `run_pde` | ungated output (`mode = 2`) |
| `run_gen2` (`paperC_tolerance`) | 2-D, per-cell parameters | settling rule, additive noise |
| `run_gen3` (`paperC_tolerance2`) | 2-D, per-cell parameters | `run_gen2` + early stop once the scored cells fired |

Every variant is asserted bit for bit against the integrator it extends before it is used (the gate functions in
each script and `tests/test_simulator.py`). `recompute = False` is the original update (a newly fired cell is
integrated one step without its own drain); `recompute = True` is the settling rule.

## Usage

Axis planar front on the 1-D row reduction, settling rule (run from the repository root):

    import sys; sys.path.insert(0, ".")
    from paperC_design import chain, cont
    ext, ok = chain(8.0, dtf=32)          # coupling d = 8, time step DT/32, 120 s: ok = True (inside the window)
    ext, ok = chain(16.0, dtf=32)         # d = 16: ok = False (above the upper edge)
    ext, ok = cont(8.0, dx=1/8, dtdiv=1)  # continuum equation, FD4, grid 1/8 cell: no front (ok = False)

`ext` is the front's extent in rows (cells) beyond the seed; `ok` is the propagation criterion (40 rows in 120 s for
`chain`, 20 cells in 60 s for `cont`). For a 2-D lattice with per-cell parameters, mismatch or noise, call
`paperC_tolerance.run_gen2` with arrays from `paperC_model.hom_params` (see `paperC_tolerance.one`).

## Pre-registered analyses

Each script's docstring holds its predictions, method and pass criteria; failed predictions are reported as failed.
Each result file is written by `core.provenance.save_result` and carries its parameters and outcomes
(`__params__`, JSON) and the commit of the code that produced it (`__git_commit__`). The outcomes below are read
from `__params__`.

| round | script → result | tests | passed | failed |
|---|---|---|---|---|
| 1 | `paperC_continuum.py` → `paperC_continuum.npz` | C1–C3, A0–A2 | C3, A0, A1, A2 | C1, C2 |
| 2 | `paperC_edges.py` → `paperC_edges.npz` | E1, E2, S1, S2, L0, W0, W1, H1, A1′ | S1, L0, W0, W1 | E1, E2, S2, H1, A1′ |
| 3 | `paperC_dt.py` → `paperC_dt.npz` | T1–T3 | — | T1, T2, T3 |
| 4 | `paperC_pde.py` → `paperC_pde.npz` | P1, P3a, P3b | P1, P3b | P3a |
| 5 | `paperC_design.py` → `paperC_design.npz` | R1, R2, J1–J6, A-static, A-race | R1, R2, J6 | J1–J5, A-static, A-race |
| 6 | `paperC_tolerance.py` → `paperC_tolerance.npz` | M1–M3 | — | M1, M2, M3 |
| 7 | `paperC_tolerance2.py` → `paperC_tolerance2.npz` | gate R; W, N, CE, S1–S3 (criteria i, ii) | R, N | W, CE, S1i, S1ii, S2i, S2ii, S3i, S3ii |
| 8 | `paperC_template.py` → `paperC_template.npz` | R8, T1, T2, SP | R8, T1, T2, SP | — |

Re-running a round: `python -s paperC_<round>.py` (multiprocessing pool of 11 workers; checkpoints in
`/tmp/paperC_*_ckpt`). Each script first asserts its integrator gate and
`paperC_model.check()`. Docstrings refer by name to earlier exploratory analyses (for example `paperC_direction`)
that informed some predictions; those analyses are not part of this release.

## Figure 1

    python -s paperC_fig_window.py

reads only the stored `processed_data/*.npz`, runs no model, and writes `Figure_window.{pdf,svg,png}` and
`Figure_window_stats.csv` (every plotted value, long format). The style file `neuro_paper.mplstyle` is loaded from
the script's directory.

## Provenance

These scripts are the pre-registered scripts; the commit history recording each pre-registration before its run is
kept in a separate repository available to the editor on request. The `__git_commit__` stamps in the result files
refer to that repository.

## Credit and licence

The original lattice integrators were written by F. K. Bayat and F. Oktay; the integrators here are verbatim copies
of them with the documented additions listed above. MIT licence (`LICENSE`). Citation metadata: `CITATION.cff`.
