"""
paperC_dt.py — paper C, round 3. Written AFTER paperC_edges.npz (commit f9303df) and BEFORE any run of this script.

Why: round 2 showed the strong-coupling threshold of the axis planar front is NOT converged in the time step:
d*_lat = 14.65 (DT, round 1), 13.55 (DT/2), 12.90 (DT/4), 12.39 (DT/8) (paperC_edges E, every seed width
identical), 12.006 (DT/16, round 1 k = 4) — the decrements shrink only by a factor ~0.6-0.8 per halving (E1 FAIL).
Every comparison between lattices (A1, A1', H1) is therefore confounded by stencil-dependent time-step error. The
round-2 FAILs (E1, E2, S2, H1, A1') stand as recorded; this round asks what holds in the limit dt -> 0.

Same model, integrator and measures as paperC_edges (run_nbr, asserted bitwise against run_chain; sim()):
homogeneous, gamma_regen = 1, noise off, tau_ref 15 s, T = 120 s, extent from the seed edge, PROPAGATES := extent >=
P rows; grid then 8-step bisection, each bisection inside a bracket from its own grid.
Sizes: N = 256 x max offset rows, seed w = 5 x max offset (seed width has no effect: paperC_edges E),
P = 40 x max offset rows.
 Axis chain at dt = DT/16, DT/32, DT/64, DT/128; grid d_lat G_T = {8 9 10 10.5 11 11.5 12 12.5 13 14 15}.
 Offset sets {1,2}, {2,3}, {1,1,2}, {1,2,3,4} at dt = DT/64; grid G_T x 2/s.
 Honeycomb zigzag at dt = DT/64; grid G_T x 2/3 plus {4, 5}.

PRE-REGISTERED PREDICTIONS
 T1 (a finite threshold exists as dt -> 0): |d*(DT/128) - d*(DT/64)| / d*(DT/64) <= 1 %, and d*(DT/128) >= 8.
    (Supporting, not a test: round 1, k = 8 at DT/64 showed no propagation at d_lat >= 12.)
 T2 (counting law in the limit): |s d*/2 - d*_ax| / d*_ax <= 5 % for each of {1,2}, {2,3}, {1,1,2}, {1,2,3,4},
    with d*_ax = axis at DT/64.
 T3 (honeycomb zigzag, s = 3): |d*_hc - (2/3) d*_ax| / ((2/3) d*_ax) <= 5 %, at DT/64.
 Descriptive: fit d*(dt) = d0 + c dt^p to the axis thresholds at DT/2 ... DT/128 (DT/2-DT/8 from paperC_edges,
 w = 5), reported with the residuals; not a test.
Failed predictions are reported as failed; criteria are not revised after the data land.
Output: processed_data/paperC_dt.npz. Checkpoints /tmp/paperC_dt_ckpt.   Run: python -s paperC_dt.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from scipy.optimize import curve_fit
from core.model import DT
from core.provenance import save_result
import paperC_model
from paperC_continuum import SETS, NBIS
from paperC_edges import run, bisect_job, bracket, assert_equivalence

CKPT = Path("/tmp/paperC_dt_ckpt")
G_T = [8, 9, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14, 15]
AX_DTF = [16, 32, 64, 128]
T_SETS = ["1,2", "2,3", "1,1,2", "1,2,3,4"]


def spec(lattice, dtf):
    mx = 1 if lattice in ("axis", "honeycomb") else max(SETS[lattice])
    return (lattice, dtf, 5 * mx, 120.0, 40 * mx, 256 * mx)


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def main():
    CKPT.mkdir(exist_ok=True)
    paperC_model.check()
    assert_equivalence()
    grids = {f"ax_dt{f}": (spec("axis", f), G_T) for f in AX_DTF}
    for k in T_SETS:
        s = 2 * len(SETS[k]); grids[f"set_{k}"] = (spec(k, 64), [x * 2 / s for x in G_T])
    grids["honeycomb"] = (spec("honeycomb", 64), [4, 5] + [x * 2 / 3 for x in G_T])
    jobs = [(n, (sp, x)) for n, (sp, g) in grids.items() for x in g]
    pool = Pool(11)
    R = ck("grids", lambda: pool.map(run, [j[1] for j in jobs], chunksize=1))
    res = {}
    for (n, _), r in zip(jobs, R):
        res.setdefault(n, []).append(r)
    for n, (sp, g) in grids.items():
        print(f"{n:12s} " + " ".join(f"{x:.3g}:{r[0]:.0f}{'+' if r[1] else '.'}" for x, r in zip(g, res[n])), flush=True)
    bj = {n: (sp, *bracket(g, [r[1] for r in res[n]], True), True) for n, (sp, g) in grids.items()
          if bracket(g, [r[1] for r in res[n]], True) is not None}
    missing = [n for n in grids if n not in bj]
    print("no bracket:", missing, flush=True)
    B = ck("bisect", lambda: dict(zip(bj, pool.map(bisect_job, list(bj.values()), chunksize=1))))
    pool.close()
    thr = {n: 0.5 * (lo + hi) for n, (lo, hi) in B.items()}
    for n in grids:
        if n in B:
            print(f"  {n:12s} ({B[n][0]:.5f}, {B[n][1]:.5f}]", flush=True)
    g_ = lambda n: thr.get(n, np.nan)
    rel = lambda a, b: abs(a - b) / b
    T1 = bool(rel(g_("ax_dt128"), g_("ax_dt64")) <= 0.01 and g_("ax_dt128") >= 8)
    ax = g_("ax_dt64")
    dev = {k: (2 * len(SETS[k]) * g_(f"set_{k}") / 2 - ax) / ax for k in T_SETS}
    T2 = bool(all(abs(v) <= 0.05 for v in dev.values()))
    dev_hc = g_("honeycomb") / (2 / 3 * ax) - 1
    T3 = bool(abs(dev_hc) <= 0.05)
    # descriptive fit with round-2 points (paperC_edges, w = 5)
    ed = np.load(Path(__file__).resolve().parent / "processed_data" / "paperC_edges.npz", allow_pickle=True)
    th2 = json.loads(str(ed["__params__"]))["thr"]
    dts = np.array([DT / 2, DT / 4, DT / 8] + [DT / f for f in AX_DTF])
    ds = np.array([th2["E_w5_dt2"], th2["E_w5_dt4"], th2["E_w5_dt8"]] + [g_(f"ax_dt{f}") for f in AX_DTF])
    fit = [np.nan] * 3
    try:
        p, _ = curve_fit(lambda x, d0, c, q: d0 + c * x ** q, dts, ds, p0=[ds[-1], 100.0, 1.0], maxfev=20000)
        fit = [float(v) for v in p]
    except Exception as e:
        print("fit failed:", e)
    print("axis d*(dt): " + " ".join(f"DT/{DT/x:.0f}:{y:.4f}" for x, y in zip(dts, ds)), flush=True)
    print(f"fit d0 {fit[0]:.4f}  c {fit[1]:.4g}  p {fit[2]:.3f}", flush=True)
    print("counting deviations at DT/64: " + " ".join(f"{k}: {100*v:+.2f}%" for k, v in dev.items()), flush=True)
    print(f"honeycomb {g_('honeycomb'):.4f} vs (2/3) axis {2/3*ax:.4f}: {100*dev_hc:+.2f}%", flush=True)
    print(f"T1 {T1}  T2 {T2}  T3 {T3}", flush=True)
    out = save_result(Path(__file__).resolve().parent / "processed_data" / "paperC_dt.npz",
                      {"model": "paperC_edges.run_nbr (asserted bitwise vs run_chain)", "T1": T1, "T2": T2, "T3": T3,
                       "thr": thr, "brackets": B, "dev_counting": dev, "dev_honeycomb": dev_hc, "fit_d0_c_p": fit,
                       "no_bracket": missing, "grids": {n: [sp, g] for n, (sp, g) in grids.items()}},
                      ax_dt=dts, ax_thr=ds, fit=np.array(fit))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
