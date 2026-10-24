"""
paperC_edges.py — paper C, round 2.
Written AFTER paperC_continuum.npz (commit 24ee702) and BEFORE any run of this script.
Same model and protocol as paperC_continuum (homogeneous lattice, gamma_regen = 1, noise off, tau_ref 15 s,
hom_params; planar fronts through their exact 1-D reduction). Thresholds by grid then 8-step bisection; unlike
round 1, every bisection uses a grid computed at ITS OWN dt and seed (round 1 bisected the DT/2 runs inside the DT
bracket, so its DT/2 values (14.0000, 14.0039] sit on the bracket floor and only bound the threshold from above).

What round 1 established (paperC_continuum.npz): C3 PASS — no planar front propagates above the strong-coupling
threshold at any refinement k = 1..8, up to d_lat = 2048; C1 FAIL — at k = 1 the threshold moves from 14.65 (DT) to
<= 14.00 (DT/2, DT/4); C2 FAIL — d*_lat 12.90 (k 2), 12.01 (k 4); A0, A1, A2 PASS.
Open after round 1, and tested here: (E) is there a dt-converged, seed-independent lattice threshold? (S) how does
the front fail at each edge of the propagating window? (W) does the counting law hold at the weak-coupling edge?
(H) does counting predict a lattice not yet tested (honeycomb)? (A1') does counting hold at converged dt?

INTEGRATOR. run_nbr = paperC_continuum.run_chain with the uniform offset stencil replaced by a per-row neighbour
table nbr[i, :] (-1 = no link) and a per-row same-row count m_same[i]; summation order kept (table order, then the
m same-row terms, then -z_i*b). main() ASSERTS bit-for-bit t_rise equality with run_chain for o = {+1,-1}, m = 2 and
o = {+1,-1,+2,-2}, m = 0 (at d 8 and 14) before anything runs; paperC_model.check() must pass too.
Lattices (row reductions): axis chain {+-1}, m = 2 (s = 2); offset sets as in round 1 (m = 0, s = 2 x #offsets);
HONEYCOMB, front normal to a bond (zigzag rows): even row n: n+1 once, n-1 twice; odd row n: n-1 once, n+1 twice
(z = 3, m = 0, s = 3). (Honeycomb armchair rows reduce to b[n+1] + b[n-1] + b[n] - 3 b[n], the axis chain itself, so
it is not a test.)
MEASURE: extent from the seed edge (rows); PROPAGATES := extent >= P rows by T. Speed (rows/s) := 1 / slope of the
least-squares line t_rise vs distance over fired rows with 10 <= distance <= 40.

E. Converged lattice threshold, axis chain, N = 512, T = 120 s, P = 40 rows. dt in {DT/2, DT/4, DT/8} x seed width
   w in {5, 10, 20, 40} rows; grid d_lat {8 9 10 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15 16}, then bisection.
   E1 (dt converged): for every w, |d*(DT/8) - d*(DT/4)| / d*(DT/4) <= 1 %.
   E2 (seed-independent limit; direction informed by round 1, where wider seeds failed earlier):
       d* falls with w from 5 to 20 and saturates: |d*(40) - d*(20)| / d*(20) <= 2 % at DT/8.
   d*_inf := d*(w = 40, DT/8).
S. How the front fails at each edge (axis chain, w = 40). eps in {0.3 0.2 0.1 0.05 0.02 0.01 0.005 0.002}.
   S1 (strong edge: abrupt failure at finite speed; informed by the stored paperC_direction speeds, which RISE with d
       up to the failure): at d = d*_inf (1 - eps), dt = DT/8, T = 120 s: v(eps = 0.002) >= 0.5 v(eps = 0.3).
   Weak edge: d_low by grid {0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1 1.5 2} and bisection, dt = DT, T = 1200 s,
       P = 40 rows. L0 (dt): d_low at DT/2 within 1 % of d_low at DT.
   S2 (weak edge: speed vanishes continuously, as at classical pinning): at d = d_low (1 + eps), dt = DT, T = 2400 s:
       v(eps = 0.002) / v(eps = 0.3) <= 0.3 AND the fit v = A (d - d_c)^beta (A, d_c, beta free) gives beta in
       [0.3, 0.7].
W. Weak edge vs direction (dt = DT, T = 1200 s, P = 40 x max offset rows, w = 20 x max offset): sets {1,1}
   (control), {1,2}, {1,3}, {2,3}, {1,1,2}; low grid divided by s/2, then bisection.
   W0 (method): {1,1} gives s d_low/2 = d_low(axis, w 20) within 1 %.
   W1 (counting does NOT hold at the weak edge, as classical pinning is direction-dependent): at least one of
       {1,2}, {1,3}, {2,3} deviates from s d_low/2 = d_low(axis) by more than 10 %.
H. Honeycomb zigzag front, strong edge. dt = DT/4, w = 20, N = 512, T = 120 s, P = 40; grid d_lat {6 7 8 8.5 9
   9.5 10 10.5 11 12 13 14}, bisection.
   H1 (counting, s = 3): d*_hc = (2/3) d*(axis, w 20, DT/4) within 5 %.
A1'. Counting at converged dt: sets {1,2}, {2,3}, {1,1,2}, {1,2,3,4} at dt = DT/4, w = 20 x max offset, N = 512 x max
   offset, P = 40 x max offset, T = 120 s; grid as round 1 (G_A), bisection.
   A1' : |s d*/2 - d*(axis, w 20, DT/4)| / d*(axis, w 20, DT/4) <= 5 % for all four.
Failed predictions are reported as failed; criteria are not revised after the data land.
Output: processed_data/paperC_edges.npz (provenance-stamped). Checkpoints /tmp/paperC_edges_ckpt.
Run: python -s paperC_edges.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from scipy.optimize import curve_fit
from core.model import DT, ETA, A_FHN, B_FHN
from core.provenance import save_result
import paperC_model
from paperC_model import hom_params, THETA, TAU_REF, C_DOWN, K_REF, SCALE, G_FLOOR
from paperC_continuum import run_chain, chain_setup, SETS, G_A, NBIS

CKPT = Path("/tmp/paperC_edges_ckpt")
WS = [5, 10, 20, 40]
DTF = [2, 4, 8]
G_E = [8, 9, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 16]
G_LOW = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2]
G_H = [6, 7, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14]
EPS = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
W_SETS = ["1,1", "1,2", "1,3", "2,3", "1,1,2"]
A_SETS = ["1,2", "2,3", "1,1,2", "1,2,3,4"]


@njit(fastmath=True, cache=False)
def run_nbr(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, nbr, m_same, dt):
    # paperC_continuum.run_chain, verbatim except the per-row neighbour table nbr (-1 = none) and per-row m_same
    nx = I0.shape[0]; no = nbr.shape[1]
    z = np.zeros(nx)
    for i in range(nx):
        c = 0
        for k in range(no):
            if nbr[i, k] >= 0:
                c += 1
        z[i] = float(c + m_same[i])
    C = np.full(nx, C_DOWN); h = (C + A_FHN) / B_FHN
    g = np.zeros(nx); reft = np.zeros(nx); prev = np.zeros(nx)
    for i in range(nx):
        if seedmask[i]:
            C[i] = 1.5; g[i] = 1.0
    prev = (0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5).astype(np.float64)
    t_rise = np.full(nx, -1.0)
    lap = np.empty(nx)
    for t in range(steps):
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        active = (C_active > 0.5).astype(np.float64)
        inref = (reft > 0.0).astype(np.float64)
        broadcast = C_active * g * (1.0 - inref)
        for i in range(nx):
            acc = broadcast[nbr[i, 0]]
            for k in range(1, no):
                if nbr[i, k] >= 0:
                    acc = acc + broadcast[nbr[i, k]]
            for k in range(m_same[i]):
                acc = acc + broadcast[i]
            lap[i] = acc - z[i] * broadcast[i]
        coupling_in = Deff * lap
        rose = active * (1.0 - prev)
        fell = prev * (1.0 - active)
        g_new = np.minimum(1.0, G_FLOOR + gamma_regen * (coupling_in / SCALE))
        g = np.where(rose > 0.5, np.maximum(g_new, 0.0), g)
        reft = np.where(fell > 0.5, tau_ref, reft)
        clamp = K_REF * inref * (C - C_DOWN)
        noise = np.zeros(nx)
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + dt * dC + np.sqrt(dt) * noise
        h = h + dt * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - dt, 0.0)
        prev = active
        for i in range(nx):
            if rose[i] > 0.5 and t_rise[i] < 0.0:
                t_rise[i] = t * dt
    return t_rise


def table(lattice, N):
    """(nbr, m_same) for a row reduction."""
    if lattice == "axis":
        offs, m = [1, -1], 2
    elif lattice == "honeycomb":
        nbr = np.full((N, 3), -1, np.int64)
        for i in range(N):
            if i % 2 == 0:
                nbr[i] = [(i + 1) % N, (i - 1) % N, (i - 1) % N]
            else:
                nbr[i] = [(i - 1) % N, (i + 1) % N, (i + 1) % N]
        return nbr, np.zeros(N, np.int64)
    else:
        offs, m = [s * x for x in SETS[lattice] for s in (1, -1)], 0
    nbr = np.array([[(i + o) % N for o in offs] for i in range(N)], np.int64)
    return nbr, np.full(N, m, np.int64)


def maxoff(lattice):
    return 1 if lattice in ("axis", "honeycomb") else max(SETS[lattice])


def sim(lattice, d, dtf, w, T, P, N):
    """returns (extent rows, propagates, speed rows/s)."""
    sm, dist = chain_setup(N, w)
    nbr, ms = table(lattice, N)
    I0, th, De, gd = (a.ravel() for a in hom_params((N, 1), d))
    dt = DT / dtf
    tr = run_nbr(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T / dt)), nbr, ms, dt)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    band = fired & (dist >= 10) & (dist <= 40)
    v = np.nan
    if band.sum() >= 20:
        sl = np.polyfit(dist[band], tr[band], 1)[0]
        v = 1.0 / sl if sl > 0 else np.nan
    return ext, bool(ext >= P), v


# job specs: (lattice, dtf, w, T, P, N); run(spec, d)
def spec_E(w, dtf): return ("axis", dtf, w, 120.0, 40, 512)
def spec_low(dtf=1): return ("axis", dtf, 20, 1200.0, 40, 512)
def spec_W(key): mx = max(SETS[key]); return (key, 1, 20 * mx, 1200.0, 40 * mx, 512 * mx)
def spec_H(): return ("honeycomb", 4, 20, 120.0, 40, 512)
def spec_A(key): mx = max(SETS[key]); return (key, 4, 20 * mx, 120.0, 40 * mx, 512 * mx)


def run(args):
    spec, d = args
    return sim(spec[0], d, *spec[1:])


def bisect_job(args):
    """upper=True: lo propagates, hi fails (strong edge); upper=False: lo fails, hi propagates (weak edge)."""
    spec, lo, hi, upper = args
    for _ in range(NBIS):
        mid = 0.5 * (lo + hi)
        p = run((spec, mid))[1]
        if p == upper:
            lo = mid
        else:
            hi = mid
    return [lo, hi]


def bracket(grid, prop, upper):
    ip = [i for i, p in enumerate(prop) if p]
    if not ip:
        return None
    if upper:
        i = ip[-1]
        return (grid[i], grid[i + 1]) if i + 1 < len(grid) else None
    i = ip[0]
    return (grid[i - 1], grid[i]) if i > 0 else None


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def assert_equivalence():
    N, w = 256, 5
    sm, _ = chain_setup(N, w)
    for offs, m in (([1, -1], 2), ([1, -1, 2, -2], 0)):
        nbr = np.array([[(i + o) % N for o in offs] for i in range(N)], np.int64)
        for d in (8.0, 14.0):
            I0, th, De, gd = (a.ravel() for a in hom_params((N, 1), d))
            a = run_chain(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, 20000, np.array(offs, np.int64), m, DT)
            b = run_nbr(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, 20000, nbr, np.full(N, m, np.int64), DT)
            assert np.array_equal(a, b), ("run_nbr != run_chain", offs, d)
    print("run_nbr reproduces paperC_continuum.run_chain bit for bit", flush=True)


def main():
    CKPT.mkdir(exist_ok=True)
    paperC_model.check()
    assert_equivalence()
    pool = Pool(11)
    rel = lambda a, b: abs(a - b) / b
    # ---------- stage 1: all grids ----------
    grids = {}
    for w in WS:
        for f in DTF:
            grids[f"E_w{w}_dt{f}"] = (spec_E(w, f), G_E, True)
    for f in (1, 2):
        grids[f"low_dt{f}"] = (spec_low(f), G_LOW, False)
    for key in W_SETS:
        s = 2 * len(SETS[key])
        grids[f"W_{key}"] = (spec_W(key), [x * 2 / s for x in G_LOW], False)
    grids["H"] = (spec_H(), G_H, True)
    for key in A_SETS:
        grids[f"A_{key}"] = (spec_A(key), G_A, True)
    jobs = [(name, (sp, x)) for name, (sp, g, up) in grids.items() for x in g]
    R = ck("grids", lambda: pool.map(run, [j[1] for j in jobs], chunksize=1))
    res = {}
    for (name, (sp, x)), r in zip(jobs, R):
        res.setdefault(name, []).append(r)
    for name, (sp, g, up) in grids.items():
        print(f"{name:12s} " + " ".join(f"{x:g}:{r[0]:.0f}{'+' if r[1] else '.'}" for x, r in zip(g, res[name])),
              flush=True)
    # ---------- stage 2: bisections ----------
    bjobs = {name: (sp, *bracket(g, [r[1] for r in res[name]], up), up)
             for name, (sp, g, up) in grids.items() if bracket(g, [r[1] for r in res[name]], up) is not None}
    missing = [n for n in grids if n not in bjobs]
    print("no bracket (not bisected):", missing, flush=True)
    B = ck("bisect", lambda: dict(zip(bjobs, pool.map(bisect_job, list(bjobs.values()), chunksize=1))))
    thr = {n: 0.5 * (lo + hi) for n, (lo, hi) in B.items()}
    for n in grids:
        if n in B:
            print(f"  {n:12s} ({B[n][0]:.5f}, {B[n][1]:.5f}]", flush=True)
    g_ = lambda n: thr.get(n, np.nan)
    E1 = bool(all(rel(g_(f"E_w{w}_dt8"), g_(f"E_w{w}_dt4")) <= 0.01 for w in WS))
    E2 = bool(g_("E_w5_dt8") > g_("E_w10_dt8") > g_("E_w20_dt8") and rel(g_("E_w40_dt8"), g_("E_w20_dt8")) <= 0.02)
    dinf = g_("E_w40_dt8")
    L0 = bool(rel(g_("low_dt2"), g_("low_dt1")) <= 0.01)
    dlow = g_("low_dt1")
    # ---------- stage 3: speeds near both edges ----------
    sj = [(spec_E(40, 8), dinf * (1 - e)) for e in EPS] + [(("axis", 1, 20, 2400.0, 40, 512), dlow * (1 + e)) for e in EPS]
    SR = ck("speeds", lambda: pool.map(run, sj, chunksize=1))
    pool.close()
    vh = np.array([r[2] for r in SR[:len(EPS)]]); vl = np.array([r[2] for r in SR[len(EPS):]])
    print("strong edge: " + " ".join(f"eps {e}: v {v:.4f}" for e, v in zip(EPS, vh)), flush=True)
    print("weak edge:   " + " ".join(f"eps {e}: v {v:.4f}" for e, v in zip(EPS, vl)), flush=True)
    S1 = bool(np.isfinite(vh[-1]) and vh[-1] >= 0.5 * vh[0])
    beta = np.nan
    dd = dlow * (1 + np.array(EPS)); ok = np.isfinite(vl)
    if ok.sum() >= 4:
        try:
            p, _ = curve_fit(lambda x, A, dc, b: A * np.clip(x - dc, 1e-12, None) ** b, dd[ok], vl[ok],
                             p0=[1.0, dlow * 0.999, 0.5], maxfev=20000)
            beta = float(p[2])
        except Exception as e:
            print("fit failed:", e)
    S2 = bool(np.isfinite(vl[-1]) and vl[-1] / vl[0] <= 0.3 and 0.3 <= beta <= 0.7)
    print(f"E1 {E1}  E2 {E2}  d*_inf {dinf:.4f}  L0 {L0}  d_low {dlow:.5f}  S1 {S1}  S2 {S2} (beta {beta:.3f})", flush=True)
    devW = {k: (2 * len(SETS[k]) * g_(f"W_{k}") / 2 - dlow) / dlow for k in W_SETS}
    W0 = bool(abs(devW["1,1"]) <= 0.01)
    W1 = bool(any(abs(devW[k]) > 0.10 for k in ("1,2", "1,3", "2,3")))
    ax4 = g_("E_w20_dt4")
    H1 = bool(rel(g_("H"), 2 / 3 * ax4) <= 0.05)
    devA = {k: (2 * len(SETS[k]) * g_(f"A_{k}") / 2 - ax4) / ax4 for k in A_SETS}
    A1p = bool(all(abs(v) <= 0.05 for v in devA.values()))
    print("weak-edge deviations " + " ".join(f"{k}: {100*v:+.1f}%" for k, v in devW.items()), flush=True)
    print(f"honeycomb d* {g_('H'):.4f} vs (2/3) axis {2/3*ax4:.4f}  dev {100*(g_('H')/(2/3*ax4)-1):+.2f}%", flush=True)
    print("A1' deviations " + " ".join(f"{k}: {100*v:+.2f}%" for k, v in devA.items()), flush=True)
    print(f"W0 {W0}  W1 {W1}  H1 {H1}  A1' {A1p}", flush=True)
    out = save_result(Path(__file__).resolve().parent / "processed_data" / "paperC_edges.npz",
                      {"model": "run_nbr = paperC_continuum.run_chain with neighbour table (asserted bitwise)",
                       "E1": E1, "E2": E2, "L0": L0, "S1": S1, "S2": S2, "beta_low": beta, "W0": W0, "W1": W1,
                       "H1": H1, "A1p": A1p, "d_inf": dinf, "d_low": dlow, "thr": thr, "devW": devW, "devA": devA,
                       "brackets": B, "no_bracket": missing, "eps": EPS,
                       "grids": {n: [sp, g, up] for n, (sp, g, up) in grids.items()}},
                      grid_names=np.array(list(grids)), grid_ext=np.array([[r[0] for r in res[n]] + [np.nan] *
                      (20 - len(res[n])) for n in grids]), grid_prop=np.array([[float(r[1]) for r in res[n]] +
                      [np.nan] * (20 - len(res[n])) for n in grids]), eps=np.array(EPS), v_strong=vh, v_weak=vl,
                      d_strong=dinf * (1 - np.array(EPS)), d_weak=dd)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
