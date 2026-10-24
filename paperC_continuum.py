"""
paperC_continuum.py — paper C, round (d)+(a).
Homogeneous lattice, gamma_regen = 1, noise off, tau_ref 15 s, T = 120 s, hom_params (every per-cell parameter at
its ensemble mean). Planar fronts are studied through their EXACT 1-D reduction (paperC_direction D1 PASS): a planar
front normal to a rational lattice direction depends only on the row index n, and the lattice Laplacian of the
broadcast b becomes a 1-D stencil  sum_o b[n+o] + m*b[n] - z*b[n]  (o: the projected offsets of the z - m
neighbours that are NOT in the cell's own row; m: neighbours in the same row, which fire with the cell).
Square axis (1,0): o = {+1,-1}, m = 2 (the 1-D chain at D). Square (p,q), p > q >= 1: o = {+-p, +-q}, m = 0.

INTEGRATOR. run_chain below is paperC_model.run_gen with three documented changes and nothing else: (i) 1-D arrays and the general stencil
above in place of core.model.laplacian (the term order of laplacian is kept: offsets in the listed order, then the
m same-row terms, then -z*b); (ii) the time step DT is an argument dt; (iii) the frame record (fld, stride) and the
g/cin/peak bookkeeping are dropped, only t_rise is returned. main() first ASSERTS that run_chain(o={+1,-1}, m=2,
dt=DT) reproduces paperC_model.run_gen on an (N,1) chain bit for bit (t_rise identical) at d = 8, 14, 16, and that
paperC_model.check() passes. If that assertion fails nothing is run.

MEASURE. Seed: w consecutive rows at the centre of a periodic chain of N rows (C = 1.5, g = 1). extent = largest
row distance from the seed edge of any row that fired (t_rise >= 0) by T. PROPAGATES := extent >= P rows.
Thresholds by BISECTION on the coupling multiplier: 8 halvings of a grid bracket (lo propagates, hi fails).

======================================================================================================
PART (d) — CONTINUUM LIMIT (decisive). Refinement: cell spacing h = 1/k, k in {1, 2, 4, 8}; physical coupling d
(the usual d_mult) fixed while the LATTICE coupling multiplier is d_lat = d / h^2 = d k^2; seed w = 5k rows
(5 physical cells); N = 256k rows (256 physical cells); P = 20k rows (20 physical cells, as paperC_direction);
dt = DT * min(1/k^2, 16/max(d_lat,16)) (diffusive scaling; the cap keeps D_lat*dt at or below its k = 1, d = 16
value). Grid in lattice units d_lat in G = {0.25 0.5 1 2 4 8 12 13 14 15 16 17 18 20 24 32 64 128 256 512 1024
2048} for every k, i.e. physical d = d_lat / k^2 (this includes physical d = 1, 4, 8 at every k where d_lat <= 2048).
Note, analytic: the chain equations contain the coupling only as d_lat, and the cell dynamics (FHN, gate, refractory
clamp, gain event) are local, so at fixed seed width and dt the outcome is a function of d_lat alone. Refinement at
fixed physical d is therefore the limit d_lat -> infinity, with the seed widened and dt shrunk alongside. The
continuum equation C_t = F(C,h) + D d_xx[ sigma(C) g (1-ref) ] (g set from D d_xx b at the rising edge) has no length
scale besides sqrt(D): x -> x sqrt(D) removes D, so in the continuum a planar front propagates for all D or for none.
 C1 (time step). At k = 1 the bisected upper threshold d*_lat at dt = DT/2 and DT/4 lies within 2 % of that at DT;
    the same holds at k = 2 and k = 4 between dt_k and dt_k/2. PASS iff all four comparisons are within 2 %.
 C2 (threshold is fixed in lattice units). The bisected d*_lat(k) for k = 2, 4, 8 lies within 10 % of d*_lat(1).
    Equivalently the physical threshold d*(k) = d*_lat(1)/k^2 -> 0 under refinement.
 C3 (no re-entry, decisive). For every k, NO grid point with d_lat > 1.1 * d*_lat(1), up to d_lat = 2048, propagates.
Pre-registered reading:
  C3 PASS (with C2) -> refinement does not remove the strong-coupling failure; the limit d_lat -> inf of the planar
     front fails, so (by the scale invariance above) the continuum medium with this output coupling at gamma = 1
     carries no planar front for any D, and the lattice front at d < d* is carried by discreteness. The failure
     SURVIVES the continuum limit.
  C3 FAIL (a propagating point above the threshold at some k) -> the failure is a finite-d_lat window, a lattice
     artefact that refinement removes.
  C2 FAIL with C3 PASS -> the threshold depends on seed width or dt; report the dependence, no clean lattice number.
 C4 (descriptive, no prediction): the lower (weak-coupling) end of the propagating window, from the same grid.

PART (a) — WHY EVERY NON-AXIS DIRECTION FAILS AT AXIS/2: COUNTING. Analytic first: in a planar front the m same-row
partners carry exactly b[n] and cancel exactly (m*b[n] - m*b[n] = 0), so "removing the lateral partners" of the axis
front gives the identical chain; that modified lattice is not a test. The counting hypothesis is instead:
 H_count: the strong-coupling threshold depends only on s = z - m, the number of neighbours that do NOT fire with
 the cell (the self-drain is s*D*b[n]), not on how far up/downstream they sit:  d*(o) = (2/s) * d*_ax,
 with d*_ax = the k = 1, dt = DT bisected chain threshold of part (d).
Offset sets (each listed offset a means neighbours at n+a and n-a; m = 0; seed w = 5*max a rows, N = 256*max a,
P = 20*max a rows, dt = DT), with the lattice direction they reduce from:
  s = 4: {1,1} square (1,1) [control: exactly the chain at 2D, so d* = d*_ax/2 by scaling];
         {1,2} square (2,1) / cubic (2,1,0);  {1,3} square (3,1);  {2,3} square (3,2);  {1,4} square (4,1);
         {1,5} square (5,1)
  s = 6: {1,1,1} cubic (1,1,1) [control: chain at 3D, d*_ax/3 exactly];  {1,1,2} triangular lattice, front normal
         to a bond;  {1,2,3} cubic (3,2,1)
  s = 8: {1,2,3,4} (hypothetical 8-neighbour stencil)
Grid on d: {2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 9 10 12 16}, then bisection.
 A0 (method): the two controls hit (2/s) d*_ax within 1 %.
 A1 (H_count): every NON-control set satisfies |s d*(o)/2 - d*_ax| / d*_ax <= 5 %. PASS iff all eight do.
 A2 (consistency with the 2-D lattice): the 1-D {1,2}, {1,3}, {2,3} thresholds lie in (7, 8], the bracket the 2-D
    stripes (2,1), (3,1), (3,2) gave in paperC_direction.npz.
If A1 fails, the per-set deviations are reported; the hypothesis is not repaired after the data land.

Predictions written 2026-10-13 before any run of this script (only timing runs of <= 2000 steps were made).
Failed predictions are reported as failed; criteria are not revised after the data land.
Output: processed_data/paperC_continuum.npz (provenance-stamped). Checkpoints /tmp/paperC_cont_ckpt.
Run: python -s paperC_continuum.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from core.model import DT, ETA, A_FHN, B_FHN
from core.provenance import save_result
import paperC_model
from paperC_model import (run_gen, hom_params, THETA, TAU_REF, T_WIN, C_DOWN, K_REF, SCALE, G_FLOOR)

CKPT = Path("/tmp/paperC_cont_ckpt")
KS = [1, 2, 4, 8]
G_LAT = [0.25, 0.5, 1, 2, 4, 8, 12, 13, 14, 15, 16, 17, 18, 20, 24, 32, 64, 128, 256, 512, 1024, 2048]
SETS = {"1,1": (1, 1), "1,2": (1, 2), "1,3": (1, 3), "2,3": (2, 3), "1,4": (1, 4), "1,5": (1, 5),
        "1,1,1": (1, 1, 1), "1,1,2": (1, 1, 2), "1,2,3": (1, 2, 3), "1,2,3,4": (1, 2, 3, 4)}
CONTROLS = ["1,1", "1,1,1"]
G_A = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12, 16]
NBIS = 8


@njit(fastmath=True, cache=False)
def run_chain(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, offs, m_same, dt):
    # paperC_model.run_gen, verbatim except: 1-D stencil for laplacian, dt argument, only t_rise recorded
    nx = I0.shape[0]; no = offs.shape[0]; z = float(no + m_same)
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
            acc = broadcast[(i + offs[0]) % nx]
            for k in range(1, no):
                acc = acc + broadcast[(i + offs[k]) % nx]
            for k in range(m_same):
                acc = acc + broadcast[i]
            lap[i] = acc - z * broadcast[i]
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


def chain_setup(N, w):
    sm = np.zeros(N, np.bool_); c = N // 2; lo = c - w // 2; hi = lo + w - 1; sm[lo:hi + 1] = True
    idx = np.arange(N)
    dist = np.where(idx < lo, lo - idx, np.where(idx > hi, idx - hi, 0))
    return sm, dist


def simulate(d_mult, offs, m_same, N, w, dt, P):
    """one planar-front run in its 1-D reduction; returns (extent rows, propagates)."""
    sm, dist = chain_setup(N, w)
    I0, th, De, gd = (a.ravel() for a in hom_params((N, 1), d_mult))
    tr = run_chain(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T_WIN / dt)), np.array(offs, np.int64),
                   m_same, dt)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    return ext, bool(ext >= P)


def dt_of(k, dlat, frac=1.0):
    return DT * min(1.0 / k ** 2, 16.0 / max(dlat, 16.0)) * frac


def run_d(args):
    k, dlat, frac = args          # the chain sees only the lattice multiplier d_lat = d k^2
    return simulate(dlat, (1, -1), 2, 256 * k, 5 * k, dt_of(k, dlat, frac), 20 * k)


def run_a(args):
    key, d = args
    a = SETS[key]; mx = max(a)
    offs = [s * x for x in a for s in (1, -1)]
    return simulate(d, tuple(offs), 0, 256 * mx, 5 * mx, DT, 20 * mx)


def bisect(job, f, lo, hi):
    """lo propagates, hi fails; NBIS halvings. Returns [lo, hi]."""
    for _ in range(NBIS):
        mid = 0.5 * (lo + hi)
        if f(job(mid))[1]:
            lo = mid
        else:
            hi = mid
    return [lo, hi]


def bis_d(args):
    k, frac, lo, hi = args
    return bisect(lambda x: (k, x, frac), run_d, lo, hi)


def bis_a(args):
    key, lo, hi = args
    return bisect(lambda x: (key, x), run_a, lo, hi)


def upper_bracket(grid, prop):
    """(largest propagating grid point, next grid point); hi = inf if the top point propagates."""
    ip = [i for i, p in enumerate(prop) if p]
    if not ip:
        return np.nan, grid[0]
    i = ip[-1]
    return grid[i], (grid[i + 1] if i + 1 < len(grid) else np.inf)


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def assert_equivalence():
    N, w = 256, 5
    sm, _ = chain_setup(N, w)
    for d in (8.0, 14.0, 16.0):
        I0, th, De, gd = hom_params((N, 1), d)
        ref = run_gen(I0, th, De, gd, THETA, sm.reshape(N, 1), 1.0, TAU_REF, int(T_WIN / DT), 5)[1].ravel()
        got = run_chain(I0.ravel(), th.ravel(), De.ravel(), gd.ravel(), THETA, sm, 1.0, TAU_REF, int(T_WIN / DT),
                        np.array([1, -1], np.int64), 2, DT)
        assert np.array_equal(ref, got), ("run_chain != run_gen", d, np.nanmax(np.abs(ref - got)))
    print("run_chain reproduces paperC_model.run_gen chain t_rise bit for bit (d 8, 14, 16)", flush=True)


def main():
    CKPT.mkdir(exist_ok=True)
    paperC_model.check()
    assert_equivalence()
    pool = Pool(11)
    jd = [(k, x, 1.0) for k in KS for x in G_LAT]
    Rd = ck("d_grid", lambda: pool.map(run_d, jd, chunksize=1))
    ext_d = np.array([r[0] for r in Rd]).reshape(len(KS), len(G_LAT)) / np.array(KS)[:, None]   # physical cells
    prop_d = np.array([r[1] for r in Rd]).reshape(len(KS), len(G_LAT))
    for a, k in enumerate(KS):
        print(f"k {k}: " + " ".join(f"{x:g}:{ext_d[a,b]:.0f}{'+' if prop_d[a,b] else '.'}" for b, x in enumerate(G_LAT)),
              flush=True)
    br = [upper_bracket(G_LAT, prop_d[a]) for a in range(len(KS))]
    ja = [(key, d) for key in SETS for d in G_A]
    Ra = ck("a_grid", lambda: pool.map(run_a, ja, chunksize=1))
    ext_a = np.array([r[0] for r in Ra]).reshape(len(SETS), len(G_A))
    prop_a = np.array([r[1] for r in Ra]).reshape(len(SETS), len(G_A))
    bra = [upper_bracket(G_A, prop_a[a]) for a in range(len(SETS))]
    jb = [(k, 1.0) for k in KS] + [(1, 0.5), (1, 0.25), (2, 0.5), (4, 0.5)]
    jb = [(k, fr, *br[KS.index(k)]) for k, fr in jb if np.all(np.isfinite(br[KS.index(k)]))]
    jab = [(key, *bra[i]) for i, key in enumerate(SETS) if np.all(np.isfinite(bra[i]))]

    def both():
        r1 = pool.map_async(bis_d, jb, chunksize=1); r2 = pool.map_async(bis_a, jab, chunksize=1)
        return [r1.get(), r2.get()]
    Bd, Ba = ck("bisect", both)
    pool.close()
    thr = {(k, fr): 0.5 * (lo + hi) for (k, fr, _, _), (lo, hi) in zip(jb, Bd)}
    for (k, fr, _, _), (lo, hi) in zip(jb, Bd):
        print(f"  d*_lat k {k} dt x{fr:g}: ({lo:.4f}, {hi:.4f}]  physical d* = {0.5*(lo+hi)/k**2:.5f}", flush=True)
    t1 = thr.get((1, 1.0), np.nan)
    rel = lambda a, b: abs(a - b) / b
    need = [(1, .5), (1, .25), (2, .5), (4, .5)]
    C1 = bool(all(key in thr for key in need) and
              all(rel(thr[(1, f)], t1) <= 0.02 for f in (0.5, 0.25)) and
              all(rel(thr[(k, 0.5)], thr[(k, 1.0)]) <= 0.02 for k in (2, 4)))
    C2 = bool(all((k, 1.0) in thr and rel(thr[(k, 1.0)], t1) <= 0.10 for k in (2, 4, 8)))
    above = np.array(G_LAT) > 1.1 * t1
    C3 = bool(np.isfinite(t1) and not prop_d[:, above].any())
    lower = [next((G_LAT[i] for i, p in enumerate(prop_d[a]) if p), np.nan) for a in range(len(KS))]
    print(f"C1 {C1}  C2 {C2}  C3 {C3}   lowest propagating d_lat per k (C4): {lower}", flush=True)
    tha = {key: 0.5 * (lo + hi) for (key, _, _), (lo, hi) in zip(jab, Ba)}
    dev = {}
    for i, key in enumerate(SETS):
        s = 2 * len(SETS[key]); v = tha.get(key, np.nan)
        dev[key] = (s * v / 2 - t1) / t1
        print(f"  set {{{key}}} s {s}: grid {''.join('+' if p else '.' for p in prop_a[i])}"
              f"  d* {v:.4f}  s d*/2 = {s*v/2:.4f}  dev {100*dev[key]:+.2f} %", flush=True)
    A0 = bool(all(abs(dev[c]) <= 0.01 for c in CONTROLS))
    A1 = bool(all(abs(dev[k]) <= 0.05 for k in SETS if k not in CONTROLS))
    A2 = bool(all(7.0 < tha.get(k, np.nan) <= 8.0 for k in ("1,2", "1,3", "2,3")))
    print(f"A0 {A0}  A1 {A1}  A2 {A2}", flush=True)
    out = save_result(Path(__file__).resolve().parent / "processed_data" / "paperC_continuum.npz",
                      {"model": "run_chain = paperC_model.run_gen verbatim with 1-D stencil + dt arg (asserted bitwise)",
                       "gamma_regen": 1.0, "ks": KS, "g_lat": G_LAT, "sets": {k: list(v) for k, v in SETS.items()},
                       "g_a": G_A, "nbis": NBIS, "C1": C1, "C2": C2, "C3": C3, "A0": A0, "A1": A1, "A2": A2,
                       "thr_d": {str(k): v for k, v in thr.items()}, "thr_a": tha, "dev_a": dev, "lower_dlat": lower},
                      ks=np.array(KS), g_lat=np.array(G_LAT), ext_d=ext_d, prop_d=prop_d,
                      bis_d_jobs=np.array([[k, fr, lo, hi] for k, fr, lo, hi in jb]), bis_d=np.array(Bd),
                      set_keys=np.array(list(SETS)), g_a=np.array(G_A), ext_a=ext_a, prop_a=prop_a,
                      bis_a_keys=np.array([j[0] for j in jab]), bis_a=np.array(Ba))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
