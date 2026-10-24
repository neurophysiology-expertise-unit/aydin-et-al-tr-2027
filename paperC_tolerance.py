"""
paperC_tolerance.py — paper C, round 6: does the operating window survive cell-to-cell parameter
mismatch and noise, and how do its edges move? Written AFTER paperC_design.npz and BEFORE any run of this script.

INTEGRATOR. run_gen2 = paperC_model.run_gen (2-D, per-cell arrays, verbatim) with two documented additions:
 recompute : the settling release rule (coupling recomputed after g is set; the recommended rule, paperC_design);
 sig       : additive noise on C, sig * sqrt(dt) * N(0,1) per cell per step (numba RNG seeded by nseed); sig = 0 -> none.
GATE (asserted first): run_gen2(recompute False, sig 0) reproduces paperC_model.run_gen t_rise bit for bit.

PROTOCOL. 2-D strip, 32 (transverse, periodic) x 256 (propagation, periodic) cells, axis planar front seeded as a band
of 5 rows (C = 1.5, g = 1). Homogeneous means from hom_params; MISMATCH: each cell's I0, tau_h and Deff multiplied
by independent factors (1 + delta xi), xi ~ U(-1, 1), realisations 1-5 (numpy default_rng(seed)). gamma = 1,
settling rule. Strong edge: dt = DT/8, T = 120 s. Weak edge: dt = DT, T = 1200 s.
PROPAGATES := every row within 40 rows of the seed (both sides) has >= 90 % of its cells fired by T; with noise,
additionally the unseeded control run (same noise seed) has fired < 10 % of those cells.
Edges per realisation by grid then 8 bisection halvings (strong: d grid {4 6 8 9 10 11 12 13 14 16 20};
weak: {0.2 0.3 0.4 0.5 0.6 0.7 0.8 1 1.5 2}).
Conditions: delta in {0, 0.1, 0.2, 0.4} (delta 0: one run); noise at delta = 0 with sig = f * sig_A, f in
{0.25, 0.5, 1}, sig_A = the model's reference level SIGMA_EM_PREDICTED (1 + 4 alpha) NOISE_MULT = 0.0727; noise realisations 1-5.
(At f = 1 the homogeneous lattice may fire spontaneously, which the control criterion counts as no propagation; f = 1
and f = 0.25 are therefore descriptive, and M3 is tested at f = 0.5.)

PRE-REGISTERED PREDICTIONS
 M1 (window survives mismatch): for every delta and every realisation both edges exist and d_low < d*.
 M2 (strong edge falls, bounded by the most strongly coupled cells): for delta in {0.1, 0.2, 0.4} the median d* over
    realisations lies in [d*(0)/(1 + delta), d*(0)], and d* decreases monotonically with delta (medians).
 M3 (window survives noise): at f = 0.5 (sig = 0.0364) both edges exist in every noise realisation, and the median strong edge
    lies within 10 % of the noise-free d*(0).
Failed predictions are reported as failed; criteria are not revised after the data land.
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, SIGMA_EM_PREDICTED, NOISE_MULT
from core.provenance import save_result
import paperC_model
from paperC_model import run_gen, hom_params, THETA, TAU_REF, C_DOWN, K_REF, SCALE, G_FLOOR, ALPHA
from paperC_continuum import NBIS

CKPT = Path("/tmp/paperC_tol_ckpt")
NX, NY, W, PR = 32, 256, 5, 40
DELTAS = [0.0, 0.1, 0.2, 0.4]
REAL = [1, 2, 3, 4, 5]
SIG_A = float(SIGMA_EM_PREDICTED * (1 + 4 * ALPHA) * NOISE_MULT)
FS = [0.25, 0.5, 1.0]
G_HI = [4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 20]
G_LO = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2]


@njit(fastmath=True, cache=False)
def run_gen2(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, dt, recompute, sig, nseed):
    # paperC_model.run_gen loop (t_rise only) + settling rule + additive noise
    np.random.seed(nseed)
    nx, ny = I0.shape
    C = np.full((nx, ny), C_DOWN); h = (C + A_FHN) / B_FHN
    g = np.zeros((nx, ny)); reft = np.zeros((nx, ny)); prev = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if seedmask[i, j]:
                C[i, j] = 1.5; g[i, j] = 1.0
    prev = (0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5).astype(np.float64)
    t_rise = np.full((nx, ny), -1.0)
    sq = np.sqrt(dt)
    for t in range(steps):
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        active = (C_active > 0.5).astype(np.float64)
        inref = (reft > 0.0).astype(np.float64)
        broadcast = C_active * g * (1.0 - inref)
        coupling_in = Deff * laplacian(broadcast)
        rose = active * (1.0 - prev)
        fell = prev * (1.0 - active)
        g_new = np.minimum(1.0, G_FLOOR + gamma_regen * (coupling_in / SCALE))
        g = np.where(rose > 0.5, np.maximum(g_new, 0.0), g)
        reft = np.where(fell > 0.5, tau_ref, reft)
        if recompute:
            broadcast = C_active * g * (1.0 - inref)
            coupling_in = Deff * laplacian(broadcast)
        clamp = K_REF * inref * (C - C_DOWN)
        if sig > 0.0:
            noise = sig * np.random.standard_normal((nx, ny))
        else:
            noise = np.zeros((nx, ny))
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + dt * dC + sq * noise
        h = h + dt * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - dt, 0.0)
        prev = active
        for i in range(nx):
            for j in range(ny):
                if rose[i, j] > 0.5 and t_rise[i, j] < 0.0:
                    t_rise[i, j] = t * dt
    return t_rise


def params(d, delta, real):
    I0, th, De, gd = hom_params((NX, NY), d)
    if delta > 0:
        rng = np.random.default_rng(real)
        I0 = I0 * (1 + delta * rng.uniform(-1, 1, (NX, NY)))
        th = th * (1 + delta * rng.uniform(-1, 1, (NX, NY)))
        De = De * (1 + delta * rng.uniform(-1, 1, (NX, NY)))
    return I0, th, De, gd


def seedmask():
    sm = np.zeros((NX, NY), np.bool_); c = NY // 2; sm[:, c - W // 2: c - W // 2 + W] = True
    return sm


def one(d, delta, real, sig, edge):
    dt, T = (DT / 8, 120.0) if edge == "hi" else (DT, 1200.0)
    I0, th, De, gd = params(d, delta, real)
    sm = seedmask(); c = NY // 2; lo, hi = c - W // 2, c - W // 2 + W - 1
    rows = list(range(lo - PR, lo)) + list(range(hi + 1, hi + 1 + PR))
    tr = run_gen2(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T / dt)), dt, True, sig, real)
    frac = (tr[:, rows] >= 0).mean(0)
    ok = bool((frac >= 0.9).all())
    if ok and sig > 0:
        trc = run_gen2(I0, th, De, gd, THETA, np.zeros((NX, NY), np.bool_), 1.0, TAU_REF, int(round(T / dt)), dt, True,
                       sig, real)
        ok = bool((trc[:, rows] >= 0).mean() < 0.1)
    return ok


def job(a):
    return one(*a)


def bis(a):
    delta, real, sig, edge, lo, hi = a
    upper = edge == "hi"
    for _ in range(NBIS):
        mid = 0.5 * (lo + hi)
        if one(mid, delta, real, sig, edge) == upper:
            lo = mid
        else:
            hi = mid
    return [lo, hi]


def gate():
    I0, th, De, gd = hom_params((16, 16), 8.0)
    sm = np.zeros((16, 16), np.bool_); sm[6:10, 6:10] = True
    a = run_gen(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, 3000, 5)[1]
    b = run_gen2(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, 3000, DT, False, 0.0, 1)
    assert np.array_equal(a, b), "gate"
    print("GATE: run_gen2(recompute False, sig 0) == run_gen bit for bit", flush=True)


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def bracket(grid, prop, upper):
    ip = [i for i, p in enumerate(prop) if p]
    if not ip:
        return None
    if upper:
        i = ip[-1]; return (grid[i], grid[i + 1]) if i + 1 < len(grid) else None
    i = ip[0]; return (grid[i - 1], grid[i]) if i > 0 else None


def main():
    CKPT.mkdir(exist_ok=True)
    paperC_model.check()
    gate()
    conds = [(0.0, 1, 0.0)] + [(dl, r, 0.0) for dl in DELTAS[1:] for r in REAL] + [(0.0, r, f * SIG_A) for f in FS for r in REAL]
    jobs = [(dl, r, s, e, x) for (dl, r, s) in conds for e, g in (("hi", G_HI), ("lo", G_LO)) for x in g]
    pool = Pool(11)
    R = ck("grids", lambda: pool.map(job, [(x, dl, r, s, e) for (dl, r, s, e, x) in jobs], chunksize=1))
    res = {}
    for (dl, r, s, e, x), p in zip(jobs, R):
        res.setdefault((dl, r, s, e), []).append(p)
    bj = {}
    for k, v in res.items():
        g = G_HI if k[3] == "hi" else G_LO
        print(f"{k}: " + "".join("+" if p else "." for p in v), flush=True)
        b = bracket(g, v, k[3] == "hi")
        if b is not None:
            bj[k] = (*k, *b)
    keys = list(bj)
    B = ck("bisect", lambda: pool.map(bis, [bj[k] for k in keys], chunksize=1))
    pool.close()
    thr = {k: 0.5 * sum(b) for k, b in zip(keys, B)}
    edge = lambda dl, r, s, e: thr.get((dl, r, s, e), np.nan)
    d0 = edge(0.0, 1, 0.0, "hi"); l0 = edge(0.0, 1, 0.0, "lo")
    M1 = all(np.isfinite(edge(dl, r, 0.0, "hi")) and np.isfinite(edge(dl, r, 0.0, "lo")) and
             edge(dl, r, 0.0, "lo") < edge(dl, r, 0.0, "hi") for dl in DELTAS[1:] for r in REAL)
    med = {dl: float(np.median([edge(dl, r, 0.0, "hi") for r in REAL])) for dl in DELTAS[1:]}
    M2 = bool(all(d0 / (1 + dl) <= med[dl] <= d0 for dl in DELTAS[1:]) and med[0.1] >= med[0.2] >= med[0.4])
    noise = {f: ([edge(0.0, r, f * SIG_A, "hi") for r in REAL], [edge(0.0, r, f * SIG_A, "lo") for r in REAL]) for f in FS}
    nh, nl = noise[0.5]
    M3 = bool(all(np.isfinite(nh)) and all(np.isfinite(nl)) and abs(np.median(nh) - d0) / d0 <= 0.10)
    print(f"d*(0) {d0:.4f}  d_low(0) {l0:.4f}  medians {med}  noise hi {nh} lo {nl}", flush=True)
    print(f"M1 {M1}  M2 {M2}  M3 {M3}", flush=True)
    out = save_result(Path(__file__).resolve().parent / "processed_data" / "paperC_tolerance.npz",
                      {"model": "run_gen2 (= run_gen + settling rule + noise), 2-D strip 32 x 256", "M1": bool(M1),
                       "M2": M2, "M3": M3, "d0": d0, "l0": l0, "medians": med, "noise_hi": nh, "noise_lo": nl,
                       "sig_A": SIG_A, "noise_all": {str(f): v for f, v in noise.items()}, "thr": {str(k): v for k, v in thr.items()},
                       "grids": {str(k): v for k, v in res.items()}},
                      d0=np.array(d0))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
