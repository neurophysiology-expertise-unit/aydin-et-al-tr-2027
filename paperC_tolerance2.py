"""
paperC_tolerance2.py — paper C, round 7: make the fragility part of the result.
Written AFTER paperC_tolerance.npz (round 6: M1-M3 FAILED, they stay failed) and BEFORE any run of this script.

WHY. Round 6 found no front at any d in its strong-edge grid 4-20 under 10-40 % mismatch or quarter/half noise,
while its weak-edge grid (dt = DT, 1200 s) propagated up to d = 2. Two questions were left open: where the strong
edge is below 4, and whether "no front" meant the front stops or that it advances but leaves > 10 % of a row unfired
(the round-6 criterion needs >= 90 % of every row).

INTEGRATOR. run_gen3 = paperC_tolerance.run_gen2 (settling rule + additive noise; itself run_gen + two documented
additions) VERBATIM, plus one addition: every 1000 steps the loop stops early once at least `nstop` cells of
`stopmask` have fired (t_rise >= 0). Stopping only truncates the run; nothing already recorded changes, and the
criteria below only use cells of stopmask (seeded run: stop when all scored cells fired; control: stop once 10 % of
them fired, which already decides the control). GATE (asserted first): run_gen3 with nstop larger than the mask
== run_gen2 bit for bit (also tests/test_simulator.py).

PROTOCOL (as round 6 unless stated). 2-D strip 32 (transverse, periodic) x 256 (propagation, periodic), axis planar
front seeded as a band of 5 rows (C = 1.5, g = 1); hom_params means; settling rule; gamma 1; dt = DT/8 (round 6's
strong-edge step). Same realisations: mismatch delta in {0.1, 0.2, 0.4}, realisations 1-5 (paperC_tolerance.params,
I0, tau_h, Deff x (1 + delta xi), xi ~ U(-1, 1), default_rng(realisation)); noise at delta = 0 with sig = f sig_A,
f in {0.25, 0.5}, noise seed = realisation 1-5 (numba RNG, as round 6). Plus the noise-free homogeneous reference
(delta = 0). Run length T = 240 s (round 6: 120 s; the longer run lets slow fronts at d ~ 1 cover 40 rows; the first
120 s of every run are the round-6 trajectory, same dt and seeds).
Scored rows: the 40 rows on each side of the seed band (80 rows x 32 cells), as round 6.

CRITERIA, each run scored twice:
 (i)  ROUND-6 CRITERION, exactly: at T = 120 s (cells with t_rise < 120 s), every scored row has >= 90 % of its cells
      fired; with noise, the unseeded control (same parameters and noise seed) has fired < 10 % of the scored cells
      by 120 s. (No control for mismatch, as round 6.)
 (ii) FRONT ARRIVAL, at T = 240 s:
      (a) every scored row, both sides, has >= 50 % of its 32 cells fired: a perforated front counts as arrived as long
          as at most half of each row is skipped; skipped cells need not fire later;
      (b) the activity travelled outward from the seed: on each side the median t_rise over the fired cells of row 40
          is larger than that of row 20, which is larger than that of row 1 (rows counted from the seed edge);
      (c) for EVERY perturbed condition (mismatch or noise), the unseeded control fires < 10 % of the scored cells by
          240 s.
      PROPAGATES(ii) := (a) and (b) and (c).
EDGE. Strong-edge grid G7 = {1 1.5 2 2.5 3 4 6 8 11 14 20} (reaches below round 6's floor of 4 and overlaps the
weak-edge range; 4 6 8 11 14 20 are round-6 grid points). Per criterion and condition, bracket = (highest grid point
that propagates, next grid point); then 5 bisection halvings, d* = midpoint. Highest point propagates: d* > 20 (no
bisection). No grid point propagates: no strong edge >= 1 (the window is closed at d >= 1 for that realisation).
d_low per realisation is round 6's (paperC_tolerance.npz thr; dt = DT, 1200 s; L0: the weak edge is dt-converged).

CONSISTENCY GATE R (asserted on the data, reported, not a prediction): criterion (i) reproduces the round-6 outcome at
every shared grid point (4 6 8 11 14 20) for all 26 conditions (the first 120 s are the same trajectory).

PRE-REGISTERED PREDICTIONS ("level" = one of delta 0.1, 0.2, 0.4, f 0.25, 0.5; five realisations each)
 W  (window survives): under criterion (i) AND under criterion (ii), at every level and in every realisation, the strong
    edge is bracketed in G7 and d* > d_low (round 6).
 N  (narrowed): every bracketed d* at every level lies in (d_low, 4], both criteria (round 6 failed at 4 under (i)).
 CE (not a criterion effect; the front stops): at every level median d*(ii) <= 1.25 x median d*(i).
 S1 (design rule for mismatch): median d*(delta) x (1 + delta) is the same at delta 0.1, 0.2, 0.4 within +-15 % of its
    mean d_r, and the medians fall monotonically with delta. Reasoning: once mismatch desynchronises the cells of a
    row, the axis front behaves like a rough (non-planar) front with a delta-independent edge d_r, and the most
    strongly coupled cells (Deff up to (1 + delta) times the mean) fail first. Tested under each criterion (S1i, S1ii).
 S2 (noise): the median d* at f 0.25 and at f 0.5 each lie within +-15 % of d_r (from S1, same criterion): noise
    desynchronises the row like mismatch, with no Deff spread. S2i, S2ii.
 S3 (where the plateau is): 2 < d_r <= 4 (homogeneous disks, i.e. curved fronts, stopped from d = 4 in round 1). S3i, S3ii.
 A median that is undefined (no bracketed edge in some realisation) makes S1-S3 for that criterion FAIL.
READING, fixed in advance, per level and criterion:
 - SURVIVES (narrowed): all five realisations have a bracketed d* > d_low.
 - CLOSES: no grid point propagates in >= 3 of 5 realisations.
 - otherwise MIXED (reported as such, realisation by realisation).
 - CRITERION EFFECT: median d*(ii) >= 1.5 x median d*(i) (the round-6 collapse at that level was mainly the 90 %
   criterion); between 1.25 and 1.5: partial.
Failed predictions are reported as failed (Supplement S2); criteria are not revised after the data land.
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from core.model import laplacian, DT, ETA, A_FHN, B_FHN
from core.provenance import save_result
import paperC_model
from paperC_model import THETA, TAU_REF, C_DOWN, K_REF, SCALE, G_FLOOR
from paperC_tolerance import run_gen2, params, seedmask, SIG_A, NX, NY, W, PR

HERE = Path(__file__).resolve().parent
CKPT = Path("/tmp/paperC_tol2_ckpt")
DT7 = DT / 8
T1, T2 = 120.0, 240.0
S1_, S2_ = int(round(T1 / DT7)), int(round(T2 / DT7))
G7 = [1, 1.5, 2, 2.5, 3, 4, 6, 8, 11, 14, 20]
NB = 5
LEVELS = [("delta", 0.1), ("delta", 0.2), ("delta", 0.4), ("f", 0.25), ("f", 0.5)]
REAL = [1, 2, 3, 4, 5]
CONDS = [(0.0, 1, 0.0)] + [(v, r, 0.0) for k, v in LEVELS if k == "delta" for r in REAL] + \
        [(0.0, r, v * SIG_A) for k, v in LEVELS if k == "f" for r in REAL]


@njit(fastmath=True, cache=False)
def run_gen3(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, dt, recompute, sig, nseed,
             stopmask, nstop):
    # paperC_tolerance.run_gen2 verbatim + early stop (every 1000 steps: >= nstop cells of stopmask fired)
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
        if t % 1000 == 999:
            n = 0
            for i in range(nx):
                for j in range(ny):
                    if stopmask[i, j] and t_rise[i, j] >= 0.0:
                        n += 1
            if n >= nstop:
                break
    return t_rise


def rows():
    c = NY // 2; lo, hi = c - W // 2, c - W // 2 + W - 1
    left = [lo - k for k in range(1, PR + 1)]; right = [hi + k for k in range(1, PR + 1)]   # row k at index k-1
    return left, right


LEFT, RIGHT = rows()
SCORED = sorted(LEFT + RIGHT)
SMASK = np.zeros((NX, NY), np.bool_); SMASK[:, SCORED] = True
NSC = int(SMASK.sum())


def sim(d, cond, seeded):
    delta, real, sig = cond
    I0, th, De, gd = params(d, delta, real)
    sm = seedmask() if seeded else np.zeros((NX, NY), np.bool_)
    nstop = NSC if seeded else int(np.ceil(0.1 * NSC))
    return run_gen3(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, S2_, DT7, True, sig, real, SMASK, nstop)


def by(tr, tmax):
    return (tr >= 0) & (tr < tmax - 0.5 * DT7)


def score(d, cond):
    """-> dict: i, ii, parts. One seeded run; the control only when a criterion needs it."""
    delta, real, sig = cond
    tr = sim(d, cond, True)
    f120 = by(tr, T1)
    ci = bool((f120[:, SCORED].mean(0) >= 0.9).all())
    f240 = by(tr, T2)
    a = bool((f240[:, SCORED].mean(0) >= 0.5).all())
    b = False
    if a:
        med = lambda r: float(np.median(tr[f240[:, r], r]))
        b = all(med(side[39]) > med(side[19]) > med(side[0]) for side in (LEFT, RIGHT))
    pert = delta > 0 or sig > 0
    ctrl_i = ctrl_ii = None
    if pert and ((ci and sig > 0) or (a and b)):
        trc = sim(d, cond, False)
        ctrl_i = float(by(trc, T1)[:, SCORED].mean()); ctrl_ii = float(by(trc, T2)[:, SCORED].mean())
    if ci and sig > 0:
        ci = ctrl_i < 0.1
    cii = a and b and (not pert or ctrl_ii < 0.1)
    return {"i": ci, "ii": bool(cii), "a": a, "b": bool(b), "ctrl_i": ctrl_i, "ctrl_ii": ctrl_ii,
            "frac120_min": float(f120[:, SCORED].mean(0).min()), "frac240_min": float(f240[:, SCORED].mean(0).min()),
            "frac240_mean": float(f240[:, SCORED].mean())}


def gridjob(a):
    d, cond = a
    return score(d, cond)


def bracket(prop):
    ip = [i for i, p in enumerate(prop) if p]
    if not ip:
        return "none"
    i = ip[-1]
    return (G7[i], G7[i + 1]) if i + 1 < len(G7) else "above"


def bisjob(a):
    cond, brs = a
    cache = {}
    out = {}
    for crit, br in brs.items():
        if not isinstance(br, (list, tuple)):
            out[crit] = br; continue
        lo, hi = br
        for _ in range(NB):
            mid = 0.5 * (lo + hi)
            if mid not in cache:
                cache[mid] = score(mid, cond)
            if cache[mid][crit]:
                lo = mid
            else:
                hi = mid
        out[crit] = [lo, hi]
    return out, {str(k): v for k, v in cache.items()}


def gate():
    I0, th, De, gd = paperC_model.hom_params((16, 16), 8.0)
    sm = np.zeros((16, 16), np.bool_); sm[6:10, 6:10] = True
    for sig in (0.0, 0.03):
        a = run_gen2(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, 3000, DT, True, sig, 3)
        b = run_gen3(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, 3000, DT, True, sig, 3, sm, 10 ** 9)
        assert np.array_equal(a, b), "gate"
    print("GATE: run_gen3 (no early stop) == run_gen2 bit for bit", flush=True)


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def main():
    CKPT.mkdir(exist_ok=True)
    paperC_model.check()
    gate()
    z = np.load(HERE / "processed_data" / "paperC_tolerance.npz", allow_pickle=True)
    p6 = json.loads(str(z["__params__"]))
    g6, thr6 = p6["grids"], p6["thr"]
    G6 = [4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 20]
    pool = Pool(11)
    jobs = [(d, c) for c in CONDS for d in G7]
    R = ck("grids", lambda: pool.map(gridjob, jobs, chunksize=1))
    res = {}
    for (d, c), r in zip(jobs, R):
        res.setdefault(tuple(c), []).append(r)
    # consistency gate R: criterion (i) at shared grid points == round 6
    mism = []
    for c, v in res.items():
        old = g6[str((c[0], c[1], c[2], "hi"))]
        for d, r in zip(G7, v):
            if d in G6 and bool(old[G6.index(d)]) != r["i"]:
                mism.append((c, d, bool(old[G6.index(d)]), r["i"]))
    print(f"R (round-6 replication at shared points): {'PASS' if not mism else 'FAIL ' + str(mism)}", flush=True)
    for c, v in res.items():
        print(f"{c}: (i) " + "".join("+" if r["i"] else "." for r in v) + "  (ii) " + "".join("+" if r["ii"] else "." for r in v)
              + "  minfrac240 " + " ".join(f"{r['frac240_min']:.2f}" for r in v), flush=True)
    brs = {c: {crit: bracket([r[crit] for r in v]) for crit in ("i", "ii")} for c, v in res.items()}
    keys = list(brs)
    B = ck("bisect", lambda: pool.map(bisjob, [(c, brs[c]) for c in keys], chunksize=1))
    pool.close()
    edge = {}
    for c, (o, _) in zip(keys, B):
        for crit, v in o.items():
            edge[(c, crit)] = 0.5 * sum(v) if isinstance(v, list) else (np.inf if v == "above" else np.nan)
    lo6 = lambda c: thr6.get(str((c[0], c[1], c[2], "lo")), np.nan)
    def lev(k, v):
        return [(v, r, 0.0) for r in REAL] if k == "delta" else [(0.0, r, v * SIG_A) for r in REAL]
    out = {"R": not mism, "R_mismatch": [str(m) for m in mism]}
    reading = {}
    for crit in ("i", "ii"):
        for k, v in LEVELS:
            e = [edge[(c, crit)] for c in lev(k, v)]; l = [lo6(c) for c in lev(k, v)]
            nprop = sum(1 for c in lev(k, v) if not any(r[crit] for r in res[c]))
            surv = all(np.isfinite(x) and x > y for x, y in zip(e, l))
            reading[f"{k}{v}_{crit}"] = "SURVIVES" if surv else ("CLOSES" if nprop >= 3 else "MIXED")
    W = all(all(np.isfinite(edge[(c, crit)]) and edge[(c, crit)] > lo6(c) for k, v in LEVELS for c in lev(k, v))
            for crit in ("i", "ii"))
    N = all(lo6(c) < edge[(c, crit)] <= 4 for crit in ("i", "ii") for k, v in LEVELS for c in lev(k, v)
            if np.isfinite(edge[(c, crit)]))
    med = {(k, v, crit): float(np.median([edge[(c, crit)] for c in lev(k, v)])) for k, v in LEVELS for crit in ("i", "ii")}
    ratio = {f"{k}{v}": med[(k, v, "ii")] / med[(k, v, "i")] for k, v in LEVELS}
    CE = all(np.isfinite(x) and x <= 1.25 for x in ratio.values())
    for k, v in LEVELS:
        x = ratio[f"{k}{v}"]
        reading[f"{k}{v}_CE"] = "CRITERION EFFECT" if x >= 1.5 else ("partial" if x > 1.25 else "no")
    S = {}
    for crit in ("i", "ii"):
        dm = [med[("delta", v, crit)] for v in (0.1, 0.2, 0.4)]
        sc = [m * (1 + v) for m, v in zip(dm, (0.1, 0.2, 0.4))]
        dr = float(np.mean(sc))
        ok = all(np.isfinite(sc))
        S[f"S1{crit}"] = bool(ok and all(abs(s - dr) <= 0.15 * dr for s in sc) and dm[0] >= dm[1] >= dm[2])
        fm = [med[("f", v, crit)] for v in (0.25, 0.5)]
        S[f"S2{crit}"] = bool(ok and all(np.isfinite(fm)) and all(abs(m - dr) <= 0.15 * dr for m in fm))
        S[f"S3{crit}"] = bool(ok and 2 < dr <= 4)
        S[f"d_r_{crit}"] = dr
    print(f"edges: {{{', '.join(f'{c}{crit}: {x:.3f}' for (c, crit), x in edge.items())}}}", flush=True)
    print(f"medians {med}\nratio ii/i {ratio}\nW {W}  N {N}  CE {CE}  {S}\nreading {reading}", flush=True)
    out.update({"model": "run_gen3 (= run_gen2 + early stop), 2-D strip 32 x 256, settling rule, DT/8, T 240 s",
                "W": bool(W), "N": bool(N), "CE": bool(CE), **S, "reading": reading,
                "medians": {str(k): v for k, v in med.items()}, "ratio": ratio,
                "edge": {str(k): v for k, v in edge.items()}, "d_low6": {str(c): lo6(c) for c in CONDS},
                "brackets": {str(k): v for k, v in brs.items()}, "grid": G7, "nb": NB, "sig_A": SIG_A,
                "grids": {str(k): v for k, v in res.items()}, "bisect_runs": {str(c): b[1] for c, b in zip(keys, B)}})
    f = save_result(HERE / "processed_data" / "paperC_tolerance2.npz", out, d0_i=np.array(edge[((0.0, 1, 0.0), "i")]))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
