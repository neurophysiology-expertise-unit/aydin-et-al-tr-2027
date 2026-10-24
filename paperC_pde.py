"""
paperC_pde.py — paper C, round 4: a DIRECT solve of the continuum equation of
the planar front, independent of the lattice stencil, to back C3 (paperC_continuum: no re-entry up to d_lat 2048).
Written AFTER paperC_dt.npz (314db70) and BEFORE any run of this script.

CONTINUUM EQUATION (1-D, the planar front; same unit, gate, refractory clamp and gain-event rule, gamma = 1):
  C_t = C - C^3/3 - h + I0 + gd + D d_xx b - K_REF ref (C - C_DOWN),   h_t = (C + A - B h)/tau_h,
  b = sigma(C) g (1 - ref),  sigma = (1 + tanh(ETA (C - theta)))/2,
  g := min(1, G_FLOOR + gamma (D d_xx b)/SCALE) at the instant sigma rises through 1/2 (g = 0 before a point fires),
  ref := tau_ref when sigma falls through 1/2, counting down.
Homogeneous parameters = hom_params; D = Deff(d) = 0.275 d (the lattice coupling of multiplier d, per cell^2);
x in lattice-cell units, so the lattice chain is the 3-point (FD2) discretisation of this equation at dx = 1.

ANALYTIC NOTE (written before the runs; not a test). A point that has not fired has g = 0, so b = 0 there whatever C is;
at the instant it fires b becomes sigma(theta) g = g/2. Hence at every front b has a jump of height ~ g/2 between the
fired and the not-yet-fired region, and D d_xx b at a moving jump is not a function: on any consistent grid the
drain on the newly fired point grows like D (g/2) / dx^2. On the lattice dx = 1 is fixed and the drain is finite
(the strong-coupling edge is where it wins); under refinement it diverges. So the continuum equation as written is
predicted to carry NO front for any D: the C3 result is a property of the gated gain (b jumps at the event), not of
the 3-point stencil.

METHOD. run_pde: explicit Euler (as the model), periodic domain of 64 cells (x in [-32, 32)), central second-derivative
stencils FD4 (5-point, 4th order) and FD6 (7-point, 6th order) — independent of the lattice 3-point stencil — on grids
dx in {1, 1/2, 1/4, 1/8, 1/16} (d = 32: down to 1/8). The event rule is the model's, verbatim in structure (g from the
coupling of the step on which sigma crosses 1/2; ref entered on the falling step). dt_s = min(DT, 0.4 * 2/(4 D lambda)),
lambda = sum|stencil|/dx^2 (Gershgorin bound with sigma' <= ETA/2 = 4). Seed: C = 1.5, g = 1 on |x| < w/2,
w in {5, 20} cells. T = 60 s. extent = largest distance (cells) from the seed edge of any point that fired;
PROPAGATES := extent >= 20 cells. d in {2, 4, 8, 16, 32} (the lattice window at DT is 0.54 < d < 14.65).
GATE (asserted before anything runs): run_pde with FD2, dx = 1, dt = DT reproduces paperC_continuum.run_chain on the
axis chain (N = 64, w = 5, T = 60 s) at d = 8, 14, 16: same extent and t_rise within 2 DT at every point. If the
gate fails nothing is run.

PRE-REGISTERED PREDICTIONS
 P1 (no continuum front; decisive): for every d, with FD4 and FD6, at the two finest grids (dx = 1/8 and 1/16;
    d = 32: 1/4 and 1/8), at dt_s AND dt_s/4, and for both seeds: NO run propagates. PASS iff all such runs fail.
 P1b (descriptive, no prediction): the coarse grids dx = 1, 1/2, 1/4 with FD4/FD6 (does a higher-order stencil on a
    coarse grid carry a front, as the 3-point lattice does?).
 P3 (DIAGNOSTIC, labelled as such: the lattice dt drift). In the model's rule g is set AFTER coupling_in is computed,
    so on its firing step a point is integrated with its own b = 0: one 'free' step without self-drain whose length
    is dt. Variant 'recompute': coupling_in recomputed from the updated g before C is advanced (no free step).
    Lattice axis chain (FD2, dx = 1, N = 256, w = 5, T = 120 s, P = 40 rows as paperC_dt), threshold bisected at
    dt = DT/2, DT/8, DT/32 with the variant.
    P3a: |d*(DT/2) - d*(DT/32)| / d*(DT/32) <= 1 % (the free step drives the drift).
    P3b: |d*(DT/32) - 10.97| / 10.97 <= 5 % (the variant sits at the dt -> 0 extrapolation of the original rule,
         paperC_dt fit d0 = 10.97).
Failed predictions are reported as failed; criteria are not revised after the data land.
Output: processed_data/paperC_pde.npz. Checkpoints /tmp/paperC_pde_ckpt.   Run: python -s paperC_pde.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from core.model import DT, ETA, A_FHN, B_FHN
from core.provenance import save_result
import paperC_model
from paperC_model import hom_params, THETA, TAU_REF, C_DOWN, K_REF, SCALE, G_FLOOR
from paperC_continuum import run_chain, chain_setup, NBIS

CKPT = Path("/tmp/paperC_pde_ckpt")
STENCIL = {"FD2": np.array([-2.0, 1.0]),
           "FD4": np.array([-5 / 2, 4 / 3, -1 / 12]),
           "FD6": np.array([-49 / 18, 3 / 2, -3 / 20, 1 / 90])}
DS = [2, 4, 8, 16, 32]
DXS = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]
WS = [5, 20]
LDOM, TP, PP = 64, 60.0, 20.0
G_P3 = [8, 9, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14, 15]


@njit(fastmath=True, cache=False)
def run_pde(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, coef, dx, dt, recompute):
    # paperC_continuum.run_chain loop; Laplacian = central stencil coef (c0, c1, c2, ...)/dx^2; optional recompute
    nx = seedmask.shape[0]; hw = coef.shape[0] - 1; inv = 1.0 / (dx * dx)
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
            acc = coef[0] * broadcast[i]
            for k in range(1, hw + 1):
                acc = acc + coef[k] * (broadcast[(i + k) % nx] + broadcast[(i - k) % nx])
            lap[i] = acc * inv
        coupling_in = Deff * lap
        rose = active * (1.0 - prev)
        fell = prev * (1.0 - active)
        g_new = np.minimum(1.0, G_FLOOR + gamma_regen * (coupling_in / SCALE))
        g = np.where(rose > 0.5, np.maximum(g_new, 0.0), g)
        reft = np.where(fell > 0.5, tau_ref, reft)
        if recompute:
            broadcast = C_active * g * (1.0 - inref)
            for i in range(nx):
                acc = coef[0] * broadcast[i]
                for k in range(1, hw + 1):
                    acc = acc + coef[k] * (broadcast[(i + k) % nx] + broadcast[(i - k) % nx])
                lap[i] = acc * inv
            coupling_in = Deff * lap
        clamp = K_REF * inref * (C - C_DOWN)
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + dt * dC
        h = h + dt * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - dt, 0.0)
        prev = active
        for i in range(nx):
            if rose[i] > 0.5 and t_rise[i] < 0.0:
                t_rise[i] = t * dt
    return t_rise


def scalars(d):
    I0, th, De, gd = hom_params((1, 1), d)
    return I0[0, 0], th[0, 0], De[0, 0], gd[0, 0]


def dt_stable(st, d, dx):
    lam = (abs(STENCIL[st][0]) + 2 * np.abs(STENCIL[st][1:]).sum()) / dx ** 2
    return min(DT, 0.4 * 2.0 / (4.0 * scalars(d)[2] * lam))


def sim(st, d, dx, w, dtdiv, recompute=False, ncell=LDOM, T=TP, P=PP):
    """extent (cells) and propagates."""
    n = int(round(ncell / dx)); x = (np.arange(n) - n // 2) * dx
    sm = np.abs(x) < w / 2
    dist = np.maximum(np.abs(x) - w / 2, 0.0)
    I0, th, De, gd = scalars(d)
    dt = dt_stable(st, d, dx) / dtdiv
    tr = run_pde(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T / dt)), STENCIL[st], dx, dt, recompute)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    return ext, bool(ext >= P)


def job(a):
    return sim(*a)


def p3_run(args):
    d, dtf = args
    I0, th, De, gd = scalars(d)
    N = 256; sm, dist = chain_setup(N, 5); dt = DT / dtf
    tr = run_pde(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(120.0 / dt)), STENCIL["FD2"], 1.0, dt, True)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    return ext, bool(ext >= 40)


def p3_bisect(args):
    dtf, lo, hi = args
    for _ in range(NBIS):
        mid = 0.5 * (lo + hi)
        if p3_run((mid, dtf))[1]:
            lo = mid
        else:
            hi = mid
    return [lo, hi]


def gate():
    N, w = 64, 5
    sm, dist = chain_setup(N, w)
    for d in (8.0, 14.0, 16.0):
        I0, th, De, gd = hom_params((N, 1), d)
        a = run_chain(I0.ravel(), th.ravel(), De.ravel(), gd.ravel(), THETA, sm, 1.0, TAU_REF, int(TP / DT),
                      np.array([1, -1], np.int64), 2, DT)
        s = scalars(d)
        b = run_pde(s[0], s[1], s[2], s[3], THETA, sm, 1.0, TAU_REF, int(TP / DT), STENCIL["FD2"], 1.0, DT, False)
        ea = dist[(a >= 0) & ~sm].max() if ((a >= 0) & ~sm).any() else 0
        eb = dist[(b >= 0) & ~sm].max() if ((b >= 0) & ~sm).any() else 0
        assert ea == eb and np.array_equal(a >= 0, b >= 0), ("gate extent", d, ea, eb)
        m = a >= 0
        assert np.max(np.abs(a[m] - b[m])) <= 2 * DT + 1e-12, ("gate t_rise", d)
    print("GATE: run_pde(FD2, dx 1, DT) reproduces run_chain (extent, t_rise within 2 DT) at d 8, 14, 16", flush=True)


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def main():
    CKPT.mkdir(exist_ok=True)
    paperC_model.check()
    gate()
    pool = Pool(11)
    fine = lambda d: ([1 / 4, 1 / 8] if d == 32 else [1 / 8, 1 / 16])
    jobs = []
    for st in ("FD4", "FD6"):
        for d in DS:
            for dx in DXS:
                if d == 32 and dx < 1 / 8:
                    continue
                for w in WS:
                    jobs.append((st, d, dx, w, 1))
                    if dx in fine(d):
                        jobs.append((st, d, dx, w, 4))
    jobs.sort(key=lambda j: -(j[1] / j[2] ** 2 * j[4] / j[2]))       # longest first
    R = ck("pde", lambda: pool.map(job, jobs, chunksize=1))
    res = {tuple(j): r for j, r in zip(jobs, R)}
    for st in ("FD4", "FD6"):
        for d in DS:
            row = []
            for dx in DXS:
                for w in WS:
                    for dv in (1, 4):
                        k = (st, d, dx, w, dv)
                        if k in res:
                            row.append(f"dx1/{1/dx:g} w{w} dt/{dv}:{res[k][0]:.1f}{'+' if res[k][1] else '.'}")
            print(f"{st} d {d}: " + "  ".join(row), flush=True)
    P1 = bool(not any(res[k][1] for k in res if k[2] in fine(k[1])))
    # P3 diagnostic
    g3 = ck("p3_grid", lambda: pool.map(p3_run, [(d, f) for f in (2, 8, 32) for d in G_P3], chunksize=1))
    br = {}
    for a, f in enumerate((2, 8, 32)):
        pr = [r[1] for r in g3[a * len(G_P3):(a + 1) * len(G_P3)]]
        print(f"P3 grid DT/{f}: " + " ".join(f"{d}:{r[0]:.0f}{'+' if r[1] else '.'}" for d, r in
                                            zip(G_P3, g3[a * len(G_P3):(a + 1) * len(G_P3)])), flush=True)
        ip = [i for i, p in enumerate(pr) if p]
        if ip and ip[-1] + 1 < len(G_P3):
            br[f] = (G_P3[ip[-1]], G_P3[ip[-1] + 1])
    B = ck("p3_bisect", lambda: dict(zip([str(f) for f in br],
                                         pool.map(p3_bisect, [(f, *br[f]) for f in br], chunksize=1))))
    pool.close()
    th3 = {int(f): 0.5 * sum(v) for f, v in B.items()}
    for f, v in B.items():
        print(f"  P3 variant d* DT/{f}: ({v[0]:.5f}, {v[1]:.5f}]", flush=True)
    P3a = bool(2 in th3 and 32 in th3 and abs(th3[2] - th3[32]) / th3[32] <= 0.01)
    P3b = bool(32 in th3 and abs(th3[32] - 10.97) / 10.97 <= 0.05)
    print(f"P1 {P1}  P3a {P3a}  P3b {P3b}", flush=True)
    keys = list(res)
    out = save_result(Path(__file__).resolve().parent / "processed_data" / "paperC_pde.npz",
                      {"model": "run_pde: continuum planar-front equation, FD4/FD6, explicit Euler, model event rule",
                       "P1": P1, "P3a": P3a, "P3b": P3b, "p3_thr": th3, "p3_brackets": B, "domain_cells": LDOM,
                       "T": TP, "P_cells": PP, "ds": DS, "dxs": DXS, "ws": WS},
                      pde_keys=np.array([[k[0], k[1], k[2], k[3], k[4]] for k in keys], dtype=object).astype(str),
                      pde_ext=np.array([res[k][0] for k in keys]), pde_prop=np.array([res[k][1] for k in keys]),
                      pde_dt=np.array([dt_stable(k[0], k[1], k[2]) / k[4] for k in keys]),
                      p3_grid=np.array(g3), p3_d=np.array(G_P3))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
