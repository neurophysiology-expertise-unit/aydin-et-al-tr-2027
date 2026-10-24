"""
paperC_design.py — paper C, round 5: design rules for event-gated output coupling.
Written AFTER paperC_pde.npz (d265a31) and BEFORE any run of this script.

INTEGRATOR. run_gx = paperC_pde.run_pde (asserted against the lattice chain there) with two documented additions:
 g0   : gain of every cell that has not fired (the model: 0). The output of an unfired cell is sigma(C) g0, so the jump
        of the output at firing shrinks from g/2 (g0 = 0) to (g - g0)/2: g0 is the one parameter that sets how sharp
        the jump is. Seed cells keep g = 1.
 mode : 0 = the model's output b = sigma(C) g (1 - ref); 1 = BINARY event-gated output b = 1[sigma(C) > 1/2] g (1 - ref)
        (a second, different gated rule: the output jumps by g at firing, twice the model's g/2).
All runs use the SETTLING release rule (recompute = True: the coupling is recomputed after g is set, so a newly fired
cell drains its own output from its firing step on; paperC_pde P3, called 'no free step' there).
GATE (asserted first): run_gx(g0 = 0, mode 0) reproduces run_pde bit for bit (t_rise) with recompute False and True.

Common protocol: axis planar front as the 1-D chain (FD2, dx = 1, N = 256, seed 5 rows, T = 120 s, PROPAGATES :=
extent >= 40 rows), homogeneous parameters (hom_params), gamma = 1, noise off; edges by grid then 8 bisection
halvings; dt = DT/32 unless stated. Continuum runs: FD4, 64-cell domain, seed 20 cells, T = 60 s, P = 20 cells,
dt_s = stability-limited step (paperC_pde.dt_stable) and dt_s/4.

PRE-REGISTERED PREDICTIONS
Step 4 — the recommended (settling) rule at converged dt:
 R1 strong edge converged: |d*(DT/128) - d*(DT/32)| / d*(DT/32) <= 1 %.
 R2 weak edge unchanged by the rule and converged: settling-rule d_low at DT and DT/4 both within 1 % of the original
    rule's 0.5385 (paperC_edges L0), using paperC_edges' weak-edge protocol (seed 20, T = 1200 s, P = 40).
Steps 2/3 — the output jump sets the strong edge:
 J1 (jump law) d*(g0) (1 - g0) lies within 15 % of d*(0) for g0 in {0.25, 0.5, 0.75, 0.875}.
 J2 (second rule, binary output, jump g instead of g/2): d*_bin within 15 % of d*(0)/2; and a weak edge exists
    (the binary chain fails at d = 0.25 and propagates at d = 2).
 J3 (no jump, g0 = 1, lattice): the chain propagates at every d in {8, 16, 32, 64, 128, 256} (no strong edge).
 J4 (no jump, continuum): g0 = 1, FD4, dx in {1/8, 1/16} (d = 32: {1/4, 1/8}, as paperC_pde P1), dt_s and dt_s/4,
    d in {2, 8, 32}: every run propagates
    (continuum fronts return when the jump is removed).
 J5 (transition map, continuum): g0 in {0.5, 0.75, 0.875}, FD4, d = 8, dx in {1, 1/2, 1/4, 1/8, 1/16}, dt_s:
    a run propagates iff d/dx^2 < d*(g0) (the lattice-unit edge measured in J1 with FD2); PASS iff at least 13 of the
    15 runs are classified correctly.
 J6 (second rule, continuum): binary output, FD4, same grids as J4, dt_s and dt_s/4, d in {2, 8, 32}: no run
    propagates.
Step 3 — analytic strong-edge criteria (axis, g0 = 0), compared with d*(0) at DT/32:
 A-static: a newly fired cell with one fully active upstream neighbour (b = 1) and s = 2 non-co-firing neighbours
    sustains its excited state iff max_C [F(C)] > D (s - u), F(C) = C - C^3/3 - h_r + I0 + g_d, u = 1, h_r = rest
    value -0.2115; max F = F(1) = 1.304, so D* = 1.304 and d*_static = 1.304 / 0.27494 = 4.74.
    PASS iff 4.74 lies within 15 % of d*(0).
 A-race (reduced two-cell handoff): cells n and n+1 integrated with the model equations and the settling rule, n-1
    frozen at b = 1 and n+2 frozen at b = 0; n starts just fired (C = theta + 0.01, g = 1, h = h_r), n+1 at rest
    (C_r, h_r, g = 0). The handoff succeeds iff n+1 fires before n falls back through theta. d*_race by bisection
    (dt = DT/32). PASS iff d*_race lies within 15 % of d*(0).
DISCLOSURE: before this commit a smoke test called race() at d = 5 and d = 30 (both: handoff succeeded). A-race was
already written as above and is left unchanged.
Failed predictions are reported as failed; criteria are not revised after the data land.
Output: processed_data/paperC_design.npz. Checkpoints /tmp/paperC_design_ckpt.   Run: python -s paperC_design.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from core.model import DT, ETA, A_FHN, B_FHN
from core.provenance import save_result
import paperC_model
from paperC_model import THETA, TAU_REF, C_DOWN, K_REF, SCALE, G_FLOOR
from paperC_continuum import chain_setup, NBIS
from paperC_pde import run_pde, scalars, dt_stable, STENCIL

CKPT = Path("/tmp/paperC_design_ckpt")
G0S = [0.0, 0.25, 0.5, 0.75, 0.875]
FAC = [0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
D_REF = 11.22                       # paperC_pde P3 settling rule at DT/32 (grid centre only, not a prediction)
G_LOW = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2]


@njit(fastmath=True, cache=False)
def run_gx(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, coef, dx, dt, recompute, g0, mode):
    # paperC_pde.run_pde + g0 (gain of unfired cells) + mode (0 sigmoid output, 1 binary output)
    nx = seedmask.shape[0]; hw = coef.shape[0] - 1; inv = 1.0 / (dx * dx)
    C = np.full(nx, C_DOWN); h = (C + A_FHN) / B_FHN
    g = np.full(nx, g0); reft = np.zeros(nx); prev = np.zeros(nx)
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
        if mode == 1:
            broadcast = active * g * (1.0 - inref)
        else:
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
            if mode == 1:
                broadcast = active * g * (1.0 - inref)
            else:
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


@njit(fastmath=True, cache=False)
def race(I0, tau_h, Deff, gdrive, theta, hr, Cr, dt, tmax):
    # reduced handoff: cell 0 = n (just fired), cell 1 = n+1 (rest); b(n-1) = 1, b(n+2) = 0 frozen; settling rule
    C0 = theta + 0.01; h0 = hr; g0_ = 1.0
    C1 = Cr; h1 = hr; g1 = 0.0
    prev1 = 0.0
    t = 0.0
    while t < tmax:
        s0 = 0.5 * (1.0 + np.tanh(ETA * (C0 - theta))); s1 = 0.5 * (1.0 + np.tanh(ETA * (C1 - theta)))
        if s0 < 0.5:
            return 0                                   # n fell before n+1 fired
        a1 = 1.0 if s1 > 0.5 else 0.0
        if a1 > 0.5 and prev1 < 0.5:
            cin1 = Deff * (s0 * g0_ + 0.0 - 2.0 * s1 * g1)
            g1 = min(1.0, G_FLOOR + cin1 / SCALE)
            if g1 < 0.0:
                g1 = 0.0
            return 1                                   # n+1 fired first
        b0 = s0 * g0_; b1 = s1 * g1
        cin0 = Deff * (1.0 + b1 - 2.0 * b0); cin1 = Deff * (b0 + 0.0 - 2.0 * b1)
        dC0 = C0 - C0 ** 3 / 3.0 - h0 + I0 + gdrive + cin0
        dC1 = C1 - C1 ** 3 / 3.0 - h1 + I0 + gdrive + cin1
        C0 += dt * dC0; C1 += dt * dC1
        h0 += dt * (C0 + A_FHN - B_FHN * h0) / tau_h; h1 += dt * (C1 + A_FHN - B_FHN * h1) / tau_h
        prev1 = a1; t += dt
    return 0


def chain(d, dtf=32, g0=0.0, mode=0, recompute=True, w=5, T=120.0, P=40, N=256):
    sm, dist = chain_setup(N, w)
    I0, th, De, gd = scalars(d)
    dt = DT / dtf
    tr = run_gx(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T / dt)), STENCIL["FD2"], 1.0, dt, recompute, g0, mode)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    return ext, bool(ext >= P)


def cont(d, dx, dtdiv, g0=0.0, mode=0, w=20, ncell=64, T=60.0, P=20.0):
    n = int(round(ncell / dx)); x = (np.arange(n) - n // 2) * dx
    sm = np.abs(x) < w / 2; dist = np.maximum(np.abs(x) - w / 2, 0.0)
    I0, th, De, gd = scalars(d)
    dt = dt_stable("FD4", d, dx) / dtdiv
    tr = run_gx(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T / dt)), STENCIL["FD4"], dx, dt, True, g0, mode)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    return ext, bool(ext >= P)


def job(a):
    kind, kw = a
    return chain(**kw) if kind == "chain" else cont(**kw)


def bis(a):
    kw, lo, hi, upper = a
    for _ in range(NBIS):
        mid = 0.5 * (lo + hi)
        if chain(**{**kw, "d": mid})[1] == upper:
            lo = mid
        else:
            hi = mid
    return [lo, hi]


def rest():
    from scipy.optimize import brentq
    I0, th, De, gd = scalars(1.0); I = I0 + gd
    Cr = brentq(lambda C: C - C ** 3 / 3 - (C + A_FHN) / B_FHN + I, -3, 0)
    return Cr, (Cr + A_FHN) / B_FHN


def race_d(d):
    I0, th, De, gd = scalars(d); Cr, hr = rest()
    return race(I0, th, De, gd, THETA, hr, Cr, DT / 32, 60.0)


def gate():
    N = 64; sm, _ = chain_setup(N, 5)
    for d in (8.0, 14.0):
        s = scalars(d)
        for rc in (False, True):
            a = run_pde(s[0], s[1], s[2], s[3], THETA, sm, 1.0, TAU_REF, 4000, STENCIL["FD2"], 1.0, DT, rc)
            b = run_gx(s[0], s[1], s[2], s[3], THETA, sm, 1.0, TAU_REF, 4000, STENCIL["FD2"], 1.0, DT, rc, 0.0, 0)
            assert np.array_equal(a, b), ("gate", d, rc)
    print("GATE: run_gx(g0 0, mode 0) == run_pde bit for bit", flush=True)


def ck(name, fn):
    f = CKPT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = fn(); f.write_text(json.dumps(r)); return r


def bracket(grid, prop, upper=True):
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
    pool = Pool(11)
    # ---- grids ----
    grids = {}
    for g0 in G0S:
        grids[f"g0_{g0}"] = ([D_REF / (1 - g0) * f for f in FAC], dict(g0=g0), True)
    grids["bin"] = ([D_REF / 2 * f for f in FAC], dict(mode=1), True)
    for f in (32, 128):
        grids[f"R1_dt{f}"] = ([D_REF * x for x in FAC], dict(dtf=f), True)
    for f in (1, 4):
        grids[f"R2_dt{f}"] = (G_LOW, dict(dtf=f, w=20, T=1200.0), False)
    grids["bin_low"] = ([0.25, 2.0], dict(mode=1), None)
    grids["J3"] = ([8, 16, 32, 64, 128, 256], dict(g0=1.0), None)
    jobs = [(n, ("chain", {**kw, "d": x})) for n, (g, kw, up) in grids.items() for x in g]
    cj = []
    for d in (2, 8, 32):
        for dx in ((1 / 4, 1 / 8) if d == 32 else (1 / 8, 1 / 16)):
            for dv in (1, 4):
                cj.append((f"J4_d{d}_dx{dx:g}_dt{dv}", ("cont", dict(d=d, dx=dx, dtdiv=dv, g0=1.0))))
                cj.append((f"J6_d{d}_dx{dx:g}_dt{dv}", ("cont", dict(d=d, dx=dx, dtdiv=dv, mode=1))))
    for g0 in (0.5, 0.75, 0.875):
        for dx in (1, 1 / 2, 1 / 4, 1 / 8, 1 / 16):
            cj.append((f"J5_g{g0}_dx{dx:g}", ("cont", dict(d=8, dx=dx, dtdiv=1, g0=g0))))
    alljobs = sorted(jobs + cj, key=lambda j: 0 if j[1][0] == "chain" else -1)
    R = ck("grids", lambda: pool.map(job, [j[1] for j in alljobs], chunksize=1))
    res = {}
    for (n, _), r in zip(alljobs, R):
        res.setdefault(n, []).append(r)
    for n, (g, kw, up) in grids.items():
        print(f"{n:10s} " + " ".join(f"{x:.4g}:{r[0]:.0f}{'+' if r[1] else '.'}" for x, r in zip(g, res[n])), flush=True)
    for n, _ in cj:
        print(f"{n:22s} {res[n][0][0]:.1f}{'+' if res[n][0][1] else '.'}", flush=True)
    # ---- bisections + race ----
    bj = {n: (kw, *bracket(g, [r[1] for r in res[n]], up), up) for n, (g, kw, up) in grids.items()
          if up is not None and bracket(g, [r[1] for r in res[n]], up) is not None}
    print("no bracket:", [n for n, v in grids.items() if v[2] is not None and n not in bj], flush=True)
    B = ck("bisect", lambda: dict(zip(bj, pool.map(bis, list(bj.values()), chunksize=1))))
    rg = [D_REF * f for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2, 3, 5]]
    RR = ck("race_grid", lambda: pool.map(race_d, rg))
    pool.close()
    thr = {n: 0.5 * sum(v) for n, v in B.items()}
    for n, v in B.items():
        print(f"  {n:10s} ({v[0]:.5f}, {v[1]:.5f}]", flush=True)
    g_ = lambda n: thr.get(n, np.nan)
    rel = lambda a, b: abs(a - b) / b
    d0 = g_("g0_0.0")
    R1 = bool(rel(g_("R1_dt128"), g_("R1_dt32")) <= 0.01)
    R2 = bool(rel(g_("R2_dt1"), 0.53848) <= 0.01 and rel(g_("R2_dt4"), 0.53848) <= 0.01)
    J1v = {g0: g_(f"g0_{g0}") * (1 - g0) / d0 - 1 for g0 in G0S[1:]}
    J1 = bool(all(abs(v) <= 0.15 for v in J1v.values()))
    J2 = bool(rel(g_("bin"), d0 / 2) <= 0.15 and (not res["bin_low"][0][1]) and res["bin_low"][1][1])
    J3 = bool(all(r[1] for r in res["J3"]))
    J4 = bool(all(res[n][0][1] for n, _ in cj if n.startswith("J4")))
    J6 = bool(not any(res[n][0][1] for n, _ in cj if n.startswith("J6")))
    ok5 = 0
    for g0 in (0.5, 0.75, 0.875):
        for dx in (1, 1 / 2, 1 / 4, 1 / 8, 1 / 16):
            pred = 8 / dx ** 2 < g_(f"g0_{g0}")
            ok5 += int(pred == res[f"J5_g{g0}_dx{dx:g}"][0][1])
    J5 = bool(ok5 >= 13)
    As = bool(rel(4.74, d0) <= 0.15)
    rp = [r for r in RR]
    i = [k for k, r in enumerate(rp) if r == 1]
    d_race = np.nan
    if i and i[-1] + 1 < len(rg) and all(r == 1 for r in rp[:i[-1] + 1]):
        lo, hi = rg[i[-1]], rg[i[-1] + 1]
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if race_d(mid) == 1:
                lo = mid
            else:
                hi = mid
        d_race = 0.5 * (lo + hi)
    Ar = bool(np.isfinite(d_race) and rel(d_race, d0) <= 0.15)
    print(f"race grid: {list(zip([round(x, 2) for x in rg], rp))}  d*_race {d_race:.4f}", flush=True)
    print(f"d*(0) {d0:.4f}  J1 dev {J1v}  bin {g_('bin'):.4f}  J5 correct {ok5}/15", flush=True)
    print(f"R1 {R1} R2 {R2} J1 {J1} J2 {J2} J3 {J3} J4 {J4} J5 {J5} J6 {J6} A-static {As} A-race {Ar}", flush=True)
    out = save_result(Path(__file__).resolve().parent / "processed_data" / "paperC_design.npz",
                      {"model": "run_gx (= run_pde + g0 + binary mode; settling rule)", "R1": R1, "R2": R2,
                       "J1": J1, "J2": J2, "J3": J3, "J4": J4, "J5": J5, "J6": J6, "A_static": As, "A_race": Ar,
                       "thr": thr, "brackets": B, "J1_dev": J1v, "J5_correct": ok5, "d_race": d_race,
                       "d_static": 4.74, "race_grid": [rg, rp],
                       "results": {n: v for n, v in res.items()}, "grids": {n: [g, kw, up] for n, (g, kw, up) in grids.items()}},
                      d0=np.array(d0), d_race=np.array(d_race))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
