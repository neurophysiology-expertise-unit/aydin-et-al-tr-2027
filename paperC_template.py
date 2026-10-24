"""
paperC_template.py — paper C, round 8.
Written AFTER paperC_tolerance2.npz (round 7, bfb5f51) and BEFORE any run of this script.

WHY.
 Item 10 (CNN instance). Written as a cellular-network feedback template acting on the output, the model's coupling on
   the axis chain is c_i = sum_k A_k b_{i+k} with A = [D, p0 - 2D, D], p0 = 0 (the cell has no output self-feedback;
   its nonlinearity is in its state equation). For the cellular nonlinear network with a fixed output function of the
   state, y = f(x), and a symmetric template, the classical results show propagation failure only at weak coupling
   (Thiran-Setti-Hasler 1998; Chua et al. 1995). Question: does the upper edge come from the GATE (the output is
   zero until the cell fires and its gain is fixed at the firing event) or from output coupling as such?
 Item 13 (self-drain). The template's centre term -2D b_i drains a fired cell's output into its unfired neighbours.
   Question: is the upper edge still there without it (template [D, 0, D], input = D sum_j b_j)?
 Item 14. The front-speed result (S1, round 2) was measured under the original release rule at DT/8 (edge 12.39),
   which round 3 showed not to converge. Repeat it under the settling rule at DT/32.

INTEGRATOR. run_gx8 = paperC_design.run_gx VERBATIM plus one documented addition, mode 2 = UNGATED output
b = sigma(C) (no gain, no refractory gate: the standard cellular-network output, a fixed function of the state).
Modes 0 (the model, b = sigma(C) g (1 - ref)) and 1 (binary) are unchanged. The template enters through `coef`
(run_gx's stencil argument): FD2 = [-2, 1] is [D, -2D, D]; [0, 1] is [D, 0, D] (no self-drain).
Settling release rule throughout (recompute = True), as recommended in the paper.
GATE (asserted first): run_gx8 == run_gx bit for bit (t_rise) for mode 0 and 1, recompute False and True, FD2 and
[0, 1]; and paperC_model.check().

PROTOCOL (paperC_design.chain): axis planar front as the 1-D chain, N = 256, seed 5 rows, T = 120 s, dt = DT/32,
homogeneous parameters (hom_params), gamma = 1, g0 = 0, noise off; PROPAGATES := extent >= 40 rows.
Grid G8 = {0.25, 0.5, 1, 2, 4, 8, 11.22, 16, 32, 64, 128, 256}.
Variants:
 M  (the model): gated output (mode 0), template [D, -2D, D].
 U  (ungated CNN output): mode 2, template [D, -2D, D].
 N  (no self-drain): gated output (mode 0), template [D, 0, D].

PRE-REGISTERED PREDICTIONS
 R8 (reproduction, reference variant M): M propagates at d = 2, 4, 8 and fails at every d >= 16 in G8
    (paperC_design: settling edge 11.14 at DT/32; lower edge 0.54).
 T1 (the gate sets the upper edge): U propagates at every d >= 2 in G8 (no upper edge up to 256).
 T2 (the self-drain is needed for the upper edge): N propagates at every d >= 2 in G8 (no upper edge up to 256).
 SP (item 14; S1 repeated under the settling rule): axis chain, N = 512, seed w = 40, T = 120 s, dt = DT/32,
    variant M, d = d*(1 - eps) with d* = 11.1433 (paperC_design R1_dt32), eps in {0.3 0.2 0.1 0.05 0.02 0.01 0.005
    0.002}; speed = 1 / slope of the least-squares line t_rise vs distance over fired rows with 10 <= distance <= 40
    (>= 20 rows; paperC_edges.sim). PASS iff every run has a speed and v(eps = 0.002) >= 0.5 v(eps = 0.3)
    (the S1 criterion). Reported descriptively: whether v rises monotonically as eps falls.
 Descriptive (not predictions): the outcome at d < 2 for U and N (lower edges are not located).
Failed predictions are reported as failed (Supplement S2); criteria are not revised after the data land.
Output: processed_data/paperC_template.npz.   Run: python -s paperC_template.py
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
from paperC_continuum import chain_setup
from paperC_pde import scalars, STENCIL
from paperC_design import run_gx

HERE = Path(__file__).resolve().parent
G8 = [0.25, 0.5, 1, 2, 4, 8, 11.22, 16, 32, 64, 128, 256]
NOSELF = np.array([0.0, 1.0])
VARIANTS = {"M": (0, STENCIL["FD2"]), "U": (2, STENCIL["FD2"]), "N": (0, NOSELF)}
D_STAR = 11.143300781250002          # paperC_design.npz thr R1_dt32
EPS = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]


@njit(fastmath=True, cache=False)
def run_gx8(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, coef, dx, dt, recompute, g0, mode):
    # paperC_design.run_gx verbatim + mode 2 (ungated output b = sigma(C))
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
        elif mode == 2:
            broadcast = C_active.copy()
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
            elif mode == 2:
                broadcast = C_active.copy()
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


def run(d, mode, coef, N=256, w=5, T=120.0, dtf=32):
    sm, dist = chain_setup(N, w)
    I0, th, De, gd = scalars(d)
    dt = DT / dtf
    tr = run_gx8(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(round(T / dt)), coef, 1.0, dt, True, 0.0, mode)
    return tr, sm, dist


def gridjob(a):
    v, d = a
    mode, coef = VARIANTS[v]
    tr, sm, dist = run(d, mode, coef)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    return [ext, bool(ext >= 40)]


def speedjob(eps):
    d = D_STAR * (1 - eps)
    tr, sm, dist = run(d, 0, STENCIL["FD2"], N=512, w=40)
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    band = fired & (dist >= 10) & (dist <= 40)
    v = np.nan
    if band.sum() >= 20:
        sl = np.polyfit(dist[band], tr[band], 1)[0]
        v = 1.0 / sl if sl > 0 else np.nan
    return [d, ext, float(v)]


def gate():
    sm, _ = chain_setup(64, 5)
    for d in (8.0, 14.0):
        s = scalars(d)
        for coef in (STENCIL["FD2"], NOSELF):
            for mode in (0, 1):
                for rc in (False, True):
                    a = run_gx(s[0], s[1], s[2], s[3], THETA, sm, 1.0, TAU_REF, 4000, coef, 1.0, DT, rc, 0.0, mode)
                    b = run_gx8(s[0], s[1], s[2], s[3], THETA, sm, 1.0, TAU_REF, 4000, coef, 1.0, DT, rc, 0.0, mode)
                    assert np.array_equal(a, b), ("gate", d, mode, rc)
    print("GATE: run_gx8 (modes 0, 1) == paperC_design.run_gx bit for bit", flush=True)


def main():
    paperC_model.check()
    gate()
    jobs = [(v, d) for v in VARIANTS for d in G8]
    with Pool(11) as pool:
        R = pool.map(gridjob, jobs, chunksize=1)
        S = pool.map(speedjob, EPS, chunksize=1)
    res = {v: [r for (vv, _), r in zip(jobs, R) if vv == v] for v in VARIANTS}
    for v, rr in res.items():
        print(f"{v}: " + " ".join(f"{d:g}:{r[0]:.0f}{'+' if r[1] else '.'}" for d, r in zip(G8, rr)), flush=True)
    prop = {v: {d: r[1] for d, r in zip(G8, rr)} for v, rr in res.items()}
    R8 = all(prop["M"][d] for d in (2, 4, 8)) and not any(prop["M"][d] for d in G8 if d >= 16)
    T1 = all(prop["U"][d] for d in G8 if d >= 2)
    T2 = all(prop["N"][d] for d in G8 if d >= 2)
    v = np.array([s[2] for s in S])
    SP = bool(np.isfinite(v).all() and v[-1] >= 0.5 * v[0])
    mono = bool(np.isfinite(v).all() and np.all(np.diff(v) >= 0))
    print(f"speed (eps, d, ext, v): {[(e, round(s[0], 4), s[1], round(s[2], 4)) for e, s in zip(EPS, S)]}", flush=True)
    print(f"R8 {R8}  T1 {T1}  T2 {T2}  SP {SP}  (monotone rise: {mono})", flush=True)
    out = save_result(HERE / "processed_data" / "paperC_template.npz",
                      {"model": "run_gx8 (= paperC_design.run_gx + mode 2 ungated output); settling rule, DT/32",
                       "R8": bool(R8), "T1": bool(T1), "T2": bool(T2), "SP": SP, "SP_monotone": mono,
                       "grid": G8, "variants": {k: [m, list(map(float, c))] for k, (m, c) in VARIANTS.items()},
                       "results": res, "d_star": D_STAR, "eps": EPS, "speed": S},
                      v_settle=v, eps=np.array(EPS))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
