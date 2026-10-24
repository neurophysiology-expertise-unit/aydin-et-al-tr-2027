"""
paperC_model.py — the model code that the round scripts use.

Homogeneous FitzHugh-Nagumo excitable lattice with event-gated, self-draining output coupling (gain g set when a
cell fires) and a refractory state:
  run_gen    = the lattice integrator (per-cell parameter arrays, noise off) with a per-cell record of the first firing
  hom_params = every per-cell parameter replaced by its ensemble mean
The release and refractory constants are C_DOWN, K_REF, SCALE, G_FLOOR.

numba fastmath copies of an integrator can diverge from the original, so `check()` asserts that run_gen reproduces
stored results bit for bit: the planar-front extent on the axis chain (k = 1, d_lat = 8 and 16) stored in
processed_data/paperC_continuum.npz.
Run: python -s paperC_model.py      (prints 'paperC_model reproduces stored results')
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)

# release and refractory constants
C_DOWN = -1.2; K_REF = 20.0; SCALE = 0.15; G_FLOOR = 0.02
# protocol constants
L, STRIDE, PATCH, UM, ALPHA, TAU_REF, T_WIN = 64, 5, 2, 25.0, 0.01, 15.0, 120.0
STEPS = int(T_WIN / DT)
THETA = THETA_BASE + 0.7 * ALPHA
HERE = pathlib.Path(__file__).resolve().parent


@njit(fastmath=True, cache=False)
def run_gen(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, stride):
    # run_drm's loop (noise off) with per-cell arrays passed in and a per-cell record of the first firing
    nx, ny = I0.shape
    C = np.full((nx, ny), C_DOWN); h = (C + A_FHN) / B_FHN
    g = np.zeros((nx, ny)); reft = np.zeros((nx, ny)); prev = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if seedmask[i, j]:
                C[i, j] = 1.5; g[i, j] = 1.0
    prev = (0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5).astype(np.float64)
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    t_rise = np.full((nx, ny), -1.0); g_rise = np.zeros((nx, ny)); cin_rise = np.zeros((nx, ny))
    peak = np.full((nx, ny), -9.0); episode = np.zeros((nx, ny))
    fi = 0
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
        clamp = K_REF * inref * (C - C_DOWN)
        noise = np.zeros((nx, ny))
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + np.sqrt(DT) * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - DT, 0.0)
        prev = active
        # bookkeeping only (does not feed back)
        for i in range(nx):
            for j in range(ny):
                if rose[i, j] > 0.5 and t_rise[i, j] < 0.0:
                    t_rise[i, j] = t * DT; g_rise[i, j] = g[i, j]; cin_rise[i, j] = coupling_in[i, j]
                    episode[i, j] = 1.0
                if episode[i, j] > 0.5:
                    if C[i, j] > peak[i, j]:
                        peak[i, j] = C[i, j]
                    if fell[i, j] > 0.5:
                        episode[i, j] = 0.0
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld, t_rise, g_rise, cin_rise, peak


def hom_params(shape, d_mult):
    """every per-cell parameter replaced by its ensemble mean (numerical mean over 2e6 draws, fixed seed)."""
    rng = np.random.default_rng(0); n = 2_000_000
    gb = rng.uniform(0.05, 0.34, n) * (1 + 2 * rng.standard_normal(n))
    tb = rng.uniform(0.5, 1.1, n); D0 = rng.uniform(0.05, 0.5, n); kb = rng.uniform(1.0, 4.0, n)
    I0 = I0_BASE + ALPHA * 0.3
    tau_h = np.mean(10.0 / ((1.0 + 0.8 * ALPHA) * tb))
    Deff = d_mult * np.mean(D0 / (1.0 + (kb * ALPHA) ** 4))
    gd = ALPHA * gb.mean()
    f = lambda v: np.full(shape, v, dtype=np.float64)
    return f(I0), f(tau_h), f(Deff), f(gd)


def check():
    ref = np.load(HERE / "processed_data" / "paperC_continuum.npz")
    k = list(ref["ks"]).index(1); g = list(ref["g_lat"])
    N, w = 256, 5
    sm = np.zeros((N, 1), np.bool_); lo = N // 2 - w // 2; hi = lo + w - 1; sm[lo:hi + 1] = True
    idx = np.arange(N); dist = np.where(idx < lo, lo - idx, np.where(idx > hi, idx - hi, 0))
    for d in (8.0, 16.0):
        I0, th, De, gd = hom_params((N, 1), d)
        tr = run_gen(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, int(T_WIN / DT), 5)[1].ravel()
        fired = (tr >= 0) & ~sm.ravel()
        ext = float(dist[fired].max()) if fired.any() else 0.0
        assert ext == ref["ext_d"][k, g.index(d)], ("run_gen chain", d, ext, ref["ext_d"][k, g.index(d)])
    print("paperC_model reproduces stored results (paperC_continuum.npz, axis chain, d_lat 8 and 16)", flush=True)


if __name__ == "__main__":
    check()
