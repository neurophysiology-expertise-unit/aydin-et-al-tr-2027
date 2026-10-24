"""
paperC_direction.py — paper C follow-up (pre-registered after paperC_critical_radius.npz):
is the gamma_regen = 1 failure of the focal front on the HOMOGENEOUS lattice a curvature effect or a
lattice-DIRECTION effect?

What paperC_critical_radius.npz showed (L = 96, gamma = 1): no disk seed of radius 1..16 propagates for any
d >= 4 (R_c jumps from ~1.25 at d = 3.5 to > 16 at d = 4), while the axis-aligned planar stripe propagates up to
d = 12 and fails by d = 16. A smooth critical radius (H1, H2) was not found. The squares of paperC_geometry
(L = 64) that 'propagated' at d >= 4 carry axis-aligned flat faces.

EXACT REDUCTION (analytic). On a homogeneous lattice a planar front normal to the lattice direction (p, q)
(gcd 1) depends only on n = p*i + q*j. The 4-neighbour Laplacian of the broadcast b becomes
   b[n+p] + b[n-p] + b[n+q] + b[n-q] - 4 b[n],
so an AXIS front (1,0) is the 1-D chain with coupling D (two lateral partners fire with the cell: b[n] twice),
and a DIAGONAL front (1,1) is 2D*(b[n+1] + b[n-1] - 2 b[n]): the chain at coupling 2D. The regeneration gain
g' = min(1, G_FLOOR + gamma*coupling_in/SCALE) is a function of coupling_in and maps with it. For p > q >= 1
a fresh front cell has NO lateral partner firing with it; its two upstream neighbours fired p and q rows
earlier, and when the farther one has already entered its refractory state the cell receives about D against
a drain of 4*D*Phi, a worse balance than the axis (D vs 2*D*Phi) or the diagonal (2D vs 4*D*Phi).

MODEL: paperC_model.run_gen (verbatim), hom_params, gamma_regen = 1, noise off, tau_ref 15 s, 120 s.
 STRIPES: L = 192; band seed {n mod L < w_n}, n = p*i + q*j, w_n = round(5*sqrt(p^2+q^2)) (about 5 cells wide);
   directions (1,0), (1,1), (2,1), (3,1), (3,2); d in D_DIR. Normal distance of a cell from the band = (distance
   in n, both sides, torus) / sqrt(p^2+q^2). PROPAGATES := normal extent >= 20 cells.
 PAIRS (for D1): diagonal band with w_n = 5 at d in {3.5, 4, 5, 6, 7, 8} vs axis band (w_n = 5) at 2d, L = 192.
 SQUARES (descriptive, no prediction): L = 96, half-width 2, 4, 8, 16, d in {3.5, 4, 4.5, 5, 6, 8}; extent
   beyond the seed along the axes and along the diagonals (cells within 10 degrees of 0/90 or of 45 degrees).

PRE-REGISTERED PREDICTIONS (written 2026-10-12 after paperC_critical_radius.npz and before any run of this script):
 D1. Exact reduction: for each pair, same outcome, and the mean first-activation time per n-row agrees within
     2 time steps (2*DT) for every row reached by both.
 D2. Consequence of D1 with the measured axis failure coupling d*_axis in (12, 16]: the diagonal stripe
     propagates at d <= 6 and fails at d >= 8 (d*_diag = d*_axis / 2).
 D3. Direction explains the disk failure: at least one oblique direction among (2,1), (3,1), (3,2) propagates at
     d = 3.5 and fails at d = 4 or 4.5, i.e. the smallest failure coupling over the five directions lies in
     (3.5, 4.5]. If every direction still propagates at d = 4.5, the direction explanation is REJECTED for the
     disk failure (a curvature or finite-size effect then remains, unexplained).
Failed predictions are reported as failed; criteria are not revised after the data land.

Output: processed_data/paperC_direction.npz.   Run: python -s paperC_direction.py   (about 40 min on 11 cores)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from core.provenance import save_result
from paperC_model import run_gen, hom_params, check, THETA, TAU_REF, STEPS, STRIDE, T_WIN, DT

LS = 192
DIRS = [(1, 0), (1, 1), (2, 1), (3, 1), (3, 2)]
D_DIR = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0]
D_PAIR = [3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
SQ_W = [2, 4, 8, 16]
SQ_D = [3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
PROP = 20.0
CKPT = Path("/tmp/paperC_ckpt")


def band(p, q, wn):
    ii, jj = np.mgrid[0:LS, 0:LS]
    n = (p * ii + q * jj) % LS
    dn = np.minimum((n - (wn - 1)) % LS, (-n) % LS)        # n-distance to the band [0, wn-1], both sides
    seed = n < wn
    dn = np.where(seed, 0, dn)
    return seed, n, dn / np.hypot(p, q)


def job_stripe(args):
    p, q, d, wn = args
    seed, n, dist = band(p, q, wn)
    I0, th, De, gd = hom_params((LS, LS), d)
    fld, tr, g, cin, pk = run_gen(I0, th, De, gd, THETA, seed, 1.0, TAU_REF, STEPS, STRIDE)
    fired = (tr >= 0) & ~seed
    ext = float(dist[fired].max()) if fired.any() else 0.0
    # mean first-activation time per n-row on the forward side (n = wn, wn+1, ...), NaN if not reached
    rows = np.full(LS // 2, np.nan)
    for k in range(LS // 2):
        m = n == (wn + k) % LS
        if (tr[m] >= 0).all():
            rows[k] = tr[m].mean()
    sp = np.nan
    ds = np.arange(5, 19); tm = np.array([tr[fired & (np.round(dist) == k)].mean() if (fired & (np.round(dist) == k)).any()
                                          else np.nan for k in ds])
    ok = np.isfinite(tm)
    if ok.sum() >= 8:
        sl = np.polyfit(ds[ok], tm[ok], 1)[0]; sp = 1.0 / sl if sl > 0 else np.nan
    return ext, sp, rows


def job_square(args):
    w, d = args
    L = 96; c = L // 2
    sm = np.zeros((L, L), np.bool_); sm[c - w:c + w + 1, c - w:c + w + 1] = True
    I0, th, De, gd = hom_params((L, L), d)
    tr = run_gen(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, STEPS, STRIDE)[1]
    dist = distance_transform_edt(~np.tile(sm, (3, 3)))[L:2 * L, L:2 * L]
    ii, jj = np.mgrid[0:L, 0:L]; ang = np.degrees(np.arctan2(abs(ii - c), abs(jj - c)))
    fired = (tr >= 0) & ~sm
    ax = fired & ((ang <= 10) | (ang >= 80)); dg = fired & (abs(ang - 45) <= 10)
    f = lambda m: float(dist[m].max()) if m.any() else 0.0
    return f(fired), f(ax), f(dg)


def main():
    CKPT.mkdir(exist_ok=True)
    check()
    js = [(p, q, d, int(round(5 * np.hypot(p, q)))) for (p, q) in DIRS for d in D_DIR]
    jp = [(1, 1, d, 5) for d in D_PAIR] + [(1, 0, 2 * d, 5) for d in D_PAIR]
    jq = [(w, d) for w in SQ_W for d in SQ_D]
    ck = CKPT / "dir_runs.npy"
    if ck.exists():
        S, P, Q = np.load(ck, allow_pickle=True)
    else:
        with Pool(11) as pool:
            S = pool.map(job_stripe, js, chunksize=1)
            P = pool.map(job_stripe, jp, chunksize=1)
            Q = pool.map(job_square, jq, chunksize=1)
        a = np.empty(3, dtype=object); a[:] = [S, P, Q]; np.save(ck, a, allow_pickle=True)
    ext = np.array([s[0] for s in S]).reshape(len(DIRS), len(D_DIR))
    spd = np.array([s[1] for s in S]).reshape(len(DIRS), len(D_DIR))
    prop = ext >= PROP
    print("stripes (normal extent, cells; '+' propagates; speed cells/s):")
    dstar = {}
    for a, (p, q) in enumerate(DIRS):
        print(f"  ({p},{q}): " + " ".join(f"{d:g}:{ext[a,b]:4.1f}{'+' if prop[a,b] else '.'}({spd[a,b]:.2f})"
                                          for b, d in enumerate(D_DIR)), flush=True)
        f = np.where(~prop[a])[0]
        dstar[(p, q)] = (D_DIR[f[0] - 1] if len(f) and f[0] > 0 else (np.nan if len(f) else D_DIR[-1]),
                         D_DIR[f[0]] if len(f) else np.inf)
    print(f"failure coupling brackets (last propagating, first failing): {dstar}")
    # D1
    d1 = []
    for k, d in enumerate(D_PAIR):
        a, b = P[k], P[k + len(D_PAIR)]
        same = (a[0] >= PROP) == (b[0] >= PROP)
        both = np.isfinite(a[2]) & np.isfinite(b[2])
        md = float(np.max(np.abs(a[2][both] - b[2][both]))) if both.any() else np.nan
        ok = bool(same and both.any() and md <= 2 * DT)
        d1.append(ok)
        print(f"  D1 pair d {d} (diag) vs {2*d} (axis): outcome {a[0]:.1f}/{b[0]:.1f}  rows compared {int(both.sum())}"
              f"  max |dt| {md:.4g} s  -> {ok}")
    D1 = all(d1)
    di = DIRS.index((1, 1))
    D2 = bool(all(prop[di, b] for b, d in enumerate(D_DIR) if d <= 6) and
              all(not prop[di, b] for b, d in enumerate(D_DIR) if d >= 8))
    fails = [dstar[k][1] for k in DIRS]
    obl = [(2, 1), (3, 1), (3, 2)]
    i35, i40, i45 = D_DIR.index(3.5), D_DIR.index(4.0), D_DIR.index(4.5)
    D3 = bool(any(prop[DIRS.index(o), i35] and (not prop[DIRS.index(o), i40] or not prop[DIRS.index(o), i45])
                  for o in obl) and 3.5 < min(fails) <= 4.5)
    print(f"D1 {D1}  D2 {D2}  D3 {D3}  (smallest failure coupling {min(fails)})")
    Q = np.array(Q).reshape(len(SQ_W), len(SQ_D), 3)
    print("squares L=96 (beyond-seed extent all / along axes / along diagonals):")
    for a, w in enumerate(SQ_W):
        print(f"  w {w:2d}: " + " ".join(f"{d:g}:{Q[a,b,0]:.0f}/{Q[a,b,1]:.0f}/{Q[a,b,2]:.0f}" for b, d in enumerate(SQ_D)))
    out = save_result(Path("processed_data") / "paperC_direction.npz",
                      {"model": "paperC_model.run_gen (verbatim), homogeneous, gamma_regen 1, noise off",
                       "L_stripe": LS, "dirs": DIRS, "d": D_DIR, "pairs": D_PAIR, "sq_w": SQ_W, "sq_d": SQ_D,
                       "propagates": f"normal extent >= {PROP} cells", "D1": D1, "D2": D2, "D3": D3,
                       "dstar_bracket": {str(k): v for k, v in dstar.items()}},
                      dirs=np.array(DIRS), d=np.array(D_DIR), extent=ext, speed=spd, propagates=prop,
                      pair_d=np.array(D_PAIR), pair_rows=np.array([p[2] for p in P]),
                      pair_ext=np.array([p[0] for p in P]), squares=Q, sq_w=np.array(SQ_W), sq_d=np.array(SQ_D))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
