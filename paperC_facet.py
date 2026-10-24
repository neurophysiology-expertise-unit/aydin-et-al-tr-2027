"""
paperC_facet.py — paper C follow-up (pre-registered after paperC_direction.npz):
does the gamma_regen = 1 front on the homogeneous lattice need an axis-aligned FACET of a critical length?

What is known (committed npz): planar fronts propagate up to d*_axis in (14, 16] along a lattice axis and up to
d*_axis/2 in (7, 8] in every other tested direction (exact reduction, paperC_direction D1/D2). Yet no disk seed
(R0 = 1..16, L = 96) propagates for d >= 4 (paperC_critical_radius), while square seeds do: half-width 4 at
d = 4, 8 at d <= 5, 16 at d <= 6 (paperC_direction, squares, L = 96). Post hoc reading (this is what is tested):
a convex front survives at 3.5 < d < 7 only if it carries an axis-aligned facet of at least l_c(d) cells; the
facet propagates as a planar axis front and its ends are eroded. A digital disk's axis tip (its outermost row)
is short: 1 cell for integer R0, at most 5 cells in the R0 grid used, so every disk failed.

MODEL: paperC_model.run_gen (verbatim), hom_params, gamma_regen = 1, noise off, tau_ref 15 s, 120 s.
 RECT:  5-row rectangles of length l (axis facets of l cells top and bottom, 5 cells at the ends), L = 96,
        l in L_RECT, d in D_FAC. PROPAGATES := beyond-seed extent >= 20 cells (as paperC_critical_radius).
        l_c(d) = smallest l from which every longer l propagates.
 DISK:  digital disks {r <= R0} on L = 128 whose outermost row (tip facet) has a chosen length:
        R0 = 16.0 (tip 1), 16.5 (9), 16.8 (11), 24.9 (13), 32.99 (17) cells; d in D_FAC.
        (tip lengths are computed and printed by the script before the runs.)

PRE-REGISTERED PREDICTIONS (written 2026-10-12 after paperC_direction.npz and before any run of this script):
 F1. Facet length, not seed area, sets survival: at each d the rectangle outcome is monotone in l, and
     l_c(4) in (5, 9], l_c(5) in (9, 17], l_c(6) in (17, 33] (the brackets the squares imply).
 F2. Disks obey the same threshold: at every d in {4, 4.5, 5, 6}, a disk propagates if and only if its tip facet
     is >= l_c(d) measured in F1 at the same d (all 4 x 5 = 20 disk outcomes correct).
 F3. For d >= 8 (above the non-axis planar failure) nothing propagates from any rectangle with l <= 41.
     (Expectation, from the squares: w = 16 failed at d = 8.)
If F2 fails, the facet reading is rejected and the disk failure at d >= 4 stays unexplained.
Failed predictions are reported as failed; criteria are not revised after the data land.

Output: processed_data/paperC_facet.npz.   Run: python -s paperC_facet.py   (about 20 min on 11 cores)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from core.provenance import save_result
from paperC_model import run_gen, hom_params, check, THETA, TAU_REF, STEPS, STRIDE, T_WIN

D_FAC = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
L_RECT = [3, 5, 7, 9, 11, 13, 17, 21, 25, 33, 41]
R0_DISK = [16.0, 16.5, 16.8, 24.9, 32.99]
PROP = 20.0


def rgrid(n):
    c = n // 2; ii, jj = np.mgrid[0:n, 0:n]
    di = np.minimum(abs(ii - c), n - abs(ii - c)); dj = np.minimum(abs(jj - c), n - abs(jj - c))
    return np.sqrt(di ** 2 + dj ** 2)


def seed(kind, v):
    if kind == "rect":
        n = 96; c = n // 2; m = np.zeros((n, n), np.bool_); h = v // 2
        m[c - 2:c + 3, c - h:c - h + v] = True
        return m
    return rgrid(128) <= v


def tip(m):
    rows = np.where(m.any(1))[0]
    return int(m[rows.min()].sum())


def job(args):
    kind, v, d = args
    sm = seed(kind, v); n = sm.shape[0]
    I0, th, De, gd = hom_params((n, n), d)
    tr = run_gen(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, STEPS, STRIDE)[1]
    dist = distance_transform_edt(~np.tile(sm, (3, 3)))[n:2 * n, n:2 * n]
    fired = (tr >= 0) & ~sm
    return float(dist[fired].max()) if fired.any() else 0.0


def lc_of(p, ls):
    p = np.asarray(p, bool)
    for k in range(len(ls)):
        if p[k:].all():
            return ls[k], bool(p[k:].all() and not p[:k].any())
    return np.inf, not p.any()


def main():
    check()
    tips = [tip(seed("disk", r)) for r in R0_DISK]
    print(f"disk tip facets: {dict(zip(R0_DISK, tips))}", flush=True)
    jobs = [("rect", l, d) for d in D_FAC for l in L_RECT] + [("disk", r, d) for d in D_FAC for r in R0_DISK]
    with Pool(11) as p:
        res = np.array(p.map(job, jobs, chunksize=1))
    nr = len(D_FAC) * len(L_RECT)
    E = res[:nr].reshape(len(D_FAC), len(L_RECT)); ED = res[nr:].reshape(len(D_FAC), len(R0_DISK))
    P, PD = E >= PROP, ED >= PROP
    LC = {}; MONO = {}
    for a, d in enumerate(D_FAC):
        lc, mono = lc_of(P[a], L_RECT); LC[d] = lc; MONO[d] = mono
        print(f"d {d:3.1f}: rect l " + " ".join(f"{l}{'+' if v else '.'}" for l, v in zip(L_RECT, P[a]))
              + f"  l_c {lc} {'' if mono else 'NON-MONOTONE'} | disks " +
              " ".join(f"R0 {r}(tip {t}){'+' if v else '.'}" for r, t, v in zip(R0_DISK, tips, PD[a])), flush=True)
    br = {4.0: (5, 9), 5.0: (9, 17), 6.0: (17, 33)}
    F1 = all(MONO[d] for d in D_FAC) and all(br[d][0] < LC[d] <= br[d][1] for d in br)
    f2 = [(d, r, t, bool(PD[D_FAC.index(d), k]), bool(t >= LC[d])) for d in (4.0, 4.5, 5.0, 6.0)
          for k, (r, t) in enumerate(zip(R0_DISK, tips))]
    F2 = all(o == pr for *_, o, pr in f2)
    F3 = bool(not P[D_FAC.index(8.0)].any())
    print(f"F1 {F1}  F2 {F2} ({sum(o == pr for *_, o, pr in f2)}/{len(f2)})  F3 {F3}")
    out = save_result(Path("processed_data") / "paperC_facet.npz",
                      {"model": "paperC_model.run_gen (verbatim), homogeneous, gamma_regen 1, noise off",
                       "d": D_FAC, "l_rect": L_RECT, "R0_disk": R0_DISK, "disk_tips": tips,
                       "propagates": f"beyond-seed extent >= {PROP} cells", "F1": F1, "F2": F2, "F3": F3,
                       "l_c": {str(k): v for k, v in LC.items()}},
                      d=np.array(D_FAC), l_rect=np.array(L_RECT), R0_disk=np.array(R0_DISK), tips=np.array(tips),
                      rect_extent=E, disk_extent=ED, rect_prop=P, disk_prop=PD,
                      l_c=np.array([LC[d] for d in D_FAC], float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
