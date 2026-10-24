"""
paperC_geometry.py — FEASIBILITY follow-up for paper C.

paperC_mechanism.npz showed: the HOMOGENEOUS 2-D lattice already stops at full regeneration for d_mult >= 4
(10.6 cells at x4, 5.0 at x5), while the 1-D chain with the same cells propagates at every d_mult up to 8.
So the self-limiting front is not a disorder effect, and it is a 2-D effect. Two 2-D explanations remain:
  (curvature)    a convex front from a small seed has fewer simultaneously-firing lateral partners, so each
                 fresh cell drains into more resting neighbours; a larger seed (flatter front) or a planar
                 stripe should then propagate where the 5x5 seed fails (critical-nucleus picture).
  (coordination) the drain is set by the 2-D coordination number regardless of curvature; then even a
                 planar stripe fails at the same d_mult.
Test: homogeneous 2-D lattice (paperC_mechanism.hom_params), gamma_regen = 1, deterministic, L = 64, 120 s,
d_mult in {3.5, 4, 4.5, 5, 6, 8}, seeds: square patch half-width 2 (reference), 4, 8, and a planar stripe
(5 columns wide, full height). Extent measured as distance from the seed's own edge (max over activated
cells of the distance to the nearest seed cell), in cells.

PRE-REGISTERED PREDICTION (2026-10-11, before running): curvature. At d_mult = 4 and 4.5 the planar stripe
and the half-width-8 patch propagate (front advances >= 15 cells beyond the seed), while half-width 2 stops
(reproducing 10.6 / 6.4 cells from the centre). If the stripe also stops at d_mult = 4, the coordination
reading holds and curvature is rejected. Reported as found; the criterion is not revised afterwards.
Output: processed_data/paperC_geometry.npz.  Run: python -s paperC_geometry.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from core.provenance import save_result
# run_gen and hom_params are identical (verbatim) in paperC_model and in the module this script was run with,
# paperC_mechanism (not released); the constants below have the same values there.
from paperC_model import run_gen, hom_params, THETA, TAU_REF, STEPS, STRIDE, L, T_WIN, DT

D_MULTS = [3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
GEOMS = ["patch2", "patch4", "patch8", "stripe"]


def mask(geom):
    m = np.zeros((L, L), np.bool_); c = L // 2
    if geom == "stripe":
        m[:, c - 2:c + 3] = True
    else:
        w = int(geom[5:]); m[c - w:c + w + 1, c - w:c + w + 1] = True
    return m


def run(d, geom):
    sm = mask(geom)
    I0, th, De, gd = hom_params((L, L), d)
    fld = run_gen(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, STEPS, STRIDE)[0]
    # distance to nearest seed cell on the torus: tile 3x3, take the centre block
    big = ~np.tile(sm, (3, 3))
    dist = distance_transform_edt(big)[L:2 * L, L:2 * L]
    cum = np.zeros((L, L), bool); ext = []
    for t in range(fld.shape[0]):
        cum |= fld[t] > 0.5; mm = cum & ~sm; ext.append(dist[mm].max() if mm.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    return final, float(np.argmax(ext >= 0.99 * final) * STRIDE * DT), float(dist.max())


def main():
    R = np.zeros((len(D_MULTS), len(GEOMS), 3))
    for i, d in enumerate(D_MULTS):
        for j, g in enumerate(GEOMS):
            R[i, j] = run(d, g)
            print(f"d {d:3.1f} {g:7s}: beyond seed {R[i,j,0]:5.1f} cells (max possible {R[i,j,2]:4.1f}), "
                  f"t99 {R[i,j,1]:5.1f} s", flush=True)
    out = save_result(Path("processed_data") / "paperC_geometry.npz",
                      {"model": "paperC_mechanism.run_gen, homogeneous", "gamma_regen": 1.0, "L": L,
                       "T_window": T_WIN, "d_mults": D_MULTS, "geoms": GEOMS,
                       "extent": "max distance beyond the seed (cells), torus"},
                      d_mults=np.array(D_MULTS), extent=R[..., 0], t99=R[..., 1], max_possible=R[..., 2])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
