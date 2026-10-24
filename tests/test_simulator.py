"""
Quick self-test of the simulator (about 1 minute on one core). Run from the repository root:
    python -s tests/test_simulator.py        (or: pytest -q tests)
Checks (1) the verbatim-integrator chain: run_chain == run_gen, run_pde == run_chain (FD2), run_gx == run_pde,
run_gen2 == run_gen, run_gen3 == run_gen2, each bit for bit on short runs; (2) one physical fact per rule: with the settling rule at DT the
axis front propagates at d = 8 and fails at d = 16 (paperC_design.npz: upper edge 11.15).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from core.model import DT
from paperC_model import run_gen, hom_params, THETA, TAU_REF
from paperC_continuum import run_chain, chain_setup
from paperC_pde import run_pde, scalars, STENCIL
from paperC_design import run_gx, chain
from paperC_tolerance import run_gen2
from paperC_tolerance2 import run_gen3


def test_integrators_bitwise():
    N = 64; sm, _ = chain_setup(N, 5)
    for d in (8.0, 14.0):
        I0, th, De, gd = hom_params((N, 1), d)
        a = run_gen(I0, th, De, gd, THETA, sm.reshape(N, 1), 1.0, TAU_REF, 3000, 5)[1].ravel()
        b = run_chain(I0.ravel(), th.ravel(), De.ravel(), gd.ravel(), THETA, sm, 1.0, TAU_REF, 3000,
                      np.array([1, -1], np.int64), 2, DT)
        assert np.array_equal(a, b)
        s = scalars(d)
        for rc in (False, True):
            c = run_pde(*s, THETA, sm, 1.0, TAU_REF, 3000, STENCIL["FD2"], 1.0, DT, rc)
            e = run_gx(*s, THETA, sm, 1.0, TAU_REF, 3000, STENCIL["FD2"], 1.0, DT, rc, 0.0, 0)
            assert np.array_equal(c, e)
    I0, th, De, gd = hom_params((16, 16), 8.0)
    sq = np.zeros((16, 16), np.bool_); sq[6:10, 6:10] = True
    assert np.array_equal(run_gen(I0, th, De, gd, THETA, sq, 1.0, TAU_REF, 2000, 5)[1],
                          run_gen2(I0, th, De, gd, THETA, sq, 1.0, TAU_REF, 2000, DT, False, 0.0, 1))
    for sig in (0.0, 0.03):     # round 7: run_gen3 without early stop == run_gen2 (settling rule, noise)
        assert np.array_equal(run_gen2(I0, th, De, gd, THETA, sq, 1.0, TAU_REF, 2000, DT, True, sig, 3),
                              run_gen3(I0, th, De, gd, THETA, sq, 1.0, TAU_REF, 2000, DT, True, sig, 3, sq, 10 ** 9))


def test_operating_window():
    assert chain(8.0, dtf=1)[1] is True          # inside the window
    assert chain(16.0, dtf=1)[1] is False        # above the upper edge


if __name__ == "__main__":
    test_integrators_bitwise(); test_operating_window(); print("simulator self-test passed")
