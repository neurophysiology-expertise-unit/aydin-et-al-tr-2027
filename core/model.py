"""
core/model.py — fixed constants of the FitzHugh-Nagumo excitable-lattice model and the 5-point periodic lattice
Laplacian. The integrators that use them are in paperC_model.py and the round scripts.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# --- fixed model constants ---
A_FHN = 1.0
B_FHN = 0.8
ETA = 8.0
THETA_BASE = 0.5
DT = 0.0034
NOISE_MULT = 3.0          # multiplier on the lattice noise term

# Reference noise amplitude in the Euler-Maruyama scheme (sigma * sqrt(dt)), ~0.0233.
SIGMA_OLD_NOMINAL = 0.4
SIGMA_EM_PREDICTED = SIGMA_OLD_NOMINAL * np.sqrt(DT)   # ~0.0233

# --- baseline excitability, I0 at A=0 ---
I0_BASE = 0.42


@njit(fastmath=True, cache=True)
def laplacian(Z):
    nx, ny = Z.shape
    L = np.empty_like(Z)
    for i in range(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            L[i, j] = Z[ip, j] + Z[im, j] + Z[i, jp] + Z[i, jm] - 4.0 * Z[i, j]
    return L
