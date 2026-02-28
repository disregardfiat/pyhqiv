"""
HQIV physical constants from the paper (Ettinger, February 22, 2026).

All values are defined in the paper or derived from first principles;
no external data files.
"""

import math
from typing import Final

# --- Entanglement monogamy & effective G exponent ---
GAMMA: Final[float] = 0.40  # thermodynamic coefficient (entanglement monogamy)
ALPHA: Final[float] = 0.60  # G_eff exponent: G_eff/G_0 = [H(a)/H_0]^α

# --- Temperature scales [GeV] ---
T_PL_GEV: Final[float] = 1.2209e19  # Planck temperature
T_LOCK_GEV: Final[float] = 1.8  # QCD lock-in (baryogenesis window centre)

# --- CMB today [K] ---
T_CMB_K: Final[float] = 2.725

# --- Discrete null lattice ---
M_TRANS: Final[int] = 500  # discrete-to-continuous transition shell index

# --- Combinatorial invariant: 6^7 √3 (stars-and-bars + Fano-plane) ---
COMBINATORIAL_INVARIANT: Final[float] = (6**7) * math.sqrt(3)  # ≈ 4.849e5

# --- Paper fiducial outputs (for tests) ---
OMEGA_TRUE_K_PAPER: Final[float] = 0.0098  # true curvature from shell integral
LAPSE_COMPRESSION_PAPER: Final[float] = 3.96  # 51.2 Gyr → 13.8 Gyr apparent
AGE_WALL_GYR_PAPER: Final[float] = 51.2  # wall-clock age at T_CMB
AGE_APPARENT_GYR_PAPER: Final[float] = 13.8  # apparent age (local chronometers)

# --- SI (for fields / FDTD when needed) ---
C_SI: Final[float] = 2.99792458e8  # m/s
E_PL_SI: Final[float] = 1.956e9  # Planck energy in J (for δθ′ normalization if used)
HBAR_SI: Final[float] = 1.054571817e-34  # J·s
K_B_SI: Final[float] = 1.380649e-23  # J/K

# --- Gyr conversion (H in 1/s, age in s → Gyr) ---
SEC_PER_GYR: Final[float] = 3.1536e16
