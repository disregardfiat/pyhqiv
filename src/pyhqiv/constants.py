"""
HQIV physical constants: paper-derived only + unit conversion tables.

Allowed: (1) Values from the paper (Ettinger, Zenodo 10.5281/zenodo.18794889).
         (2) Unit conversion factors for the module to be widely useful.
No other physics constants. Fiducial cosmology (Ω_m, Ω_Λ) only for optional
comoving_distance geometry; documented below.
"""

import math
from typing import Final

# ============== PAPER-DERIVED ==============

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

# --- Paper fiducial outputs ---
OMEGA_TRUE_K_PAPER: Final[float] = 0.0098  # true curvature from shell integral
LAPSE_COMPRESSION_PAPER: Final[float] = 3.96  # 51.2 Gyr → 13.8 Gyr apparent
AGE_WALL_GYR_PAPER: Final[float] = 51.2  # wall-clock age at T_CMB
AGE_APPARENT_GYR_PAPER: Final[float] = 13.8  # apparent age (local chronometers)

# H0 is the radial gradient — the same as time (paper single-source).
# Derived from paper apparent age, not an independent constant: H0 ~ 1/(age).
SEC_PER_GYR: Final[float] = 3.1536e16  # s/Gyr (unit)
MPC_M: Final[float] = 3.086e22  # m/Mpc (unit)
H0_KM_S_MPC_PAPER: Final[float] = (
    1e-3 * MPC_M / (AGE_APPARENT_GYR_PAPER * SEC_PER_GYR)
)  # radial gradient from paper age

# ============== UNIT CONVERSIONS ONLY ==============

# --- SI (for fields / FDTD) ---
C_SI: Final[float] = 2.99792458e8  # m/s
E_PL_SI: Final[float] = 1.956e9  # Planck energy in J
HBAR_SI: Final[float] = 1.054571817e-34  # J·s
K_B_SI: Final[float] = 1.380649e-23  # J/K

# --- Temperature: K ↔ GeV ---
K_B_GEV_PER_K: Final[float] = 8.617333e-14  # GeV/K
GEV_TO_K: Final[float] = 1.160452e13  # K/GeV (T_K = T_GeV * GEV_TO_K)
T_PL_K: Final[float] = T_PL_GEV * GEV_TO_K  # Planck T in K (unit conversion)

# --- CMB monopole in μK (unit conversion from T_CMB_K) ---
T_CMB_MUK: Final[float] = T_CMB_K * 1e6  # μK

# --- Recombination redshift (standard cosmology reference; used for z_rec in transfer) ---
Z_RECOMB: Final[float] = 1090.0

# --- Molecular / protein (Å units) ---
HBAR_C_EV_ANG: Final[float] = 1973.27  # eV·Å
A_LOC_ANG: Final[float] = 1.0

# ============== FIDUCIAL (not from paper) ==============
# Used only for optional geometry / phenomenological sigma8. Not HQIV-derived.
OMEGA_M0_FIDUCIAL: Final[float] = 0.315
OMEGA_L0_FIDUCIAL: Final[float] = 0.685
K_PIVOT_FIDUCIAL: Final[float] = 0.05  # 1/Mpc (primordial P(k) pivot)
