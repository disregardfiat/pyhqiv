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

# --- Polarization / birefringence (z_rec = exp(beta_rad/KAPPA_BETA) - 1). Paper-derived when set. ---
KAPPA_BETA: Final[float] = 1.0  # placeholder; replace with paper value when available

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
J_TO_MEV: Final[float] = 1.0 / 1.602176634e-13  # MeV per J (CODATA)
E_PL_MEV: Final[float] = E_PL_SI * J_TO_MEV  # Planck energy in MeV (~1.22e22)
HBAR_SI: Final[float] = 1.054571817e-34  # J·s
K_B_SI: Final[float] = 1.380649e-23  # J/K
# Planck length (m); √(ℏG/c³). For 0 < x < θ horizon-distance ratio in lattice.
L_PLANCK_M: Final[float] = 1.616255e-35  # m (CODATA 2018 order)

# --- Temperature: K ↔ GeV ---
K_B_GEV_PER_K: Final[float] = 8.617333e-14  # GeV/K
GEV_TO_K: Final[float] = 1.160452e13  # K/GeV (T_K = T_GeV * GEV_TO_K)
T_PL_K: Final[float] = T_PL_GEV * GEV_TO_K  # Planck T in K (unit conversion)

# --- CMB monopole in μK (unit conversion from T_CMB_K) ---
T_CMB_MUK: Final[float] = T_CMB_K * 1e6  # μK

# --- Dipole: solar system only (galactic screens at ~3500 ly) ---
C_KM_S: Final[float] = 2.99792458e5  # km/s (for dipole δT/T = v/c)
V_EARTH_ORBIT_KM_S: Final[float] = 29.78  # Earth orbital speed; dipole δT_μK ≈ (v/c) T_CMB_MUK

# --- Recombination redshift (standard cosmology reference; used for z_rec in transfer) ---
Z_RECOMB: Final[float] = 1090.0

# --- Comoving sound horizon at recombination [Mpc] ---
# Paper: acoustic scale set by horizon at recombination; CLASS-HQIV thermo gives rs_rec ~ 218 Mpc.
# Used so transfer argument k·r_s is dimensionless when k is in 1/Mpc (peak position ℓ_A ≈ π χ_rec / r_s).
R_S_REC_MPC: Final[float] = 218.0

# --- Silk damping scale at recombination [Mpc] ---
# Without this, damping_scale from lattice can be 0 (constant m) and peaks look like |sin|.
# Standard CMB: Silk scale ~ 5–50 Mpc; use fraction of r_s so peaks broaden and decay at high ℓ.
SILK_DAMPING_MPC: Final[float] = 50.0

# --- Molecular / protein (Å units) ---
HBAR_C_EV_ANG: Final[float] = 1973.27  # eV·Å
A_LOC_ANG: Final[float] = 1.0

# --- Avogadro (for environment-derived Θ_ref: mean interparticle spacing) ---
N_A: Final[float] = 6.02214076e23  # 1/mol (CODATA)

# --- Standard Model / EM / QCD (reference values for tests and scalings) ---
# Fine structure constant α_EM = e²/(4πε₀ℏc) ≈ 1/137.036 (CODATA); used in Coulomb mode reduction.
ALPHA_EM_INV: Final[float] = 137.036  # 1/α_EM (dimensionless)
# Weak mixing angle at M_Z: sin²θ_W ≈ 0.23122 (PDG)
SIN2_THETA_W_MZ: Final[float] = 0.23122
# Light quark masses (MeV/c², PDG central values; in full HQIV these are derived from the mass equation at now).
M_U_MEV_QCD: Final[float] = 2.2
M_D_MEV_QCD: Final[float] = 4.7

# --- Nuclear / decay: first principles only; no preset B or Θ in engine ---
# ħc and nucleon masses from CODATA (unit conversion / standard particle data only).
HBAR_C_MEV_FM: Final[float] = 197.3  # ħc in MeV·fm (ΔE_info = ħc(1/Θ_u - 1/Θ_s))
M_PROTON_MEV: Final[float] = 938.272  # proton rest mass (MeV/c²)
M_NEUTRON_MEV: Final[float] = 939.565  # neutron rest mass (MeV/c²)
T_PLANCK_S: Final[float] = 5.39e-44  # s (Planck time)
# Observer-centric lattice tick: wall-clock tick scaled by lapse (51.2/13.8 Gyr → ~3.96)
TAU_TICK_OBSERVER_S: Final[float] = T_PLANCK_S * (
    AGE_WALL_GYR_PAPER / AGE_APPARENT_GYR_PAPER
)  # ≈ 2.0e-43 s effective
# Macroscopic decay scaling from forward_4d_evolution (Spin(8) triality / relational volume)
MACROSCOPIC_DECAY_SCALE: Final[float] = LAPSE_COMPRESSION_PAPER * 1.0e54  # λ_macro = λ_raw / this
# φ_crit: universal threshold from CMB T0 "now" hypersurface (m²/s²); damping φ/(φ+φ_crit)
PHI_CRIT_SI: Final[float] = 1.0e30  # paper-derived order; nuclear φ ~ 2.8e30 so ratio ~1

# ============== FIDUCIAL (not from paper) ==============
# Used only for optional geometry / phenomenological sigma8. Not HQIV-derived.
OMEGA_M0_FIDUCIAL: Final[float] = 0.315
OMEGA_L0_FIDUCIAL: Final[float] = 0.685
K_PIVOT_FIDUCIAL: Final[float] = 0.05  # 1/Mpc (primordial P(k) pivot)
