"""
HQIV Solar Core: φ(r) profile, lapse, and fusion corrections for the Sun.

Core axiom: E_tot = m c² + ħ c/Δx with Δx ≤ Θ_local(x) → φ(x) = 2 c²/Θ_local(x)
→ local lapse f(a_loc, φ) = a_loc / (a_loc + φ/6).

Uses ρ(r) from a simple standard solar model (polytrope-like or tabulated)
to build Θ_local(r) and φ(r) from center to surface.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import C_SI, GAMMA, LAPSE_COMPRESSION_PAPER
from pyhqiv.fluid import f_inertia
from pyhqiv.utils import phi_from_theta_local

# --- Standard solar model fiducials (SI where noted) ---
R_SUN_M: float = 6.9634e8  # m
M_SUN_KG: float = 1.9884e30  # kg
G_SI: float = 6.67430e-11  # m³/(kg·s²)
R_SUN_CM: float = 6.9634e10  # cm
RHO_CORE_CGS: float = 150.0  # g/cm³ central density
T_CORE_MK: float = 15.7  # MK central temperature
L_SUN_W: float = 3.828e26  # W
AGE_SOLAR_APPARENT_GYR: float = 4.57  # apparent age (local chronometers)


def schwarzschild_radius_sun_m() -> float:
    """Schwarzschild radius of the Sun, in metres."""
    return 2.0 * G_SI * M_SUN_KG / (C_SI**2)


def theta_local_solar(
    r: Union[float, np.ndarray],
    R_sun: float = R_SUN_M,
    r_core_frac: float = 0.2,
    use_enclosed_mass: bool = False,
    M_enclosed_callback: Optional[callable] = None,
) -> np.ndarray:
    """
    Local horizon scale Θ(r) in the Sun. Units: same as R_sun (m).

    Simple model: Θ(r) = r + r_core so that at center Θ = r_core = r_core_frac * R_sun.
    Optionally: Θ ∝ (G M(r)/c²)^{1/2} if use_enclosed_mass and M_enclosed_callback given.
    """
    r = np.asarray(r, dtype=float)
    r_core = r_core_frac * R_sun
    if use_enclosed_mass and M_enclosed_callback is not None:
        M_r = np.asarray(M_enclosed_callback(r))
        # Θ ~ characteristic length from gravity: (G M/c²) or (G M/c² * r)^{1/2}
        rs_local = 2.0 * G_SI * np.maximum(M_r, 1e-30) / (C_SI**2)
        theta = np.maximum(np.sqrt(rs_local * np.abs(r) + 1e-30), r_core * 0.01)
        return theta
    # Default: linear + core floor
    theta = np.maximum(np.abs(r) + r_core, r_core * 0.1)
    return theta


def rho_solar_polytrope(
    r: Union[float, np.ndarray],
    R_sun: float = R_SUN_M,
    rho_c: float = RHO_CORE_CGS * 1e3,  # kg/m³
    n_polytrope: float = 3.0,
) -> np.ndarray:
    """
    Simple polytropic density profile ρ(r) for the Sun. ρ_c in kg/m³.
    ρ/ρ_c ≈ (1 - (r/R)^2)^n for rough core; here use exp(- (r/r_core)^2) for simplicity.
    """
    r = np.asarray(r, dtype=float)
    r_core = 0.2 * R_sun
    x = np.minimum(np.abs(r) / np.maximum(r_core, 1e-30), 20.0)
    # Gaussian-like core, taper to surface
    r_frac = np.abs(r) / np.maximum(R_sun, 1e-30)
    rho = rho_c * np.exp(-(x**2)) * np.maximum(1.0 - r_frac**2, 0.0) ** 0.5
    return np.maximum(rho, 1e-30)


class HQIVSolarCore:
    """
    Solar (or stellar) core with HQIV φ(r), lapse f(r), and fusion/luminosity corrections.

    φ(r) = 2 c² / Θ_local(r); f(r) = a_loc / (a_loc + φ(r)/6).
    Standard vs HQIV quantities: T_c, ρ_c, luminosity shift, age implication.
    """

    def __init__(
        self,
        R_star: float = R_SUN_M,
        M_star: float = M_SUN_KG,
        r_core_frac: float = 0.2,
        rho_c_cgs: float = RHO_CORE_CGS,
        T_c_MK: float = T_CORE_MK,
        gamma: float = GAMMA,
        c_si: float = C_SI,
    ) -> None:
        self.R_star = R_star
        self.M_star = M_star
        self.r_core_frac = r_core_frac
        self.r_core = r_core_frac * R_star
        self.rho_c_cgs = rho_c_cgs
        self.rho_c_si = rho_c_cgs * 1e3  # kg/m³
        self.T_c_MK = T_c_MK
        self.T_c_K = T_c_MK * 1e6
        self.gamma = gamma
        self.c_si = c_si

    def theta_local(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """Θ_local(r) in m. Uses r + r_core model."""
        r = np.asarray(r, dtype=float)
        return np.maximum(np.abs(r) + self.r_core, self.r_core * 0.01)

    def phi(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """φ(r) = 2 c² / Θ_local(r). Units: m²/s² (acceleration scale × length)."""
        theta = self.theta_local(r)
        return phi_from_theta_local(theta, c=self.c_si)

    def lapse_factor(
        self,
        r: Union[float, np.ndarray],
        a_loc: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        f(a_loc, φ) = a_loc / (a_loc + φ/6). If a_loc is None, use surface gravity
        g_surf = G M / R^2 as reference.
        """
        phi = self.phi(r)
        if a_loc is None:
            a_loc = float(G_SI * self.M_star / (self.R_star**2))
        return f_inertia(np.asarray(a_loc), phi)

    def fusion_rate_correction(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Correction to fusion rate from lapse (clock rate): ε_pp ∝ T^n * f.
        Returns f(r) so that rate_HQIV = rate_standard * f(r).
        """
        return self.lapse_factor(r)

    def luminosity_shift_apparent_to_true(
        self, lapse_compression: float = LAPSE_COMPRESSION_PAPER
    ) -> float:
        """
        Apparent luminosity L_app vs true (wall-clock) L_true: L_app = L_true / lapse_compression
        for bolometric power per unit apparent time. Returns lapse_compression (e.g. 3.96).
        """
        return lapse_compression

    def standard_vs_hqiv_table(
        self,
        lapse_compression: float = LAPSE_COMPRESSION_PAPER,
    ) -> dict:
        """
        Numerical comparison: standard SSM vs HQIV-corrected.
        Returns dict with T_c, ρ_c, L_shift, age_apparent, age_wall_clock.
        """
        r_c = 0.0  # center
        phi_c = float(self.phi(r_c))
        f_c = float(self.lapse_factor(r_c))
        return {
            "T_c_MK_standard": self.T_c_MK,
            "T_c_MK_HQIV_effective": self.T_c_MK * np.sqrt(f_c),  # rough: T_eff ∝ √f for same flux
            "rho_c_g_cm3": self.rho_c_cgs,
            "phi_core_m2_s2": phi_c,
            "f_core": f_c,
            "lapse_compression_global": lapse_compression,
            "age_apparent_Gyr": AGE_SOLAR_APPARENT_GYR,
            "age_wall_clock_Gyr": AGE_SOLAR_APPARENT_GYR * lapse_compression,
            "L_shift_factor": lapse_compression,
        }


def phi_solar_radial_profile(
    n_radii: int = 128,
    R_sun: float = R_SUN_M,
    r_core_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Radial profiles r (m), φ(r) (m²/s²), and f(r) for the Sun.
    Returns (r, phi, f) for plotting.
    """
    r = np.linspace(0.0, R_sun, n_radii)
    sun = HQIVSolarCore(R_star=R_sun, r_core_frac=r_core_frac)
    phi = sun.phi(r)
    f = sun.lapse_factor(r)
    return r, phi, f
