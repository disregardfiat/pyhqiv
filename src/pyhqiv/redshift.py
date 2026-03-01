"""
HQIV Redshift: full decomposition into expansion, gravitational, and HQIV mass-lapse.

z_total (apparent) = (1+z_exp)(1+z_grav)(1+z_HQIV) - 1
Wall-clock vs apparent time from lapse_compression (e.g. 3.96).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from pyhqiv.constants import (
    AGE_APPARENT_GYR_PAPER,
    AGE_WALL_GYR_PAPER,
    GAMMA,
    LAPSE_COMPRESSION_PAPER,
    OMEGA_TRUE_K_PAPER,
)
from pyhqiv.fluid import f_inertia
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.phase import HQIVPhaseLift


def z_expansion_from_scale_factor(a_emit: Union[float, np.ndarray], a_0: float = 1.0) -> np.ndarray:
    """(1 + z_exp) = a_0 / a_emit. Standard FLRW."""
    a_emit = np.asarray(a_emit)
    return np.maximum(a_0 / np.maximum(a_emit, 1e-30) - 1.0, 0.0)


def z_gravitational_from_potential(
    delta_Phi_over_c2: Union[float, np.ndarray],
) -> np.ndarray:
    """(1 + z_grav) ≈ 1 + ΔΦ/c². ΔΦ = Φ_obs - Φ_emit (e.g. in J/kg); delta_Phi_over_c2 = ΔΦ/c²."""
    x = np.asarray(delta_Phi_over_c2)
    return np.maximum(1.0 + x, 1e-30) - 1.0  # z_grav ≈ ΔΦ/c² for small ΔΦ


def z_HQIV_mass_lapse_from_lapse_ratio(
    f_emit: Union[float, np.ndarray],
    f_obs: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    (1 + z_HQIV) = f_obs / f_emit from clock rate ratio (lapse).
    f = a_loc/(a_loc + φ/6). If observer is in weak field, f_obs ≈ 1.
    """
    f_emit = np.asarray(f_emit)
    f_obs = np.asarray(f_obs)
    return np.maximum(f_obs / np.maximum(f_emit, 1e-30), 1e-30) - 1.0


def z_total_apparent(
    z_exp: Union[float, np.ndarray],
    z_grav: Union[float, np.ndarray] = 0.0,
    z_HQIV: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
    """(1 + z_app) = (1+z_exp)(1+z_grav)(1+z_HQIV). Returns z_app."""
    a = 1.0 + np.asarray(z_exp)
    b = 1.0 + np.asarray(z_grav)
    c = 1.0 + np.asarray(z_HQIV)
    return a * b * c - 1.0


def wall_clock_age_at_emission_Gyr(
    z_apparent: Union[float, np.ndarray],
    age_wall_Gyr: float = AGE_WALL_GYR_PAPER,
    lapse_compression: float = LAPSE_COMPRESSION_PAPER,
) -> np.ndarray:
    """
    In HQIV, apparent age at emission (from FLRW with z_app) times lapse_compression
    gives wall-clock time at emission (relative to wall-clock today).
    Simplified: t_emit_wall = (age_wall_Gyr) * (1 - 1/sqrt(1+z)) style * lapse factor.
    Here return age_wall_Gyr * (1 - a_emit) as proxy for time since emission in wall-clock
    if a_emit = 1/(1+z_app).
    """
    z = np.asarray(z_apparent)
    a_emit = 1.0 / (1.0 + z)
    # Comoving time fraction: roughly t_emit/t_today ~ a_emit^{3/2} for matter-dominated
    # Simplified: wall-clock age at emission ≈ age_wall_Gyr * (1 - a_emit^1.5)
    time_frac = 1.0 - np.power(a_emit, 1.5)
    return np.maximum(age_wall_Gyr * time_frac, 0.0)


class HQIVRedshift:
    """
    Full redshift decomposition: z_expansion, z_gravitational, z_HQIV_mass_lapse,
    z_total (apparent vs true/wall-clock). Integrates with DiscreteNullLattice
    and HQIVPhaseLift for cosmology.
    """

    def __init__(
        self,
        lapse_compression: float = LAPSE_COMPRESSION_PAPER,
        age_wall_Gyr: float = AGE_WALL_GYR_PAPER,
        gamma: float = GAMMA,
    ) -> None:
        self.lapse_compression = lapse_compression
        self.age_wall_Gyr = age_wall_Gyr
        self.gamma = gamma
        self._lattice: Optional[DiscreteNullLattice] = None
        self._phase: Optional[HQIVPhaseLift] = None

    def with_lattice(self, lattice: DiscreteNullLattice) -> HQIVRedshift:
        """Attach a DiscreteNullLattice for evolve_to_cmb and Ω_k."""
        self._lattice = lattice
        return self

    def with_phase(self, phase: HQIVPhaseLift) -> HQIVRedshift:
        """Attach HQIVPhaseLift for ˙δθ′ and lapse."""
        self._phase = phase
        return self

    def z_expansion(self, a_emit: Union[float, np.ndarray], a_0: float = 1.0) -> np.ndarray:
        """(1+z_exp) = a_0 / a_emit."""
        return z_expansion_from_scale_factor(a_emit, a_0)

    def z_gravitational(self, delta_Phi_over_c2: Union[float, np.ndarray]) -> np.ndarray:
        """z_grav from potential difference ΔΦ/c²."""
        return z_gravitational_from_potential(delta_Phi_over_c2)

    def z_HQIV_from_f(self, f_emit: Union[float, np.ndarray], f_obs: float = 1.0) -> np.ndarray:
        """(1+z_HQIV) = f_obs / f_emit."""
        return z_HQIV_mass_lapse_from_lapse_ratio(f_emit, f_obs)

    def z_HQIV_from_phi(
        self,
        phi_emit: Union[float, np.ndarray],
        phi_obs: Union[float, np.ndarray] = 0.0,
        a_loc_emit: float = 1.0,
        a_loc_obs: float = 1.0,
    ) -> np.ndarray:
        """Compute z_HQIV from φ at emitter and observer using f = a/(a+φ/6)."""
        f_emit = f_inertia(a_loc_emit, phi_emit)
        f_obs = f_inertia(a_loc_obs, phi_obs)
        return z_HQIV_mass_lapse_from_lapse_ratio(f_emit, f_obs)

    def z_total(
        self,
        z_exp: Union[float, np.ndarray],
        z_grav: Union[float, np.ndarray] = 0.0,
        z_HQIV: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        """Apparent redshift (1+z_app) = (1+z_exp)(1+z_grav)(1+z_HQIV)."""
        return z_total_apparent(z_exp, z_grav, z_HQIV)

    def decompose_from_apparent(
        self,
        z_app: Union[float, np.ndarray],
        z_grav: Union[float, np.ndarray] = 0.0,
        z_HQIV_fraction: float = 0.0,
    ) -> dict:
        """
        Given z_app, optionally assume z_grav and a fraction of remainder as z_HQIV.
        z_HQIV_fraction in [0,1]: 0 = pure expansion, 1 = max HQIV component.
        Returns z_exp, z_grav, z_HQIV, z_app (check).
        """
        z_app = np.asarray(z_app)
        one_plus_z = 1.0 + z_app
        one_plus_z_grav = 1.0 + np.asarray(z_grav)
        # (1+z_exp)(1+z_HQIV) = (1+z_app)/(1+z_grav)
        remainder = one_plus_z / np.maximum(one_plus_z_grav, 1e-30)
        # Split: (1+z_exp) = remainder^(1 - z_HQIV_fraction), (1+z_HQIV) = remainder^z_HQIV_fraction
        one_plus_z_exp = np.power(remainder, 1.0 - z_HQIV_fraction)
        one_plus_z_HQIV = np.power(remainder, z_HQIV_fraction)
        z_exp = one_plus_z_exp - 1.0
        z_HQIV = one_plus_z_HQIV - 1.0
        return {
            "z_exp": z_exp,
            "z_grav": np.asarray(z_grav),
            "z_HQIV": z_HQIV,
            "z_app": z_total_apparent(z_exp, z_grav, z_HQIV),
        }

    def wall_clock_age_at_emission(
        self,
        z_apparent: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Wall-clock age at emission (Gyr) for given z_apparent."""
        return wall_clock_age_at_emission_Gyr(
            z_apparent,
            age_wall_Gyr=self.age_wall_Gyr,
            lapse_compression=self.lapse_compression,
        )

    def cosmology_result(self) -> dict:
        """If lattice attached, return evolve_to_cmb result; else paper defaults."""
        if self._lattice is not None:
            return self._lattice.evolve_to_cmb()
        return {
            "Omega_true_k": OMEGA_TRUE_K_PAPER,
            "age_wall_Gyr": self.age_wall_Gyr,
            "age_apparent_Gyr": AGE_APPARENT_GYR_PAPER,
            "lapse_compression": self.lapse_compression,
        }
