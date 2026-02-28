"""
HQIVAtom: position, charge, species, local Θ(x), φ(x)=2c²/Θ_local,
delta_theta_prime(), modified field contribution. Paper: φ = 2c²/Θ_local.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import C_SI
from pyhqiv.phase import delta_theta_prime
from pyhqiv.utils import local_theta_from_distance, phi_from_theta_local


class HQIVAtom:
    """
    Single atom (or source) with position, charge, species; local horizon Θ(x)
    and auxiliary field φ(x) = 2c²/Θ_local; phase δθ′(E′) and modified contribution.
    """

    def __init__(
        self,
        position: Union[Tuple[float, float, float], np.ndarray],
        charge: float = 0.0,
        species: str = "H",
        c_si: float = C_SI,
    ) -> None:
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.charge = float(charge)
        self.species = species
        self.c_si = c_si

    def local_theta(self, x: np.ndarray) -> np.ndarray:
        """
        Local horizon scale Θ(x) at field points x. Shape (..., 3) → (...,).
        Default: radial distance from this atom.
        """
        r = x - self.position
        return local_theta_from_distance(r)

    def phi_local(self, x: np.ndarray) -> np.ndarray:
        """φ(x) = 2c²/Θ_local(x)."""
        theta = self.local_theta(x)
        return phi_from_theta_local(theta, c=self.c_si)

    def delta_theta_prime_at(self, E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """δθ′(E′) = arctan(E′)×(π/2)."""
        return delta_theta_prime(E_prime)

    def modified_field_contribution(
        self,
        x: np.ndarray,
        E_prime: float = 0.5,
        gamma: float = 0.40,
    ) -> np.ndarray:
        """
        Coefficient for γ(φ/c²)(˙δθ′/c) type correction at positions x.
        Returns array same shape as phi_local(x) for use in constitutive relations.
        """
        phi = self.phi_local(x)
        phi_over_c2 = phi / (self.c_si ** 2)
        # Homogeneous ˙δθ′ ≈ H; use E′ to set scale: ˙δθ′/c ∝ arctan(E′)
        dtdc = np.arctan(E_prime) * (np.pi / 2.0) / self.c_si
        return gamma * phi_over_c2 * dtdc
