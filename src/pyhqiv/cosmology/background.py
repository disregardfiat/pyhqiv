"""
HQIV Cosmology background: lattice evolution, Ω_k, and lapse.

Single-entry API for evolve_to_cmb(T0_K), omega_k_true, get_delta_E_grid.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from pyhqiv.constants import T_CMB_K
from pyhqiv.lattice import DiscreteNullLattice


class HQIVCosmology:
    """
    HQIV cosmology from the discrete null lattice: true curvature Ω_k,
    wall-clock and apparent ages, and lapse compression.

    Single call to evolve_to_cmb(T0_K) returns paper fiducials:
    Omega_true_k ≈ 0.0098, age_wall_Gyr = 51.2, age_apparent_Gyr = 13.8,
    lapse_compression ≈ 3.96.
    """

    def __init__(
        self,
        m_trans: int = 500,
        gamma: float = 0.40,
        alpha: float = 0.60,
    ) -> None:
        self._lattice = DiscreteNullLattice(m_trans=m_trans, gamma=gamma, alpha=alpha)
        self.m_trans = m_trans
        self.gamma = gamma
        self.alpha = alpha

    @property
    def lattice(self) -> DiscreteNullLattice:
        """Underlying discrete null lattice for shell integrals and δE(m)."""
        return self._lattice

    def evolve_to_cmb(
        self,
        T0_K: float = T_CMB_K,
        E_0_factor: float = 1.0,
        use_jax: bool = False,
    ) -> Dict[str, float]:
        """
        Evolve lattice to CMB hypersurface T = T0_K.

        Returns Omega_true_k, age_wall_Gyr, age_apparent_Gyr, lapse_compression,
        T_CMB_K, m_trans, gamma.
        """
        return self._lattice.evolve_to_cmb(
            T0_K=T0_K, E_0_factor=E_0_factor, use_jax=use_jax
        )

    def omega_k_true(self, E_0_factor: float = 1.0, use_jax: bool = False) -> float:
        """True curvature from shell integral (paper ≈ +0.0098)."""
        return self._lattice.omega_k_true(
            E_0_factor=E_0_factor, use_jax=use_jax
        )

    def get_delta_E_grid(self, E_0_factor: float = 1.0) -> np.ndarray:
        """Curvature imprint δE(m) for m = 0, ..., m_trans-1."""
        return self._lattice.get_delta_E_grid(E_0_factor=E_0_factor)
