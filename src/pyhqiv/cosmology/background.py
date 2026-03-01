"""
HQIV Cosmology background: lattice evolution, Ω_k, and lapse.

Single-entry API for evolve_to_cmb(T0_K), omega_k_true, get_delta_E_grid.
Curved-sky: comoving_distance(z, omega_k), curved_line_of_sight(theta, phi, omega_k, k).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from pyhqiv.constants import T_CMB_K
from pyhqiv.lattice import DiscreteNullLattice

# Fiducial flat-ΛCDM for E(z) when computing curved χ(z). Only unit conversions.
H0_KM_S_MPC = 67.36
OMEGA_M0 = 0.315
OMEGA_L0 = 0.685
C_H0_MPC = 299792.458 / H0_KM_S_MPC  # c/H0 in Mpc


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
        return self._lattice.evolve_to_cmb(T0_K=T0_K, E_0_factor=E_0_factor, use_jax=use_jax)

    def omega_k_true(self, E_0_factor: float = 1.0, use_jax: bool = False) -> float:
        """True curvature from shell integral (paper ≈ +0.0098)."""
        return self._lattice.omega_k_true(E_0_factor=E_0_factor, use_jax=use_jax)

    @property
    def Ok0(self) -> float:
        """Ω_k^true from lattice (paper ≈ +0.0098). Alias for observable pipeline."""
        return self.omega_k_true()

    def get_delta_E_grid(self, E_0_factor: float = 1.0) -> np.ndarray:
        """Curvature imprint δE(m) for m = 0, ..., m_trans-1."""
        return self._lattice.get_delta_E_grid(E_0_factor=E_0_factor)

    def lapse_factor(self, z: float) -> float:
        """Lapse factor f(z) from lattice (1 today, from evolve_to_cmb)."""
        result = self.evolve_to_cmb()
        # f = 1/lapse_compression at z=0; approximate f(z) ∝ (1+z)^0.1 for structure
        lapse_comp = result["lapse_compression"]
        f0 = 1.0 / max(lapse_comp, 1.0)
        return f0 * ((1.0 + z) ** 0.1) / (1.0 ** 0.1) if z <= 1100 else f0 * 0.5

    @property
    def lapse_now(self) -> float:
        """Lapse factor today (1 / lapse_compression from paper)."""
        result = self.evolve_to_cmb()
        return 1.0 / max(result["lapse_compression"], 1.0)

    def line_of_sight(
        self,
        latitude: float,
        longitude: float,
        n_k: int = 400,
    ) -> np.ndarray:
        """
        Line-of-sight weight for CMB projection (axiom-pure stub).

        Returns weight array (len n_k) for LOS integration. Full implementation
        integrates transfer along LOS with lapse(z). Stub: isotropic (ones).
        """
        _ = latitude
        _ = longitude
        return np.ones(n_k, dtype=float)

    def comoving_distance(
        self,
        z: float,
        omega_k: Optional[float] = None,
        n_z: int = 200,
    ) -> float:
        """
        Comoving distance χ(z) in Mpc, respecting curvature Ω_k.

        χ = R_c sinn(D/R_c): sinn = sin for Ω_k > 0 (closed), sinh for Ω_k < 0 (open).
        D(z) = (c/H0) ∫_0^z dz'/E(z'); R_c = (c/H0)/√|Ω_k|.
        """
        if omega_k is None:
            omega_k = self.Ok0
        omega_m = OMEGA_M0
        omega_l = 1.0 - omega_m - omega_k
        z_arr = np.linspace(0.0, z, max(n_z, 2))
        e_z = np.sqrt(
            omega_m * (1.0 + z_arr) ** 3
            + omega_k * (1.0 + z_arr) ** 2
            + np.maximum(omega_l, 0.0)
        )
        integrand = 1.0 / np.maximum(e_z, 1e-20)
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if _trapz is None:
            from scipy.integrate import trapezoid as _trapz
        d_flat = C_H0_MPC * _trapz(integrand, z_arr)
        if abs(omega_k) < 1e-10:
            return float(d_flat)
        r_c = C_H0_MPC / np.sqrt(abs(omega_k))
        eta = d_flat / r_c
        if omega_k > 0:
            return float(r_c * np.sin(eta))
        return float(r_c * np.sinh(eta))

    def curved_line_of_sight(
        self,
        theta: float,
        phi: float,
        omega_k: Optional[float] = None,
        k: Optional[np.ndarray] = None,
        z_rec: float = 1090.0,
    ) -> np.ndarray:
        """
        LOS weight for curved sky (Ω_k) so peak positions shift correctly.

        Uses χ(z_rec) from comoving_distance(z_rec, omega_k) and spherical
        Bessel j0(k χ) for isotropic weight; direction (theta, phi) reserved
        for future anisotropic ISW/Doppler.
        """
        if omega_k is None:
            omega_k = self.Ok0
        chi_rec = self.comoving_distance(z_rec, omega_k=omega_k)
        if k is None:
            k = np.logspace(-5, 0, 800)
        k = np.asarray(k, dtype=float)
        k = np.maximum(k, 1e-20)
        x = k * chi_rec
        try:
            from scipy.special import spherical_jn

            los = spherical_jn(0, x).astype(float)
        except ImportError:
            los = np.sinc(x / np.pi)
        los = np.clip(los, -0.5, 1.0)
        _ = theta
        _ = phi
        return los
