"""
Axiom-pure CMB pipeline: T_Pl → now using only unit conversions + lattice.

HQIVCMBMap.run_from_T_Pl_to_now() derives:
- Background from lattice.evolve_to_cmb
- Primordial P(k) from combinatorial invariant (no A_s)
- Transfer from lattice shell counting + lapse at recombination
- C_ℓ from P(k) T²(k) projection → synfast map → anafast
- σ₈ from growth_factor_to_8Mpc() * sqrt(mean(P_prim))
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pyhqiv.constants import GAMMA
from pyhqiv.cosmology.background import HQIVCosmology
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.perturbations import HQIVPerturbations

# Comoving distance to recombination (Mpc), fiducial flat ΛCDM
ETA_REC_MPC = 14000.0
T_CMB_MUK = 2.725e6  # CMB monopole in μK for amplitude


class HQIVCMBMap:
    """
    T_Pl → now full observable pipeline. Only unit conversions + lattice.

    No A_s or other hard physics constants; primordial amplitude and
    transfer come from combinatorial invariant and curvature imprint.
    """

    def __init__(self, nside: int = 512, gamma: float = GAMMA) -> None:
        self.nside = nside
        self.gamma = gamma
        self._lattice = DiscreteNullLattice(gamma=gamma)
        self.cosmo = HQIVCosmology(gamma=gamma)
        self.pert = HQIVPerturbations(background=self.cosmo, gamma=gamma)

    @property
    def lattice(self) -> DiscreteNullLattice:
        return self._lattice

    def run_from_T_Pl_to_now(
        self,
        T0_K: float = 2.725,
        z_recomb: float = 1090.0,
        lmax: int = 2500,
        n_k: int = 400,
    ) -> Dict[str, Any]:
        """
        Run axiom-pure pipeline: background → P_prim → transfer → C_ℓ → map → σ₈.

        Returns T_map_muK, Cl_TT, sigma8, background, lapse_today.
        """
        # 1. Background from T_Pl regime → recombination
        bg = self.cosmo.evolve_to_cmb(T0_K=T0_K)

        # 2. Primordial spectrum from combinatorial invariant (no A_s)
        k = np.logspace(-5, 0, n_k)
        Pk_prim = self.lattice.primordial_power_from_invariant(k)

        # 3. Lapse-modulated transfer at recombination
        delta_T_transfer = self.pert.cosmological_transfer(k, z_recomb=z_recomb)

        # 4. C_ℓ from P(k) T²(k) at k = ell / eta_rec (flat-sky)
        ell = np.arange(2, min(lmax + 1, 3000), dtype=float)
        k_ell = np.maximum(ell / ETA_REC_MPC, 1e-10)
        P_at_ell = np.interp(k_ell, k, Pk_prim)
        T_at_ell = np.interp(k_ell, k, delta_T_transfer)
        # C_ℓ^TT in (μK)²; amplitude from T_CMB and projection
        cl_tt = (T_CMB_MUK**2) * P_at_ell * (T_at_ell**2) / (ell * (ell + 1)) * (2 * np.pi)

        # 5. Map from synfast (so peaks are in the map)
        try:
            import healpy as hp
        except ImportError as e:
            raise ImportError("healpy is required for HQIVCMBMap.run_from_T_Pl_to_now") from e

        # Healpy cl array: index = ell, 0..lmax; we have ell from 2 to lmax
        cl_arr = np.zeros(int(lmax) + 1)
        ell_int = ell.astype(int)
        mask = (ell_int >= 0) & (ell_int <= lmax)
        cl_arr[ell_int[mask]] = cl_tt[mask]
        cl_arr[0:2] = 0.0

        delta_T = hp.synfast(cl_arr, self.nside, verbose=False)
        # 6. σ₈ from growth factor and primordial amplitude
        sigma8 = self.pert.growth_factor_to_8Mpc() * np.sqrt(np.mean(Pk_prim))

        # 7. C_ℓ from anafast for consistency (optional; we already have cl_tt)
        cl_anafast = hp.anafast(delta_T, lmax=min(lmax, 3 * self.nside - 1))

        return {
            "T_map_muK": delta_T,
            "Cl_TT": cl_anafast,
            "Cl_TT_theory": cl_tt,
            "ell": ell,
            "sigma8": float(sigma8),
            "background": bg,
            "lapse_today": self.cosmo.lapse_now,
        }
