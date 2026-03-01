"""
Axiom-pure CMB pipeline: T_Pl → now using only unit conversions + lattice.

Respects Ω_k^true = +0.0098 in observables: curved χ(z), curved_line_of_sight,
curvature-aware transfer, ISW from peculiar velocity, growth_to_sigma8.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pyhqiv.constants import GAMMA
from pyhqiv.cosmology.background import HQIVCosmology
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.perturbations import HQIVPerturbations

T_CMB_MUK = 2.725e6  # CMB monopole in μK (δT/T → μK)


class HQIVCMBMap:
    """
    T_Pl → now full pipeline that respects Ω_k^true = +0.0098 in observables.

    Curved-sky projection (curved_line_of_sight), curvature in transfer and σ₈,
    ISW from accelerated galaxy motion. Only unit conversions + lattice.
    """

    def __init__(
        self,
        nside: int = 512,
        gamma: float = GAMMA,
        lmax: int = 2500,
    ) -> None:
        self.nside = nside
        self.gamma = gamma
        self.lmax = lmax
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
        n_k: int = 600,
        use_curved_pixel_loop: bool = False,
    ) -> Dict[str, Any]:
        """
        Run axiom-pure pipeline with Ω_k in observables.

        If use_curved_pixel_loop True: map from pixel loop (curved LOS + ISW per pixel).
        Else: C_ℓ from curved χ(z_rec), synfast → map, then anafast (faster).
        Returns T_map_muK, Cl_TT, sigma8, Omega_k_true, background.
        """
        import healpy as hp

        omega_k = self.cosmo.Ok0
        bg = self.cosmo.evolve_to_cmb(T0_K=T0_K)

        k = np.logspace(-5, 0, n_k)
        Pk_prim = self.lattice.primordial_power_from_invariant(k)
        delta_T_transfer = self.pert.cosmological_transfer(
            k, z_recomb=z_recomb, omega_k=omega_k
        )

        chi_rec = self.cosmo.comoving_distance(z_recomb, omega_k=omega_k)
        ell = np.arange(2, min(self.lmax + 1, 3000), dtype=float)
        k_ell = np.maximum(ell / max(chi_rec, 1.0), 1e-10)
        P_at_ell = np.interp(k_ell, k, Pk_prim)
        T_at_ell = np.interp(k_ell, k, delta_T_transfer)
        cl_tt = (T_CMB_MUK**2) * P_at_ell * (T_at_ell**2) / (ell * (ell + 1)) * (2 * np.pi)

        lapse0 = self.cosmo.lapse_factor(0.0)

        if use_curved_pixel_loop:
            npix = hp.nside2npix(self.nside)
            delta_T = np.zeros(npix)
            for ipix in range(npix):
                theta, phi = hp.pix2ang(self.nside, ipix)
                los = self.cosmo.curved_line_of_sight(
                    theta, phi, omega_k=omega_k, k=k, z_rec=z_recomb
                )
                isw = self.pert.isw_from_peculiar_velocity(theta, phi, omega_k=omega_k)
                delta_T[ipix] = (
                    np.sum(delta_T_transfer * los * lapse0) + isw
                )
            T_map_muK = delta_T * T_CMB_MUK
        else:
            cl_arr = np.zeros(int(self.lmax) + 1)
            ell_int = ell.astype(int)
            mask = (ell_int >= 0) & (ell_int <= self.lmax)
            cl_arr[ell_int[mask]] = cl_tt[mask]
            cl_arr[0:2] = 0.0
            delta_T = hp.synfast(cl_arr, self.nside, verbose=False)
            isw_dipole = self.pert.isw_from_peculiar_velocity(0.0, 0.0, omega_k=omega_k)
            T_map_muK = delta_T + isw_dipole * T_CMB_MUK

        sigma8 = self.pert.growth_to_sigma8(omega_k=omega_k) * np.sqrt(
            np.mean(Pk_prim)
        )
        cl_anafast = hp.anafast(T_map_muK, lmax=min(self.lmax, 3 * self.nside - 1))

        return {
            "T_map_muK": T_map_muK,
            "Cl_TT": cl_anafast,
            "Cl_TT_theory": cl_tt,
            "ell": ell,
            "sigma8": float(sigma8),
            "Omega_k_true": omega_k,
            "background": bg,
            "lapse_today": self.cosmo.lapse_now,
        }

    def plot_multipole(
        self,
        result: Dict[str, Any],
        show: bool = True,
        out_path: Optional[str] = None,
    ) -> None:
        """Plot ℓ(ℓ+1)C_ℓ/2π vs ℓ with Ω_k label."""
        import matplotlib.pyplot as plt

        cl = result["Cl_TT"]
        ell_plot = np.arange(len(cl), dtype=float)
        d_ell = ell_plot * (ell_plot + 1) * cl / (2.0 * np.pi)
        ok = result.get("Omega_k_true", getattr(self.cosmo, "Ok0", 0.0098))

        plt.figure(figsize=(11, 7))
        plt.loglog(
            ell_plot,
            np.maximum(d_ell, 1e-10),
            label=f"HQIV (Ω_k = {ok:.4f})",
        )
        plt.xlabel("Multipole ℓ")
        plt.ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$ [μK²]")
        plt.title("HQIV CMB Multipole — Respecting Ω_k^true = +0.0098")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        if out_path:
            plt.savefig(out_path, dpi=120)
        if show:
            plt.show()
        else:
            plt.close()
