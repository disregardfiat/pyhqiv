"""
HQIV CMB: full T_Pl → now pipeline in one place.

- HQIVCMBMap: axiom-pure, respects Ω_k^true = +0.0098 (curved LOS, ISW, growth_to_sigma8).
- run_hqiv_cmb_to_map: thin wrapper over cosmology_full (phenomenological).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.perturbations import HQIVPerturbations
from pyhqiv.cosmology.background import HQIVCosmology

# CMB monopole in μK (δT/T → μK)
T_CMB_MUK = 2.725e6

try:
    import healpy as hp
except ImportError:
    hp = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class HQIVCMBMap:
    """Full T_Pl → now pipeline. Respects Ω_k^true = +0.0098 everywhere."""

    def __init__(self, nside: int = 1024, gamma: float = 0.40, lmax: int = 2500) -> None:
        self.lattice = DiscreteNullLattice(gamma=gamma)
        self.cosmo = HQIVCosmology(gamma=gamma)
        self.pert = HQIVPerturbations(background=self.cosmo, gamma=gamma)
        self.nside = nside
        self.lmax = lmax

    def run_from_T_Pl_to_now(self) -> Dict[str, Any]:
        if hp is None:
            raise ImportError("healpy is required for HQIVCMBMap.run_from_T_Pl_to_now")

        # 1. Background (perfect)
        bg = self.lattice.evolve_to_cmb(T0_K=2.725)

        # 2. Primordial power from lattice invariant
        k = np.logspace(-5, 0, 800)
        Pk_prim = self.lattice.primordial_power_from_invariant(k)

        # 3. Lapse + curvature-aware transfer
        delta_T_transfer = self.pert.cosmological_transfer(
            k, z_recomb=1090.0, omega_k=self.cosmo.Ok0
        )

        # 4. Full-sky projection with galaxy accelerated motion (ISW)
        npix = hp.nside2npix(self.nside)
        delta_T = np.zeros(npix, dtype=float)
        for ipix in range(npix):
            theta, phi = hp.pix2ang(self.nside, ipix)
            los = self.cosmo.curved_line_of_sight(
                theta, phi, omega_k=self.cosmo.Ok0, k=k
            )
            isw = self.pert.isw_from_peculiar_velocity(
                theta, phi, omega_k=self.cosmo.Ok0
            )
            delta_T[ipix] = (
                np.sum(delta_T_transfer * los * self.cosmo.lapse_factor(z=0)) + isw
            )

        # 5. σ₈ from lapse-corrected growth
        sigma8 = self.pert.growth_to_sigma8(omega_k=self.cosmo.Ok0) * np.sqrt(
            np.mean(Pk_prim)
        )

        # 6. C_ℓ multipole
        cl = hp.anafast(delta_T * T_CMB_MUK, lmax=self.lmax)

        return {
            "T_map_muK": delta_T * T_CMB_MUK,
            "Cl_TT": cl,
            "sigma8": float(sigma8),
            "Omega_k_true": float(self.cosmo.Ok0),
            "background": bg,
        }

    def plot_multipole(
        self,
        result: Dict[str, Any],
        show: bool = True,
        out_path: Optional[str] = None,
    ) -> None:
        if plt is None:
            raise ImportError("matplotlib is required for plot_multipole")
        ell = np.arange(len(result["Cl_TT"]))
        d_ell = np.maximum(ell * (ell + 1) * result["Cl_TT"] / (2.0 * np.pi), 1e-10)
        plt.figure(figsize=(12, 7))
        plt.loglog(
            ell,
            d_ell,
            label=f"HQIV (Ω_k = {result['Omega_k_true']:.4f}, σ₈ = {result['sigma8']:.3f})",
        )
        plt.xlabel("Multipole ℓ")
        plt.ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$  [μK²]")
        plt.title("HQIV CMB Multipole — T_Pl to now (axiom-pure)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        if out_path:
            plt.savefig(out_path, dpi=120)
        if show:
            plt.show()
        else:
            plt.close()


def run_hqiv_cmb_to_map(
    n_side: int = 256,
    max_ell: int = 1500,
    include_polarization: bool = True,
    cosmology: Optional[HQIVCosmology] = None,
    bulk_seed: Optional[Any] = None,
    include_isw_rees_sciama: bool = True,
    frame_velocity_km_s: Optional[float] = None,
    frame_gal_l_deg: float = 264.0,
    frame_gal_b_deg: float = 48.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run HQIV CMB pipeline: C_ℓ → full-sky map in μK (cosmology_full wrapper).

    When bulk_seed is provided (from pyhqiv.bulk_seed.get_bulk_seed()), the pipeline
    uses HQIV bulk.py output as the authoritative seed until baryogenesis complete.
    """
    from pyhqiv import cosmology_full

    return cosmology_full.hqiv_cmb(
        n_side=n_side,
        max_ell=max(max_ell, 1500),
        include_polarization=include_polarization,
        cosmology=cosmology,
        bulk_seed=bulk_seed,
        frame_velocity_km_s=frame_velocity_km_s,
        frame_gal_l_deg=frame_gal_l_deg,
        frame_gal_b_deg=frame_gal_b_deg,
        **kwargs,
    )


__all__ = ["HQIVCMBMap", "run_hqiv_cmb_to_map"]
