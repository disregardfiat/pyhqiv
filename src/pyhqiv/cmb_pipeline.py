"""
HQIV CMB pipeline: full universe evolution → synthetic CMB map.

This module is the **entry point** for the first-principles, single-axiom CMB
pipeline that replaces the Boltzmann hierarchy + ΛCDM initial conditions with
the discrete null lattice + lapse-compressed perturbations.

Pipeline (see docs/HQIV_CMB_Pipeline.md for full design):

  1. Initialize background HQIV cosmology at z ≈ 1100 (recombination from lattice).
  2. Seed primordial fluctuations (scale-invariant from combinatorial invariant).
  3. Evolve perturbations with HQIV-modified Boltzmann equations (δT/T, velocity, f(φ)).
  4. Line-of-sight integration to z = 0 → full-sky T + polarization (E/B).
  5. Secondaries: lensing (φ-corrected LSS), ISW, Rees–Sciama (peculiar velocities).

Peculiar velocities and accelerated galactic motion: growth factor D(a) is
modified by f(φ) → distinctive low-ℓ power (CMB anomaly), v_pec ~ 300–600 km/s
today, and cross-correlations with DESI/Euclid.

Current status: design and stub. Background (lattice, cosmology) and linear
perturbations (HQIVPerturbations) exist; Boltzmann hierarchy replacement,
LOS integration, and map generation are not yet implemented.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pyhqiv.cosmology import HQIVCosmology
from pyhqiv.perturbations import HQIVPerturbations

__all__ = ["HQIVCMBPipeline", "cmb_pipeline_status"]


def cmb_pipeline_status() -> Dict[str, Any]:
    """
    Return status of CMB pipeline components (implemented vs planned).

    Use this to see what exists and what remains for full map generation.
    """
    cosmo = HQIVCosmology()
    result = cosmo.evolve_to_cmb()
    return {
        "background": "implemented",
        "lattice_evolve_to_cmb": "implemented",
        "Omega_k_true": result.get("Omega_true_k"),
        "lapse_compression": result.get("lapse_compression"),
        "perturbations_class": "implemented",
        "cosmological_perturbation": "implemented",
        "boltzmann_hierarchy": "not_implemented",
        "line_of_sight_integration": "not_implemented",
        "map_generation_healpix": "not_implemented",
        "secondaries_lensing_isw_rees_sciama": "not_implemented",
        "design_doc": "docs/HQIV_CMB_Pipeline.md",
    }


class HQIVCMBPipeline:
    """
    Full universe evolution from recombination (z ≈ 1100) to now (z = 0)
    and synthetic CMB map with lapse/φ corrections.

    Uses DiscreteNullLattice + HQIVCosmology for background; HQIVPerturbations
    for linear response. Final map includes secondaries (lensing, ISW,
    Rees–Sciama from peculiar velocities with f(φ)-modified D(a)).

    Usage (when implemented):

        pipeline = HQIVCMBPipeline()
        result = pipeline.run(z_rec=1100, n_side=256)
        # result["T_map"], result["E_map"], result["B_map"], result["C_ell"]

    Currently run() returns a minimal structure and raises NotImplementedError
    for full map generation; use cmb_pipeline_status() for component status.
    """

    def __init__(
        self,
        cosmology: Optional[HQIVCosmology] = None,
        gamma: float = 0.40,
        alpha: float = 0.60,
    ) -> None:
        self.cosmology = cosmology or HQIVCosmology(gamma=gamma, alpha=alpha)
        self.gamma = gamma
        self.alpha = alpha
        self._pert: Optional[HQIVPerturbations] = None

    @property
    def perturbations(self) -> HQIVPerturbations:
        """Lazy-build perturbation solver around cosmology background."""
        if self._pert is None:
            self._pert = HQIVPerturbations(
                self.cosmology, gamma=self.gamma, alpha=self.alpha
            )
        return self._pert

    def run(
        self,
        z_rec: float = 1100.0,
        n_side: Optional[int] = None,
        max_ell: int = 2000,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline from recombination to CMB map.

        When not implemented, returns a minimal dict with background and
        perturbation info and raises NotImplementedError for map generation.
        """
        result = self.cosmology.evolve_to_cmb()
        # Cosmological perturbation at a representative k, z
        delta_growth, f = self.perturbations.cosmological_perturbation(
            k=0.01, z=0.0
        )
        out = {
            "Omega_k_true": result["Omega_true_k"],
            "lapse_compression": result["lapse_compression"],
            "age_apparent_Gyr": result["age_apparent_Gyr"],
            "age_wall_Gyr": result["age_wall_Gyr"],
            "delta_growth_z0": delta_growth,
            "lapse_factor_z0": f,
            "z_rec": z_rec,
        }
        if n_side is not None:
            raise NotImplementedError(
                "Full CMB map generation (Boltzmann hierarchy replacement, "
                "line-of-sight integration, HEALPix maps, secondaries) is not "
                "yet implemented. See docs/HQIV_CMB_Pipeline.md and "
                "cmb_pipeline_status() for the roadmap."
            )
        out["max_ell"] = max_ell
        return out
