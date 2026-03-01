"""
HQIV CMB pipeline: full universe evolution → synthetic CMB map.

This module is the **entry point** for the CMB pipeline. The design
(docs/HQIV_CMB_Pipeline.md) is first-principles: lattice + lapse replace
Boltzmann hierarchy + ΛCDM initial conditions. Pipeline steps:

  1. Initialize background at z ≈ 1100 (lattice).
  2. Seed primordial fluctuations (combinatorial invariant).
  3. Evolve perturbations (δT/T, velocity, f(φ)).
  4. Line-of-sight integration to z = 0 → full-sky T + polarization.
  5. Secondaries: lensing, ISW, Rees–Sciama (peculiar velocities).

**Current status:** The pipeline stops at scalar background evolution. Steps
2–4 (primordial seeding, forward evolution, project_to_sky) are **not**
implemented. What exists: background + point-wise cosmological_perturbation;
optional cosmology_full provides **phenomenological** σ₈, C_ℓ template, and
Healpy map (synfast(C_ℓ)), not from a projected sky. Use cmb_pipeline_status()
for the exact gap; see doc §0.1 for "what's missing / why it feels wrong".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pyhqiv.cosmology import HQIVCosmology
from pyhqiv.perturbations import HQIVPerturbations

__all__ = ["HQIVCMBPipeline", "cmb_pipeline_status"]


def cmb_pipeline_status() -> Dict[str, Any]:
    """
    Return status of CMB pipeline components (implemented vs planned).

    Use this to see what exists and what remains. The pipeline currently
    stops at scalar background; σ₈, C_ℓ, and Healpy map are phenomenological
    (templates / growth-based), not from first-principles LOS projection.
    See docs/HQIV_CMB_Pipeline.md §0.1 for "what's missing".
    """
    cosmo = HQIVCosmology()
    result = cosmo.evolve_to_cmb()
    return {
        "background": "implemented",
        "lattice_evolve_to_cmb": "implemented",
        "Omega_k_true": result.get("Omega_true_k"),
        "lapse_compression": result.get("lapse_compression"),
        "perturbations_class": "implemented",
        "cosmological_perturbation": "implemented (point-wise k,z only)",
        # First-principles chain: not implemented
        "primordial_seeding": "not_implemented",
        "forward_evolution_boltzmann": "not_implemented",
        "line_of_sight_projection": "not_implemented (no project_to_sky)",
        "c_ell_from_sky": "not_implemented (no anafast(projected map))",
        "sigma8_from_evolved_field": "not_implemented",
        "nonlinear_rees_sciama": "not_implemented",
        "boltzmann_hierarchy": "not_implemented",
        # Phenomenological (implemented in cosmology_full)
        "sigma8": "phenomenological (cosmology_full.sigma8: growth + P(k) template)",
        "c_ell_spectrum": "phenomenological (cosmology_full.c_ell_spectrum: template)",
        "map_generation_healpix": "phenomenological (synfast(C_ell_template), not from projected sky)",
        "line_of_sight_isw_delta_cl": "phenomenological (cosmology_full.line_of_sight_isw_rees_sciama)",
        "universe_evolver": "implemented (cosmology_full.universe_evolver: z, a, D, f)",
        "design_doc": "docs/HQIV_CMB_Pipeline.md",
        "optional_module": "pyhqiv.cosmology_full (install pyhqiv[cosmology] for healpy map)",
    }


class HQIVCMBPipeline:
    """
    Full universe evolution from recombination to now (design); stub implementation.

    Uses DiscreteNullLattice + HQIVCosmology for background; HQIVPerturbations
    for point-wise linear response. First-principles map (seed → evolve →
    project_to_sky → anafast) is not implemented: run(n_side=...) raises
    NotImplementedError. For phenomenological map/σ₈/C_ℓ use
    HQIVUniverseEvolver (cosmology package). See cmb_pipeline_status() and
    docs/HQIV_CMB_Pipeline.md §0.1 for what's missing.
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
            self._pert = HQIVPerturbations(self.cosmology, gamma=self.gamma, alpha=self.alpha)
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
        delta_growth, f = self.perturbations.cosmological_perturbation(k=0.01, z=0.0)
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
