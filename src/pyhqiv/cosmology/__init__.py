"""
HQIV Cosmology: lattice evolution, Ω_k, lapse, and optional T_Pl → now CMB.

.. warning::
   Experimental. All features are experimental. The CMB pipeline has known issues.
   Public contribution and feedback are greatly appreciated.

Package contents
----------------
- **HQIVCosmology**: Background from discrete null lattice (evolve_to_cmb, omega_k_true).
- **HQIVCMBMap**: Axiom-pure CMB orchestrator (run_from_T_Pl_to_now → map, C_ℓ, σ₈).
- **HQIVUniverseEvolver**: Full-sky CMB from Planck epoch to now (run_from_T_Pl_to_now).

Core axiom: E_tot = m c² + ħ c/Δx with Δx ≤ Θ_local(x) → φ = 2c²/Θ_local,
lapse compression f(a_loc, φ) = a_loc/(a_loc + φ/6).
"""

from pyhqiv.cosmology.background import HQIVCosmology
from pyhqiv.cosmology.hqiv_cmb import HQIVCMBMap
from pyhqiv.cosmology.universe_evolver import HQIVUniverseEvolver

__all__ = ["HQIVCosmology", "HQIVCMBMap", "HQIVUniverseEvolver"]
