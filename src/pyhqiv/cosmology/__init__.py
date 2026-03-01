"""
HQIV Cosmology: lattice evolution, Ω_k, lapse, and optional T_Pl → now CMB.

Package contents:
- HQIVCosmology: background from discrete null lattice (evolve_to_cmb, omega_k_true).
- HQIVUniverseEvolver: full-sky CMB from Planck epoch to now (run_from_T_Pl_to_now).

Core axiom: E_tot = m c² + ħ c/Δx with Δx ≤ Θ_local(x) → φ = 2c²/Θ_local,
lapse compression f(a_loc, φ) = a_loc/(a_loc + φ/6).
"""

from pyhqiv.cosmology.background import HQIVCosmology
from pyhqiv.cosmology.universe_evolver import HQIVUniverseEvolver

__all__ = ["HQIVCosmology", "HQIVUniverseEvolver"]
