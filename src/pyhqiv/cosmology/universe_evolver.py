"""
Full universe evolver: T_Pl → now with CMB map + σ₈ (phenomenological).

HQIVUniverseEvolver(nside=1024).run_from_T_Pl_to_now() returns a full-sky map
(T_map_muK), σ₈, and C_ℓ. Currently these are **phenomenological**: the map
is synfast(C_ℓ_template), σ₈ from growth + P(k) template, C_ℓ from a template —
not from first-principles primordial seeding, forward evolution, or
line-of-sight projection. See docs/HQIV_CMB_Pipeline.md §0.1 for what's missing.

Install pyhqiv[cosmology] for healpy map generation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pyhqiv.cosmology import HQIVCosmology
from pyhqiv.cosmology.hqiv_cmb import run_hqiv_cmb_to_map


class HQIVUniverseEvolver:
    """
    Evolve from Planck epoch to now and produce CMB map + σ₈ (phenomenological).

    Returns T_map_muK (from C_ℓ template + synfast), σ₈ (growth + P(k) template),
    and C_ℓ. Not yet from first-principles: no primordial seeding, no forward
    evolution of δT/T, no project_to_sky() or anafast(projected map). See
    cmb_pipeline_status() and docs/HQIV_CMB_Pipeline.md.

    Usage:

        evolver = HQIVUniverseEvolver(nside=1024)
        result = evolver.run_from_T_Pl_to_now()
        hp.mollview(result["T_map_muK"], title="HQIV CMB from Planck epoch to now")
        print(f"σ₈ = {result['sigma8']:.4f}")
    """

    def __init__(
        self,
        nside: int = 256,
        cosmology: Optional[HQIVCosmology] = None,
        max_ell: int = 1500,
        bulk_seed: Optional[Any] = None,
        frame_velocity_km_s: Optional[float] = None,
        frame_gal_l_deg: float = 264.0,
        frame_gal_b_deg: float = 48.0,
    ) -> None:
        self.nside = nside
        self.cosmology = cosmology or HQIVCosmology()
        self.max_ell = max(max_ell, 1500)
        self.bulk_seed = bulk_seed
        self.frame_velocity_km_s = frame_velocity_km_s
        self.frame_gal_l_deg = frame_gal_l_deg
        self.frame_gal_b_deg = frame_gal_b_deg

    def run_from_T_Pl_to_now(self) -> Dict[str, Any]:
        """
        Run full pipeline from Planck epoch to z=0; return map in μK and σ₈.

        Returns dict with at least:
          - T_map_muK: full-sky HEALPix map in μK (or None if healpy not installed)
          - sigma8: σ₈(z=0)
          - ell, C_ell_TT, (C_ell_EE, C_ell_TE, C_ell_BB if polarization)
          - n_side, cosmology
        """
        raw = run_hqiv_cmb_to_map(
            n_side=self.nside,
            max_ell=self.max_ell,
            include_polarization=True,
            cosmology=self.cosmology,
            bulk_seed=self.bulk_seed,
            frame_velocity_km_s=self.frame_velocity_km_s,
            frame_gal_l_deg=self.frame_gal_l_deg,
            frame_gal_b_deg=self.frame_gal_b_deg,
        )
        # Healpy synfast returns maps in μK when C_ℓ are in μK²; expose as T_map_muK
        t_map = raw.get("T_map")
        out = {
            "T_map_muK": t_map,
            "sigma8": raw["sigma8"],
            "ell": raw["ell"],
            "C_ell_TT": raw["C_ell_TT"],
            "n_side": raw.get("n_side"),
            "cosmology": raw.get("cosmology"),
        }
        if "C_ell_EE" in raw:
            out["C_ell_EE"] = raw["C_ell_EE"]
            out["C_ell_TE"] = raw["C_ell_TE"]
            out["C_ell_BB"] = raw["C_ell_BB"]
        if "Q_map" in raw:
            out["Q_map_muK"] = raw["Q_map"]
            out["U_map_muK"] = raw["U_map"]
        return out
