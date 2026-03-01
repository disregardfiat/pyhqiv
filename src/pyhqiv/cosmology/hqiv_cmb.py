"""
HQIV CMB: C_ℓ + σ₈ + full-sky Healpy map from cosmology_full.

Thin wrapper over pyhqiv.cosmology_full so the cosmology package exposes
a single T_Pl → now API. Replace this module with the full simulator when ready.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pyhqiv.cosmology import HQIVCosmology


def run_hqiv_cmb_to_map(
    n_side: int = 256,
    max_ell: int = 1500,
    include_polarization: bool = True,
    cosmology: Optional[HQIVCosmology] = None,
    include_isw_rees_sciama: bool = True,
    frame_velocity_km_s: Optional[float] = None,
    frame_gal_l_deg: float = 264.0,
    frame_gal_b_deg: float = 48.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run HQIV CMB pipeline: C_ℓ → full-sky map in μK.

    Delegates to cosmology_full.hqiv_cmb. Multipoles out to max_ell (≥1500).
    Set frame_velocity_km_s to add kinematic dipole and generate low-ℓ region.
    """
    from pyhqiv import cosmology_full

    return cosmology_full.hqiv_cmb(
        n_side=n_side,
        max_ell=max(max_ell, 1500),
        include_polarization=include_polarization,
        cosmology=cosmology,
        frame_velocity_km_s=frame_velocity_km_s,
        frame_gal_l_deg=frame_gal_l_deg,
        frame_gal_b_deg=frame_gal_b_deg,
        **kwargs,
    )
