"""
HQIV CMB: C_ℓ + σ₈ + full-sky map. Two paths:

- HQIVCMBMap (here and in cmb_map): axiom-pure, respects Ω_k^true = +0.0098
  (curved_line_of_sight, curvature in transfer, ISW, growth_to_sigma8).
- run_hqiv_cmb_to_map: thin wrapper over cosmology_full (phenomenological).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pyhqiv.cosmology import HQIVCosmology
from pyhqiv.cosmology.cmb_map import HQIVCMBMap

__all__ = ["HQIVCMBMap", "run_hqiv_cmb_to_map"]


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
    Run HQIV CMB pipeline: C_ℓ → full-sky map in μK.

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
