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
    max_ell: int = 2000,
    include_polarization: bool = True,
    cosmology: Optional[HQIVCosmology] = None,
    include_isw_rees_sciama: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run HQIV CMB pipeline: primordial → C_ℓ → full-sky map in μK.

    Delegates to cosmology_full.hqiv_cmb. Returns dict with T_map (μK),
    sigma8, ell, C_ell_*, and optionally Q_map, U_map.
    """
    from pyhqiv import cosmology_full

    return cosmology_full.hqiv_cmb(
        n_side=n_side,
        max_ell=max_ell,
        include_polarization=include_polarization,
        cosmology=cosmology,
        **kwargs,
    )
