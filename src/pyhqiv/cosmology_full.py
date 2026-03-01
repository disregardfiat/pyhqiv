"""
Optional heavy cosmology module: full universe evolution → CMB map.

This module is the **optional** home for the full pipeline (heavy dependencies).
Perturbations and the lattice stay in **main** (pyhqiv.perturbations, pyhqiv.lattice).

Planned API (all stubs until implemented):

- universe_evolver: seed from lattice, evolve perturbations with lapse to z=0.
- hqiv_cmb: run full CMB pipeline (primordial → LOS → secondaries).
- sigma8: amplitude of matter fluctuations (R=8 h⁻¹ Mpc) from HQIV growth.
- c_ell_spectrum: C_ℓ TT / EE / TE / BB multipole power spectrum (and chart).
- full_sky_healpy_map: HEALPix full-sky T, Q, U with ISW/Rees–Sciama.
- line_of_sight_isw_rees_sciama: LOS projection with galaxy accelerated motion.

Install: pip install pyhqiv[cosmology] for healpy and related deps.
Design: docs/HQIV_CMB_Pipeline.md
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "universe_evolver",
    "hqiv_cmb",
    "sigma8",
    "c_ell_spectrum",
    "full_sky_healpy_map",
    "line_of_sight_isw_rees_sciama",
]

_NOT_IMPL = (
    "Full cosmology/CMB pipeline (universe evolver, Healpy maps, C_ℓ, σ₈, "
    "LOS/ISW) is not yet implemented. Install pyhqiv[cosmology] when available; "
    "see docs/HQIV_CMB_Pipeline.md."
)


def universe_evolver(
    z_start: float = 1100.0,
    z_end: float = 0.0,
    n_steps: int = 500,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evolve universe from z_start (e.g. recombination) to z_end (now).

    Seeds primordial fluctuations from the lattice combinatorial invariant;
    evolves perturbations forward with lapse f(φ); returns transfer functions
    and growth for LOS projection. Heavy: requires integration of HQIV-modified
    perturbation equations.
    """
    raise NotImplementedError(_NOT_IMPL)


def hqiv_cmb(
    n_side: int = 256,
    max_ell: int = 2000,
    include_polarization: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run full HQIV CMB pipeline: primordial → Boltzmann → LOS → map.

    Returns full-sky maps (T, Q, U if polarization), C_ℓ, and optional σ₈.
    Uses lattice + perturbations from main package; Healpy for map output.
    """
    raise NotImplementedError(_NOT_IMPL)


def sigma8(
    z: float = 0.0,
    cosmology: Optional[Any] = None,
    **kwargs: Any,
) -> float:
    """
    Amplitude of matter fluctuations at R = 8 h⁻¹ Mpc.

    σ₈ from HQIV growth factor D(a) with f(φ) and curvature imprint.
    """
    raise NotImplementedError(_NOT_IMPL)


def c_ell_spectrum(
    spectrum_type: str = "TT",
    max_ell: int = 2000,
    cosmology: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    C_ℓ multipole power spectrum (and optional chart/plot data).

    spectrum_type: 'TT', 'EE', 'BB', 'TE', etc.
    Returns (ell, C_ell) arrays.
    """
    raise NotImplementedError(_NOT_IMPL)


def full_sky_healpy_map(
    n_side: int = 256,
    map_type: str = "T",
    include_isw_rees_sciama: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Full-sky HEALPix map (T and/or Q, U) with lapse and secondaries.

    include_isw_rees_sciama: add ISW + Rees–Sciama from galaxy accelerated motion.
    Requires healpy (pip install pyhqiv[cosmology]).
    """
    raise NotImplementedError(_NOT_IMPL)


def line_of_sight_isw_rees_sciama(
    ell: np.ndarray,
    z_range: Tuple[float, float] = (0.0, 1100.0),
    n_z: int = 200,
    cosmology: Optional[Any] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Line-of-sight projection: ISW + Rees–Sciama contribution to C_ℓ.

    Uses HQIV growth D(a) and f(φ) for peculiar velocities and potential decay.
    Returns ΔC_ℓ (additive to primary C_ℓ) or full C_ℓ^{ISW+RS}.
    """
    raise NotImplementedError(_NOT_IMPL)
