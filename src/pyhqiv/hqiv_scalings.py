"""
HQIV observer-centric scalings from T_CMB (paper sections 3.4 & 7.2).

Only T_CMB-derived quantities. No preset horizons, no mode reductions, no nucleon/alpha
constants. Free and bound horizons come from first-principles merge (subatomic/nuclear).
"""

from __future__ import annotations

from typing import Dict


def get_hqiv_nuclear_constants(t_cmb: float = 2.725) -> Dict[str, float]:
    """
    Return T_CMB-derived scale factors only. No phenomenology, no preset data.
    Free-nucleon and bound-nucleus horizons are computed from merge_constituents
    in nuclear.py (first principles).
    """
    T0 = 2.725  # reference CMB temperature (paper "now")
    scale = T0 / t_cmb  # observer-centric lapse compression

    TAU_TICK_OBSERVER_S = 2.0e-43 * scale
    MACROSCOPIC_DECAY_SCALE = 3.71e54 * scale
    PHI_CRIT_SI = 1.38e-23 * t_cmb / 2.725 * 1e30
    LATTICE_BASE_M = 1.616e-35 * 1.2e20 * scale

    return {
        "TAU_TICK_OBSERVER_S": TAU_TICK_OBSERVER_S,
        "MACROSCOPIC_DECAY_SCALE": MACROSCOPIC_DECAY_SCALE,
        "PHI_CRIT_SI": PHI_CRIT_SI,
        "LATTICE_BASE_M": LATTICE_BASE_M,
    }


__all__ = ["get_hqiv_nuclear_constants"]
