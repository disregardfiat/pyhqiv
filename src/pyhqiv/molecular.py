"""
Molecular / protein HQIV: Θ from Z and coordination, bond length from Θ, damping.

Single source of truth for PROtien and other molecular HQIV uses. Re-exports from
utils and constants so callers can do: from pyhqiv.molecular import theta_local, ...
"""

from __future__ import annotations

from pyhqiv.constants import A_LOC_ANG, GAMMA, HBAR_C_EV_ANG
from pyhqiv.utils import (
    bond_length_from_theta,
    damping_force_magnitude,
    phi_from_theta_local,
    theta_for_atom,
    theta_local,
)

__all__ = [
    "theta_local",
    "theta_for_atom",
    "bond_length_from_theta",
    "damping_force_magnitude",
    "phi_from_theta_local",
    "GAMMA",
    "HBAR_C_EV_ANG",
    "A_LOC_ANG",
]
