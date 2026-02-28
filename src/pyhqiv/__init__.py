"""
pyhqiv — Horizon-Quantized Informational Vacuum (HQIV) framework.

Paper: Steven Ettinger Jr, February 22, 2026.
"""

from pyhqiv.constants import (
    GAMMA,
    ALPHA,
    T_PL_GEV,
    T_LOCK_GEV,
    T_CMB_K,
    M_TRANS,
    COMBINATORIAL_INVARIANT,
    OMEGA_TRUE_K_PAPER,
    LAPSE_COMPRESSION_PAPER,
    AGE_WALL_GYR_PAPER,
    AGE_APPARENT_GYR_PAPER,
)
from pyhqiv.algebra import OctonionHQIVAlgebra
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.phase import HQIVPhaseLift
from pyhqiv.atom import HQIVAtom
from pyhqiv.system import HQIVSystem
from pyhqiv.fields import PhaseHorizonFDTD

__all__ = [
    "GAMMA",
    "ALPHA",
    "T_PL_GEV",
    "T_LOCK_GEV",
    "T_CMB_K",
    "M_TRANS",
    "COMBINATORIAL_INVARIANT",
    "OMEGA_TRUE_K_PAPER",
    "LAPSE_COMPRESSION_PAPER",
    "AGE_WALL_GYR_PAPER",
    "AGE_APPARENT_GYR_PAPER",
    "OctonionHQIVAlgebra",
    "DiscreteNullLattice",
    "HQIVPhaseLift",
    "HQIVAtom",
    "HQIVSystem",
    "PhaseHorizonFDTD",
]

__version__ = "0.1.0"
