"""
Defect and doping utilities: formation energy with HQIV vacuum correction,
charged-defect supercell helpers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from pyhqiv.constants import GAMMA
from pyhqiv.crystal import hqiv_potential_shift


def formation_energy(
    E_defect: float,
    E_bulk: float,
    n_defect: int = 1,
    mu_removed: Optional[float] = None,
    mu_added: Optional[float] = None,
    q: int = 0,
    E_vacuum: float = 0.0,
    phi_avg_defect: float = 0.0,
    phi_avg_bulk: float = 0.0,
    dot_delta_theta_avg: float = 0.0,
    gamma: float = GAMMA,
) -> float:
    """
    Defect formation energy with optional HQIV vacuum correction.

    ΔH_f = E_defect - E_bulk + sum_i n_i μ_i + q * (E_vacuum + V_HQIV).

    The HQIV correction to the reference vacuum (for charged defects) is
    V_HQIV = hqiv_potential_shift(φ_avg, δ̇θ′) so that the alignment between
    defect and bulk supercells uses the same horizon potential.

    Parameters
    ----------
    E_defect : float
        Total energy of defect supercell.
    E_bulk : float
        Total energy of bulk supercell (same size).
    n_defect : int
        Number of defect sites (e.g. 1 for single vacancy).
    mu_removed : float, optional
        Chemical potential of removed atoms (e.g. μ_Si for vacancy).
    mu_added : float, optional
        Chemical potential of added atoms (e.g. for interstitial).
    q : int
        Defect charge state.
    E_vacuum : float
        Reference vacuum level (e.g. from bulk band structure).
    phi_avg_defect : float
        Average φ in defect supercell (for HQIV alignment).
    phi_avg_bulk : float
        Average φ in bulk supercell.
    dot_delta_theta_avg : float
        Average δ̇θ′ (shared).
    gamma : float
        HQIV monogamy coefficient.

    Returns
    -------
    float
        Formation energy (same units as E_* and μ).
    """
    dE = E_defect - E_bulk
    if mu_removed is not None:
        dE += n_defect * mu_removed
    if mu_added is not None:
        dE -= n_defect * mu_added
    v_hqiv_def = hqiv_potential_shift(phi_avg_defect, dot_delta_theta_avg, gamma=gamma)
    v_hqiv_bulk = hqiv_potential_shift(phi_avg_bulk, dot_delta_theta_avg, gamma=gamma)
    dE += q * (E_vacuum + 0.5 * (v_hqiv_def + v_hqiv_bulk))
    return float(dE)


def charged_defect_supercell(
    lattice_vectors: np.ndarray,
    positions: np.ndarray,
    charges: List[float],
    defect_charge: int = 0,
    supercell_shape: Tuple[int, int, int] = (2, 2, 2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a charged-defect supercell: replicate lattice and return positions,
    charges, and defect index. Does not add compensating background; user should
    apply jellium or similar in the electronic structure code.

    Parameters
    ----------
    lattice_vectors : (3, 3) array
        Unit cell lattice vectors.
    positions : (n, 3) array
        Fractional or Cartesian positions in unit cell (Cartesian assumed if max > 2).
    charges : list of float
        Per-atom charges in unit cell.
    defect_charge : int
        Total charge of the defect (e.g. +1 for V_Si^+).
    supercell_shape : (3,) tuple
        Replication (n1, n2, n3).

    Returns
    -------
    pos_sc : (N, 3) array
        Supercell positions (Cartesian).
    charges_sc : (N,) array
        Supercell charges.
    defect_center : (3,) array
        Approximate defect position (centre of first cell).
    """
    lat = np.asarray(lattice_vectors, dtype=float)
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
    frac = np.all(np.abs(pos) <= 1.5) and np.all(pos >= -0.5)
    if not frac:
        pos_frac = np.linalg.solve(lat.T, pos.T).T
    else:
        pos_frac = pos
    ch = np.asarray(charges, dtype=float).ravel()
    _ = len(ch)  # number of cells (for future use)
    n1, n2, n3 = supercell_shape
    positions_list: List[np.ndarray] = []
    charges_list: List[float] = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                shift = np.array([i, j, k], dtype=float)
                positions_list.append(pos_frac + shift)
                charges_list.extend(ch.tolist())
    pos_frac_sc = np.vstack(positions_list)
    pos_sc = pos_frac_sc @ lat
    charges_sc = np.array(charges_list)
    defect_center = 0.5 * (pos_sc[0] + pos_sc[min(1, pos_sc.shape[0] - 1)])
    return pos_sc, charges_sc, defect_center
