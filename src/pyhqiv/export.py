"""
Export and hybrid interfaces: VESTA/OVITO charge-density format with HQIV corrections,
PySCF periodic hook (hqiv_potential_shift).

PySCF: use hqiv_potential_shift(phi_avg, dot_delta_theta_avg) as an additive
potential in pyscf.pbc; the shift is applied to the effective one-particle potential.

VESTA/OVITO: export_charge_density_* write 3D grids that can be loaded into
VESTA or OVITO; optional HQIV correction multiplies the density by a factor
(1 + gamma * phi/phi_ref) at each point for visualization of horizon modulation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import GAMMA
from pyhqiv.crystal import hqiv_potential_shift


def export_charge_density_vesta(
    grid: np.ndarray,
    cell: np.ndarray,
    out_path: str,
    phi_grid: Optional[np.ndarray] = None,
    gamma: float = GAMMA,
    phi_ref: float = 1.0,
) -> None:
    """
    Export 3D charge (or scalar) density to a format readable by VESTA.

    If phi_grid is provided (same shape as grid), the exported value at each point
    is grid * (1 + gamma * phi_grid/phi_ref) as an HQIV correction for visualization.

    Parameters
    ----------
    grid : 3D array
        Density or scalar field (e.g. from DFT).
    cell : (3, 3) array
        Lattice vectors (rows).
    out_path : str
        Output file path (.xsf or .cube recommended).
    phi_grid : 3D array, optional
        φ(x) on the same grid for HQIV modulation.
    gamma : float
        HQIV coefficient.
    phi_ref : float
        Reference φ for dimensionless correction.
    """
    data = np.asarray(grid, dtype=float)
    if phi_grid is not None:
        phi_grid = np.asarray(phi_grid, dtype=float)
        if phi_grid.shape != data.shape:
            raise ValueError("phi_grid must have the same shape as grid")
        data = data * (1.0 + gamma * phi_grid / max(phi_ref, 1e-30))
    with open(out_path, "w") as f:
        f.write("CRYSTAL\nPRIMVEC\n")
        for row in cell:
            f.write(f" {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        f.write("PRIMCOORD\n 1 1\n")
        f.write("0.0 0.0 0.0 0\n")
        f.write("BEGIN_BLOCK_DATAGRID_3D\n")
        f.write("density\n")
        f.write("BEGIN_DATAGRID_3D\n")
        nx, ny, nz = data.shape
        f.write(f" {nx} {ny} {nz}\n")
        f.write(" 0.0 0.0 0.0\n")
        for row in cell:
            f.write(f" {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f" {data[ix, iy, iz]:.6e}")
                f.write("\n")
        f.write("END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n")
    return None


def export_charge_density_ovito(
    grid: np.ndarray,
    cell: np.ndarray,
    out_path: str,
    phi_grid: Optional[np.ndarray] = None,
    gamma: float = GAMMA,
    phi_ref: float = 1.0,
) -> None:
    """
    Export 3D scalar field for OVITO (e.g. as a data file OVITO can load).

    Same HQIV correction as export_charge_density_vesta. Writes a simple 3D array
    with shape and origin/cell metadata in a header (custom text format).
    OVITO can import grid data via its Python interface or compatible formats.
    """
    data = np.asarray(grid, dtype=float)
    if phi_grid is not None:
        phi_grid = np.asarray(phi_grid, dtype=float)
        if phi_grid.shape != data.shape:
            raise ValueError("phi_grid must have the same shape as grid")
        data = data * (1.0 + gamma * phi_grid / max(phi_ref, 1e-30))
    with open(out_path, "w") as f:
        f.write("# HQIV-corrected scalar field for OVITO\n")
        f.write(f"# shape: {data.shape[0]} {data.shape[1]} {data.shape[2]}\n")
        f.write("# cell:\n")
        for row in cell:
            f.write(f"# {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        np.savetxt(f, data.ravel(), fmt="%.6e")
    return None


def pyscf_hqiv_shift(
    phi_avg: float,
    dot_delta_theta_avg: float,
    gamma: float = GAMMA,
) -> float:
    """
    HQIV potential shift for PySCF periodic calculations.

    Add this to the effective potential or use as a constant shift in the
    Hamiltonian so that band edges include the phase-horizon correction.
    Same as hqiv_potential_shift; this name is for discoverability.
    """
    return hqiv_potential_shift(phi_avg, dot_delta_theta_avg, gamma=gamma)
