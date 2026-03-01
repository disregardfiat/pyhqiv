"""
HQIV crystal: periodic boundary conditions, Bloch sum, lattice vectors.
Minimal extension for full periodic DFT + HQIV corrections (band gaps, stress).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from pyhqiv.atom import HQIVAtom
from pyhqiv.constants import GAMMA
from pyhqiv.system import HQIVSystem


class HQIVCrystal(HQIVSystem):
    """
    HQIV system with lattice vectors and PBC. Replicates the unit cell into a
    supercell for Bloch-phase sums. Observer-centric horizon φ(x) and δ̇θ′
    modulation enter via phase-lifted Bloch states.
    """

    def __init__(
        self,
        atoms: List[HQIVAtom],
        lattice_vectors: np.ndarray,
        gamma: float = GAMMA,
        supercell_shape: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__(atoms, gamma=gamma)
        self.lattice_vectors = np.asarray(lattice_vectors, dtype=float)
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError("lattice_vectors must be 3×3")
        self.supercell_shape = tuple(supercell_shape)
        self._supercell_positions: Optional[np.ndarray] = None
        self._supercell_charges: Optional[np.ndarray] = None

    @property
    def a1(self) -> np.ndarray:
        """First lattice vector."""
        return self.lattice_vectors[0]

    @property
    def a2(self) -> np.ndarray:
        """Second lattice vector."""
        return self.lattice_vectors[1]

    @property
    def a3(self) -> np.ndarray:
        """Third lattice vector."""
        return self.lattice_vectors[2]

    def _build_supercell(self) -> Tuple[np.ndarray, np.ndarray]:
        """Replicate unit-cell positions and charges under PBC."""
        if self._supercell_positions is not None:
            return self._supercell_positions, self._supercell_charges
        pos = self.positions
        ch = self.charges
        n1, n2, n3 = self.supercell_shape
        positions_list: List[np.ndarray] = []
        charges_list: List[float] = []
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    R = i * self.a1 + j * self.a2 + k * self.a3
                    positions_list.append(pos + R)
                    charges_list.extend(ch.tolist())
        self._supercell_positions = np.vstack(positions_list)
        self._supercell_charges = np.array(charges_list)
        return self._supercell_positions, self._supercell_charges

    def bloch_sum(
        self,
        k_point: Union[np.ndarray, Tuple[float, float, float]],
        phase_modulation: Optional[np.ndarray] = None,
    ) -> complex:
        """
        Phase-lifted Bloch sum: Σ_R exp(i k·R) over supercell lattice points.
        If phase_modulation is provided (per replica), multiplies as exp(i δθ′) style.
        Returns complex amplitude (e.g. for overlap or structure factor).
        """
        k = np.asarray(k_point, dtype=float).ravel()[:3]
        pos_sc, ch_sc = self._build_supercell()
        n1, n2, n3 = self.supercell_shape
        n_cell = len(self.atoms)
        total = 0.0 + 0.0j
        idx = 0
        for i in range(n1):
            for j in range(n2):
                for kk in range(n3):
                    R = i * self.a1 + j * self.a2 + kk * self.a3
                    phase = np.exp(1j * np.dot(k, R))
                    if phase_modulation is not None and phase_modulation.size > idx:
                        phase *= np.exp(1j * phase_modulation.ravel()[idx])
                    total += phase * np.sum(ch_sc[idx : idx + n_cell])
                    idx += n_cell
        return total

    def supercell_positions(self) -> np.ndarray:
        """(N_super, 3) positions in the repeated supercell."""
        pos, _ = self._build_supercell()
        return pos

    def supercell_charges(self) -> np.ndarray:
        """(N_super,) charges in the supercell."""
        _, ch = self._build_supercell()
        return ch

    def reciprocal_vectors(self) -> np.ndarray:
        """Reciprocal lattice vectors (2π × standard definition). b_i · a_j = 2π δ_ij."""
        vol = np.abs(np.linalg.det(self.lattice_vectors))
        if vol < 1e-30:
            raise ValueError("Lattice vectors are singular")
        b1 = 2.0 * np.pi * np.cross(self.a2, self.a3) / vol
        b2 = 2.0 * np.pi * np.cross(self.a3, self.a1) / vol
        b3 = 2.0 * np.pi * np.cross(self.a1, self.a2) / vol
        return np.array([b1, b2, b3])

    def volume(self) -> float:
        """Unit cell volume."""
        return float(np.abs(np.linalg.det(self.lattice_vectors)))


def high_symmetry_k_path(
    lattice_vectors: np.ndarray,
    path: Union[str, List[Tuple[str, Tuple[float, float, float]]]],
    npoints: int = 50,
    special_points: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    """
    Generate k-points along a high-symmetry path in reciprocal space (HQIV-aware).

    Parameters
    ----------
    lattice_vectors : (3, 3) array
        Unit cell lattice vectors (rows).
    path : str or list
        If str: sequence of special point labels, e.g. "GXWG" or "G-X-W-G".
        If list: [(label, (kx, ky, kz)_frac), ...] in fractional reciprocal coords.
    npoints : int
        Total number of k-points along the full path.
    special_points : dict, optional
        Map label -> (kx, ky, kz) in fractional coordinates of reciprocal lattice.
        Default: cubic/fcc common points (G, X, W, K, L, U).

    Returns
    -------
    kpts_cart : (npoints, 3) array
        K-points in Cartesian reciprocal coordinates (e.g. 1/Å).
    kpts_frac : (npoints, 3) array
        K-points in fractional reciprocal coordinates.
    path_segments : list of (label, index)
        Segment start labels and indices for plotting.
    """
    lat = np.asarray(lattice_vectors, dtype=float)
    if lat.shape != (3, 3):
        raise ValueError("lattice_vectors must be 3×3")
    vol = np.abs(np.linalg.det(lat))
    if vol < 1e-30:
        raise ValueError("Lattice vectors are singular")
    b1 = 2.0 * np.pi * np.cross(lat[1], lat[2]) / vol
    b2 = 2.0 * np.pi * np.cross(lat[2], lat[0]) / vol
    b3 = 2.0 * np.pi * np.cross(lat[0], lat[1]) / vol
    rec = np.array([b1, b2, b3])

    if special_points is None:
        # Cubic / FCC common high-symmetry points (fractional coords in reciprocal)
        special_points = {
            "G": (0, 0, 0),
            "Gamma": (0, 0, 0),
            "X": (0.5, 0, 0.5),
            "W": (0.5, 0.25, 0.75),
            "K": (0.375, 0.375, 0.75),
            "L": (0.5, 0.5, 0.5),
            "U": (0.625, 0.25, 0.625),
        }

    if isinstance(path, str):
        path_str = path.replace("-", " ").split()
        path_str = [p.strip() for p in path_str if p.strip()]
        if not path_str:
            path_str = list(path.strip())
        elif len(path_str) == 1 and path_str[0] not in (special_points or {}):
            path_str = list(path_str[0])  # "GXWG" -> ["G","X","W","G"]
        pts_frac = []
        for label in path_str:
            if label not in special_points:
                raise ValueError(f"Unknown special point: {label}")
            pts_frac.append((label, np.array(special_points[label], dtype=float)))
    else:
        pts_frac = [(str(label), np.asarray(k, dtype=float)) for label, k in path]

    if len(pts_frac) < 2:
        raise ValueError("Path must have at least 2 points")

    all_frac = []
    segment_starts: List[Tuple[str, int]] = []
    nsegments = len(pts_frac) - 1
    per_seg = max(1, npoints // nsegments)
    for i in range(nsegments):
        label, k0 = pts_frac[i]
        _, k1 = pts_frac[i + 1]
        segment_starts.append((label, len(all_frac)))
        n_here = per_seg if i < nsegments - 1 else max(1, npoints - len(all_frac))
        for t in np.linspace(0, 1, n_here, endpoint=(i == nsegments - 1)):
            all_frac.append(k0 + t * (k1 - k0))
    segment_starts.append((pts_frac[-1][0], len(all_frac)))
    kpts_frac = np.array(all_frac)
    kpts_cart = kpts_frac @ rec
    return kpts_cart, kpts_frac, segment_starts


def hqiv_potential_shift(
    phi_avg: float,
    dot_delta_theta_avg: float,
    gamma: float = GAMMA,
) -> float:
    """
    HQIV effective potential shift for band structure: V_shift ∝ γ φ δ̇θ′.
    Use in PySCF or other DFT as an additive constant (or diagonal shift).
    Units: same as φ and δ̇θ′ (e.g. eV if φ in (eV) and δ̇θ′ in 1/s with ℏ).
    """
    return gamma * phi_avg * dot_delta_theta_avg
