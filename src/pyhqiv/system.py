"""
HQIVSystem: multi-atom system, collective horizon with monogamy γ,
vectorized E/B/D/H on 3D grid, total Hamiltonian (QuTiP-ready). from_pdb optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from pyhqiv.atom import HQIVAtom
from pyhqiv.constants import C_SI, GAMMA


def _coulomb_field(
    grid: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    epsilon_eff: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E and B (B=0 for static) from point charges. grid (N, 3); positions (M, 3); charges (M,).
    If epsilon_eff is (N,) or (N, 1), D = epsilon_eff * E.
    """
    grid = np.asarray(grid)
    if grid.ndim == 1:
        grid = grid.reshape(1, 3)
    N = grid.shape[0]
    positions = np.asarray(positions).reshape(-1, 3)
    charges = np.asarray(charges).ravel()
    M = len(charges)

    E = np.zeros((N, 3))
    for i in range(M):
        r = grid - positions[i]
        d = np.linalg.norm(r, axis=1, keepdims=True)
        d = np.maximum(d, 1e-12)
        E += charges[i] * r / (d ** 3)

    if epsilon_eff is not None:
        eps = np.asarray(epsilon_eff).reshape(-1, 1)
        if eps.size == 1:
            E = E * (1.0 / eps)
        else:
            E = E / np.maximum(eps, 1e-30)

    B = np.zeros_like(E)
    return E, B


def _phase_corrected_epsilon(
    grid: np.ndarray,
    atoms: List[HQIVAtom],
    gamma: float,
    E_prime: float = 0.5,
) -> np.ndarray:
    """Effective 1/ε from horizon: 1 + γ φ/Λ² style; here use φ/c² and ˙δθ′/c."""
    phi_sum = np.zeros(grid.shape[0], dtype=float)
    for at in atoms:
        phi_sum += at.phi_local(grid)
    phi_over_c2 = phi_sum / (C_SI ** 2)
    dtdc = np.arctan(E_prime) * (np.pi / 2.0) / C_SI
    # Constitutive: ε_eff ∝ 1 + γ φ/c² (˙δθ′/c) for phase-horizon correction
    return 1.0 + gamma * phi_over_c2 * dtdc


class HQIVSystem:
    """
    Multi-atom HQIV system: list of HQIVAtom, collective horizon with monogamy γ,
    vectorized E/B (and D/H via constitutive relations) on any 3D grid.
    """

    def __init__(
        self,
        atoms: List[HQIVAtom],
        gamma: float = GAMMA,
    ) -> None:
        self.atoms = list(atoms)
        self.gamma = gamma

    @property
    def positions(self) -> np.ndarray:
        """(M, 3) positions."""
        return np.array([a.position for a in self.atoms])

    @property
    def charges(self) -> np.ndarray:
        """(M,) charges."""
        return np.array([a.charge for a in self.atoms])

    @classmethod
    def from_atoms(
        cls,
        positions: Union[List[Tuple[float, float, float]], np.ndarray],
        charges: Optional[List[float]] = None,
        species: Optional[List[str]] = None,
        gamma: float = GAMMA,
    ) -> "HQIVSystem":
        """Build HQIVSystem from list of positions and optional charges/species."""
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
        M = positions.shape[0]
        if charges is None:
            charges = [0.0] * M
        if species is None:
            species = ["X"] * M
        atoms = [
            HQIVAtom(positions[i], charge=charges[i], species=species[i])
            for i in range(M)
        ]
        return cls(atoms, gamma=gamma)

    @classmethod
    def from_pdb(
        cls,
        path: Union[str, Path],
        gamma: float = GAMMA,
    ) -> "HQIVSystem":
        """
        Load atoms from PDB file. Requires optional dependency ase or MDAnalysis.
        Falls back to empty system with warning if neither available.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        try:
            import ase.io
            atoms_ase = ase.io.read(str(path))
            positions = atoms_ase.get_positions()
            symbols = atoms_ase.get_chemical_symbols()
            # Simple charge from valence (approximate)
            charge_map = {"H": 1, "C": 0, "N": -1, "O": -2, "S": -2}
            charges = [charge_map.get(s, 0) for s in symbols]
            return cls.from_atoms(positions, charges=charges, species=symbols, gamma=gamma)
        except ImportError:
            pass

        try:
            import MDAnalysis as mda
            u = mda.Universe(str(path))
            positions = u.atoms.positions
            names = u.atoms.names
            charges = getattr(u.atoms, "charges", np.zeros(len(u.atoms)))
            if hasattr(charges, "tolist"):
                charges = charges.tolist()
            return cls.from_atoms(positions, charges=charges, species=list(names), gamma=gamma)
        except ImportError:
            pass

        raise ImportError("Install ase or MDAnalysis to use from_pdb: pip install pyhqiv[ase] or pyhqiv[mda]")

    def compute_fields(
        self,
        grid: np.ndarray,
        t: float = 0.0,
        E_prime: float = 0.5,
        phase_corrected: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute phase-corrected E and B on grid. grid (N, 3).
        Monogamy weighting γ is applied in constitutive relation.
        """
        grid = np.asarray(grid)
        if grid.ndim == 1:
            grid = grid.reshape(1, 3)

        if phase_corrected:
            eps_eff = _phase_corrected_epsilon(grid, self.atoms, self.gamma, E_prime=E_prime)
            E, B = _coulomb_field(grid, self.positions, self.charges, epsilon_eff=1.0 / eps_eff)
        else:
            E, B = _coulomb_field(grid, self.positions, self.charges)

        return E, B

    def total_hamiltonian_qubit_form(self) -> Optional[Any]:
        """
        Return total Hamiltonian in a form suitable for QuTiP (qutip.Qobj).
        Optional: requires qutip. Returns None if not available.
        """
        try:
            import qutip
        except ImportError:
            return None
        # Minimal placeholder: identity (actual HQIV Hamiltonian would use field modes)
        return qutip.qeye(2)
