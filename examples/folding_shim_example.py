"""
Thin shim: HQIVSystem → folding minimizer.

Shows how to use HQIV fields/φ as an energy term for structure relaxation.
Use with ASE (pip install pyhqiv[ase]) or any minimizer that accepts positions
and an energy/force callback.

Example: python folding_shim_example.py [path_to.pdb]
  Without PDB: runs on a small synthetic chain and optionally relaxes with ASE.
  With PDB: loads structure and reports HQIV energy term (no minimization by default).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pyhqiv import HQIVSystem
from pyhqiv.atom import HQIVAtom


def hqiv_energy_at_positions(system: HQIVSystem, positions: np.ndarray) -> float:
    """
    Sum of φ/2 (potential) at each atom position from the rest of the system.
    Usable as an extra term in a folding Hamiltonian: E_total = E_bond + E_hqiv + ...
    """
    positions = np.asarray(positions)
    if positions.shape[0] != len(system.atoms):
        raise ValueError("positions length must match number of atoms")
    total = 0.0
    for i, pos in enumerate(positions):
        # φ at atom i due to all others (evaluate on a 1-point grid)
        grid = pos.reshape(1, 3)
        phi_i = 0.0
        for j, at in enumerate(system.atoms):
            if i == j:
                continue
            at_other = HQIVAtom(positions[j], charge=at.charge, species=getattr(at, "species", "X"))
            phi_i += at_other.phi_local(grid).item()
        total += phi_i
    return float(total) * 0.5  # scale as needed for your units


def hqiv_forces_numerical(
    system: HQIVSystem, positions: np.ndarray, delta: float = 1e-5
) -> np.ndarray:
    """
    Numerical gradient of hqiv_energy_at_positions w.r.t. positions.
    Returns (N, 3) force = -dE/dx (minimizers typically minimize E, so gradient descent uses -grad).
    """
    positions = np.asarray(positions, dtype=float)
    n, _ = positions.shape
    forces = np.zeros((n, 3))
    E0 = hqiv_energy_at_positions(system, positions)
    for i in range(n):
        for d in range(3):
            pos_plus = positions.copy()
            pos_plus[i, d] += delta
            E_plus = hqiv_energy_at_positions(system, pos_plus)
            forces[i, d] = -(E_plus - E0) / delta
    return forces


def run_synthetic_chain(use_ase: bool = False) -> None:
    """Small synthetic chain; optionally relax with ASE BFGS."""
    np.random.seed(42)
    n_atoms = 8
    positions = np.cumsum(np.random.randn(n_atoms, 3) * 1.5, axis=0)
    charges = [0.0] * n_atoms
    system = HQIVSystem.from_atoms(positions, charges=charges, gamma=0.40)

    E = hqiv_energy_at_positions(system, positions)
    print(f"HQIV energy term (synthetic {n_atoms} atoms): {E:.6e}")

    if use_ase:
        try:
            import ase
            from ase.optimize import BFGS
            from ase.calculators.calculator import Calculator, all_changes

            class HQIVCalculator(Calculator):
                implemented_properties = ["energy", "forces"]

                def __init__(self, charges: list, gamma: float = 0.40, **kwargs):
                    super().__init__(**kwargs)
                    self.charges = charges
                    self.gamma = gamma

                def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
                    super().calculate(atoms, properties, system_changes)
                    pos = atoms.get_positions()
                    system = HQIVSystem.from_atoms(pos, charges=self.charges, gamma=self.gamma)
                    self.results = {
                        "energy": hqiv_energy_at_positions(system, pos),
                        "forces": -hqiv_forces_numerical(system, pos),
                    }

            ase_atoms = ase.Atoms(positions=positions, symbols=["C"] * n_atoms)
            ase_atoms.calc = HQIVCalculator(charges=charges, gamma=0.40)
            opt = BFGS(ase_atoms, trajectory="folding_shim_traj.xyz")
            opt.run(fmax=0.05, steps=20)
            print("ASE BFGS relaxation finished. See folding_shim_traj.xyz")
        except ImportError:
            print("ASE not installed; pip install pyhqiv[ase] for minimizer demo.")


def main() -> None:
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else None
    use_ase = "--ase" in sys.argv

    if pdb_path:
        system = HQIVSystem.from_pdb(pdb_path, gamma=0.40)
        positions = system.positions
        E = hqiv_energy_at_positions(system, positions)
        print(f"Loaded {len(system.atoms)} atoms from {pdb_path}")
        print(f"HQIV energy term: {E:.6e}")
        return

    run_synthetic_chain(use_ase=use_ase)


if __name__ == "__main__":
    main()
