"""
Geometry relaxation with HQIV potential using the full ASE Calculator.

Usage (requires pip install pyhqiv[ase]):

    python relax_with_hqiv.py              # 2-atom Si2
    python relax_with_hqiv.py --dimer 4    # 4-atom chain
    python relax_with_hqiv.py --fmax 0.01 --steps 50

Then:
    from ase.optimize import BFGS
    from ase import Atoms
    from pyhqiv import HQIVCalculator

    atoms = Atoms(...)
    calc = HQIVCalculator(gamma=0.40)
    atoms.calc = calc
    BFGS(atoms).run(fmax=0.05)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Relax structure with HQIV ASE calculator")
    ap.add_argument("--dimer", type=int, default=2, help="Number of atoms (dimer/chain)")
    ap.add_argument("--fmax", type=float, default=0.05, help="Force convergence (eV/Å)")
    ap.add_argument("--steps", type=int, default=30, help="Max BFGS steps")
    ap.add_argument("--traj", type=str, default="hqiv_relax.xyz", help="Trajectory file")
    ap.add_argument("--gamma", type=float, default=0.40, help="HQIV gamma")
    args = ap.parse_args()

    try:
        from ase import Atoms
        from ase.optimize import BFGS
        from pyhqiv import HQIVCalculator
    except ImportError as e:
        print("Install ASE and pyhqiv with: pip install pyhqiv[ase]")
        raise SystemExit(1) from e

    n = args.dimer
    # Simple chain along x
    positions = np.zeros((n, 3))
    positions[:, 0] = np.linspace(0, (n - 1) * 2.5, n)
    symbols = ["Si"] * n
    atoms = Atoms(symbols=symbols, positions=positions)
    cell = 10.0 * np.eye(3)
    atoms.set_cell(cell)
    atoms.pbc = False

    calc = HQIVCalculator(gamma=args.gamma)
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    print(f"Initial energy: {e0:.6e}")
    print(f"Initial forces (max abs): {np.abs(atoms.get_forces()).max():.6e}")

    opt = BFGS(atoms, trajectory=args.traj)
    opt.run(fmax=args.fmax, steps=args.steps)

    e1 = atoms.get_potential_energy()
    print(f"Final energy:   {e1:.6e}")
    print(f"Final forces (max abs): {np.abs(atoms.get_forces()).max():.6e}")
    print(f"Trajectory written to {args.traj}")


if __name__ == "__main__":
    main()
