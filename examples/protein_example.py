"""
Protein-scale HQIV example: load PDB (e.g. hemoglobin), compute fields on 128³ grid.
Requires: pip install pyhqiv[ase] or pyhqiv[mda]
"""

import sys
import time
import numpy as np
from pyhqiv import HQIVSystem

def main():
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdb_path:
        print("Usage: python protein_example.py <path_to.pdb>")
        print("Example: python protein_example.py 1a3n.pdb")
        # Demo without PDB: fake protein-sized system
        n_atoms = 1000
        np.random.seed(42)
        positions = np.random.randn(n_atoms, 3) * 20.0
        charges = np.random.randn(n_atoms) * 0.3
        system = HQIVSystem.from_atoms(positions, charges=charges.tolist(), gamma=0.40)
        print(f"Using synthetic {n_atoms} atoms (no PDB)")
    else:
        system = HQIVSystem.from_pdb(pdb_path, gamma=0.40)
        print(f"Loaded {len(system.atoms)} atoms from {pdb_path}")

    # 128³ grid (or smaller for quick run)
    n = 32  # 32³ for quick demo; use 128 for full run
    grid = np.mgrid[-30:30:n*1j, -30:30:n*1j, -30:30:n*1j].reshape(3, -1).T

    t0 = time.perf_counter()
    E, B = system.compute_fields(grid, t=0.0, phase_corrected=True)
    elapsed = time.perf_counter() - t0
    print(f"Grid: {n}^3 = {grid.shape[0]} points")
    print(f"compute_fields: {elapsed:.3f} s")
    print(f"|E| max: {np.linalg.norm(E, axis=1).max():.6e}")

if __name__ == "__main__":
    main()
