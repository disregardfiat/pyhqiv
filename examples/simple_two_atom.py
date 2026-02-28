"""
Simple two-atom HQIV example: build system, compute phase-corrected E/B on a grid.
"""

import numpy as np
from pyhqiv import HQIVSystem, DiscreteNullLattice

def main():
    # Paper values
    lattice = DiscreteNullLattice(m_trans=500, gamma=0.40)
    result = lattice.evolve_to_cmb(T0_K=2.725)
    print("Lattice evolve_to_cmb:")
    print(f"  Omega_true_k = {result['Omega_true_k']:.6f}")
    print(f"  age_wall_Gyr = {result['age_wall_Gyr']:.1f}")
    print(f"  lapse_compression = {result['lapse_compression']:.2f}")

    # Two-atom system
    sys = HQIVSystem.from_atoms(
        [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        charges=[1.0, -1.0],
        gamma=0.40,
    )
    grid = np.mgrid[-2:2:11j, -2:2:11j, -2:2:11j].reshape(3, -1).T
    E, B = sys.compute_fields(grid, t=0.0, phase_corrected=True)
    print("\nTwo-atom system:")
    print(f"  Grid points: {grid.shape[0]}")
    print(f"  E shape: {E.shape}, B shape: {B.shape}")
    print(f"  |E| at origin (first point): {np.linalg.norm(E[0]):.6e}")

if __name__ == "__main__":
    main()
