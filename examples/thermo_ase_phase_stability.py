"""
Integration with ASE: relax a crystal structure then compute its phase stability.

Uses thermo_ase_phase_stability(G = E + P V - T S with HQIV correction).
Requires: pip install pyhqiv[ase]
"""

import numpy as np
from pyhqiv.thermo import thermo_ase_phase_stability

# Example: after ASE relaxation, you have E, V, n_atoms
# E in J, V in m³, P in Pa, T in K
potential_energy_J = -100.0  # e.g. from ASE calculator
volume_m3 = 1e-28  # 100 Å³
P_Pa = 1e5  # 1 bar
T_K = 300.0
n_atoms = 8

G = thermo_ase_phase_stability(
    potential_energy_J,
    volume_m3,
    P_Pa,
    T_K,
    n_atoms,
    gamma=0.40,
)
print(f"Gibbs free energy (HQIV-corrected): G = {G:.6e} J")

# With ASE present, you would do:
# from ase.optimize import BFGS
# from pyhqiv import HQIVCalculator
# atoms.calc = HQIVCalculator(gamma=0.40)
# BFGS(atoms).run()
# E = atoms.get_potential_energy()  # in eV → convert to J
# V = atoms.get_volume() * 1e-30    # Å³ → m³
# G = thermo_ase_phase_stability(E * 1.602e-19, V, 1e5, 300, len(atoms))
