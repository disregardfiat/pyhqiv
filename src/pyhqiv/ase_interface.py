"""
ASE Calculator interface: get_potential_energy(), get_forces(), get_stress().

Enables geometry relaxation with ASE optimizers, e.g.:

    from ase.optimize import BFGS
    from pyhqiv import HQIVCalculator

    calc = HQIVCalculator(gamma=0.40)
    atoms.calc = calc
    BFGS(atoms).run()

Requires the optional dependency: pip install pyhqiv[ase]
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pyhqiv.atom import HQIVAtom
from pyhqiv.constants import C_SI, GAMMA
from pyhqiv.system import HQIVSystem


def hqiv_energy_at_positions(
    system: HQIVSystem,
    positions: np.ndarray,
    energy_scale: float = 1.0,
) -> float:
    """
    HQIV potential energy: (1/2) * sum_i sum_{j≠i} φ_j(x_i) with φ = 2c²/Θ, Θ = r.

    Usable as the HQIV contribution in E_total = E_bond + E_hqiv + ...
    positions: (N, 3) in the same length units as C_SI (e.g. metres) unless energy_scale
    converts; energy_scale multiplies the raw sum (e.g. use 14.4 for eV when r in Å).
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape[0] != len(system.atoms):
        raise ValueError("positions length must match number of atoms")
    total = 0.0
    for i, pos in enumerate(positions):
        grid = pos.reshape(1, 3)
        phi_i = 0.0
        for j, at in enumerate(system.atoms):
            if i == j:
                continue
            at_other = HQIVAtom(
                positions[j],
                charge=at.charge,
                species=getattr(at, "species", "X"),
            )
            phi_i += at_other.phi_local(grid).item()
        total += phi_i
    return float(total) * 0.5 * energy_scale


def hqiv_forces_analytic(
    positions: np.ndarray,
    charges: np.ndarray,
    c_si: float = C_SI,
    gamma: float = GAMMA,
    energy_scale: float = 1.0,
) -> np.ndarray:
    """
    Analytical forces F_i = -dE/dx_i for E = (1/2) * sum φ with φ = 2c²/r.

    E = energy_scale * c² * sum_{i≠j} 1/r_ij  (pair sum). So
    dE/dx_i = energy_scale * c² * sum_{j≠i} (-1) * (x_i - x_j)/r_ij^3,
    hence force (ASE convention) F_i = -dE/dx_i = energy_scale * 2*c² * sum_{j≠i} (x_i - x_j)/r_ij^3.

    positions: (N, 3), charges (N,) unused in this minimal φ∝1/r model (kept for API).
    Returns (N, 3) in same units as energy_scale (e.g. eV/Å when energy_scale in eV·Å and r in Å).
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    forces = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = positions[i] - positions[j]
            r_norm = np.linalg.norm(r)
            r_norm = max(r_norm, 1e-12)
            # F_i += 2 * c² * (x_i - x_j) / r_ij^3  (contribution from pair i,j)
            forces[i] += (2.0 * (c_si**2) * energy_scale) * r / (r_norm**3)
    return forces


def hqiv_stress_virial(
    positions: np.ndarray,
    forces: np.ndarray,
    volume: float,
) -> np.ndarray:
    """
    Stress tensor (3x3) from virial: σ_ab = (1/V) sum_i x_ia F_ib.
    ASE expects stress in the form [xx, yy, zz, yz, xz, xy] (Voigt), and
    convention is stress = -dE/d(strain), so we use -virial/volume.
    """
    positions = np.asarray(positions, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if volume <= 0:
        return np.zeros(6)
    virial = np.einsum("ia,ib->ab", positions, forces)
    stress_3x3 = -virial / volume
    # Voigt order: xx, yy, zz, yz, xz, xy
    return np.array(
        [
            stress_3x3[0, 0],
            stress_3x3[1, 1],
            stress_3x3[2, 2],
            stress_3x3[1, 2],
            stress_3x3[0, 2],
            stress_3x3[0, 1],
        ]
    )


try:
    from ase.calculators.calculator import Calculator as _ASECalculator
    from ase.calculators.calculator import all_changes
except ImportError:
    _ASECalculator = None  # type: ignore
    all_changes = None


def _hqiv_calc_calculate(
    self: Any,
    atoms: Optional[Any] = None,
    properties: Optional[List[str]] = None,
    system_changes: Optional[List[str]] = None,
) -> None:
    if atoms is None:
        atoms = self.atoms
    if properties is None:
        properties = self.implemented_properties
    if system_changes is None and all_changes is not None:
        system_changes = all_changes
    if _ASECalculator is not None:
        _ASECalculator.calculate(self, atoms, properties, system_changes)

    pos = atoms.get_positions()
    system = self._atoms_to_system(atoms)
    charges = np.array([a.charge for a in system.atoms])

    self.results = {}

    if "energy" in properties:
        self.results["energy"] = self.gamma * hqiv_energy_at_positions(
            system, pos, energy_scale=self.energy_scale
        )

    if "forces" in properties:
        # ASE convention: forces = -dE/dx
        self.results["forces"] = self.gamma * hqiv_forces_analytic(
            pos,
            charges,
            gamma=self.gamma,
            energy_scale=self.energy_scale,
        )

    if "stress" in properties:
        if "forces" not in self.results:
            self.results["forces"] = self.gamma * hqiv_forces_analytic(
                pos, charges, gamma=self.gamma, energy_scale=self.energy_scale
            )
        vol = atoms.get_volume()
        self.results["stress"] = hqiv_stress_virial(pos, self.results["forces"], vol)


def _atoms_to_system(self: Any, atoms: Any) -> HQIVSystem:
    pos = atoms.get_positions()
    if self._charges is not None:
        charges = self._charges
    else:
        sym = atoms.get_chemical_symbols()
        charge_map = {"H": 1, "C": 0, "N": -1, "O": -2, "S": -2, "Si": 0, "Ga": 3, "As": -3}
        charges = [charge_map.get(s, 0) for s in sym]
    species = self._species or atoms.get_chemical_symbols()
    return HQIVSystem.from_atoms(pos, charges=charges, species=species, gamma=self.gamma)


if _ASECalculator is not None:

    class HQIVCalculator(_ASECalculator):
        """
        Full ASE Calculator: get_potential_energy(), get_forces(), get_stress().

        Use for geometry relaxation:

            from ase.optimize import BFGS
            calc = HQIVCalculator(gamma=0.40)
            atoms.calc = calc
            BFGS(atoms).run()
        """

        implemented_properties = ["energy", "forces", "stress"]

        def __init__(
            self,
            gamma: float = GAMMA,
            energy_scale: float = 1.0,
            charges: Optional[List[float]] = None,
            species: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.gamma = gamma
            self.energy_scale = energy_scale
            self._charges = charges
            self._species = species

        _atoms_to_system = _atoms_to_system
        calculate = _hqiv_calc_calculate

else:

    class HQIVCalculator:  # type: ignore
        """Placeholder when ASE is not installed. Install with: pip install pyhqiv[ase]."""

        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "ASE is required for HQIVCalculator. Install with: pip install pyhqiv[ase]"
            )
