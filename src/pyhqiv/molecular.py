"""
HQIV Molecule / Protein container.

Pure core (HQIVMolecule): SI units only, list of HQIVAtom, bond graph.
Public wrapper (Molecule): natural language, unit conversion, rigid groups,
                          surface EM field, fast angle-deficit list for tree search.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Callable, Union

import numpy as np
from pint import UnitRegistry

from pyhqiv.atom import Atom, HQIVAtom   # the clean Atom we just built
from pyhqiv.constants import C_SI
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants  # for T_CMB if needed

ureg = UnitRegistry()
ureg.default_format = "~P"


# ===================================================================
# Pure first-principles core
# ===================================================================
class HQIVMolecule:
    """Pure HQIV molecule — SI metres only. No strings, no units."""

    def __init__(self, atoms: List[HQIVAtom], t_cmb: float = 2.725):
        self.atoms = atoms
        self.t_cmb = float(t_cmb)
        # Bond graph: atom_id → set of (partner_id, bond_type)
        self.bond_graph: Dict[Union[str, int], Set[Tuple[Union[str, int], str]]] = defaultdict(set)
        for atom in atoms:
            for partner_id, btype in atom.bonds:
                self.bond_graph[atom.atom_id].add((partner_id, btype))
                self.bond_graph[partner_id].add((atom.atom_id, btype))  # undirected

    def add_bond(self, id1: Union[str, int], id2: Union[str, int], bond_type: str = "covalent"):
        self.bond_graph[id1].add((id2, bond_type))
        self.bond_graph[id2].add((id1, bond_type))
        # Update the underlying atoms too
        for atom in self.atoms:
            if atom.atom_id == id1:
                atom.add_bond(id2, bond_type)
            if atom.atom_id == id2:
                atom.add_bond(id1, bond_type)

    def break_bond(self, id1: Union[str, int], id2: Union[str, int]):
        self.bond_graph[id1].discard((id2, "covalent"))
        self.bond_graph[id1].discard((id2, "peptide"))
        # ... add other types if needed
        self.bond_graph[id2].discard((id1, "covalent"))
        self.bond_graph[id2].discard((id1, "peptide"))
        # Update atoms
        for atom in self.atoms:
            if atom.atom_id == id1:
                atom.bonds = [b for b in atom.bonds if b[0] != id2]
            if atom.atom_id == id2:
                atom.bonds = [b for b in atom.bonds if b[0] != id1]


# ===================================================================
# Friendly public Molecule (what you will use in protein_folder)
# ===================================================================
class Molecule:
    """
    User-friendly HQIV molecule / protein.
    Build, modify, get rigid groups, surface EM field, angle deficits for tree search.
    """

    def __init__(
        self,
        atoms: Optional[List[Atom]] = None,
        t_cmb: float = 2.725,
    ):
        self.atoms: List[Atom] = atoms or []
        self.t_cmb = float(t_cmb)
        self._core = HQIVMolecule([a._core for a in self.atoms], t_cmb=t_cmb)

        # Cache for rigid groups and angle deficits (fast tree search)
        self._rigid_cache: Optional[List[Dict]] = None
        self._angle_deficit_cache: Optional[List[Dict]] = None

    # ===================================================================
    # High-level editing
    # ===================================================================
    def add_atom(self, atom: Atom):
        self.atoms.append(atom)
        self._core = HQIVMolecule([a._core for a in self.atoms], self.t_cmb)  # rebuild core
        self._clear_caches()

    def make_bond(self, atom1_id: Union[str, int], atom2_id: Union[str, int], bond_type: str = "covalent"):
        self._core.add_bond(atom1_id, atom2_id, bond_type)
        self._clear_caches()   # angles & rigid groups may change

    def break_bond(self, atom1_id: Union[str, int], atom2_id: Union[str, int]):
        self._core.break_bond(atom1_id, atom2_id)
        self._clear_caches()

    def _clear_caches(self):
        self._rigid_cache = None
        self._angle_deficit_cache = None

    # ===================================================================
    # BONDING ANGLES + ENERGY DEFICITS (fast for tree search)
    # ===================================================================
    def get_bonding_angles(self) -> List[Dict]:
        """List of every valence/dihedral angle with HQIV energy deficit (MeV)."""
        if self._angle_deficit_cache is None:
            self._angle_deficit_cache = []
            for atom in self.atoms:
                for ang in atom.get_bonding_angles():
                    self._angle_deficit_cache.append(ang)
        return self._angle_deficit_cache

    def total_angle_energy_deficit_mev(self) -> float:
        """Single number for quick tree-search scoring."""
        return sum(a['energy_deficit_mev'] for a in self.get_bonding_angles())

    # ===================================================================
    # RIGID GROUPS (the big speed-up you asked for)
    # ===================================================================
    def get_rigid_groups(self) -> List[Dict]:
        """
        Returns list of rigid groups.
        Each group has:
            'atoms': list of atom_ids
            'type': 'helix', 'sheet', 'ring', 'loop', 'single'
            'break_energy_mev': cost to break this rigid unit (HQIV horizon tension)
        """
        if self._rigid_cache is None:
            self._rigid_cache = self._detect_rigid_groups()
        return self._rigid_cache

    def _detect_rigid_groups(self) -> List[Dict]:
        """Simple but effective rigidity detection based on bond types and angles."""
        groups = []
        visited = set()

        for atom in self.atoms:
            if atom.atom_id in visited:
                continue

            # Start a new group
            group_atoms = [atom.atom_id]
            visited.add(atom.atom_id)

            # Flood-fill connected rigid components (helices, sheets, rings)
            stack = [atom]
            while stack:
                current = stack.pop()
                for partner_id, btype in current.bonds:
                    if partner_id not in visited and btype in ("covalent", "peptide"):
                        visited.add(partner_id)
                        group_atoms.append(partner_id)
                        # Find the partner atom object
                        partner = next((a for a in self.atoms if a.atom_id == partner_id), None)
                        if partner:
                            stack.append(partner)

            # Classify group
            group_type = "single"
            if len(group_atoms) >= 6:
                # Very crude but effective for now: helices have ~3.6 res/turn
                group_type = "helix" if len(group_atoms) % 4 == 0 else "sheet" if len(group_atoms) > 8 else "ring"

            # Break energy = sum of interface horizon tension (simple approximation)
            break_energy = 0.15 * len(group_atoms)  # tunable HQIV-style cost

            groups.append({
                'atoms': group_atoms,
                'type': group_type,
                'break_energy_mev': float(break_energy),
                'size': len(group_atoms)
            })

        return groups

    # ===================================================================
    # SURFACE EM FIELD (for finding new bonding sites)
    # ===================================================================
    def get_surface_em_field(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a callable that gives the total HQIV EM field at any point x (metres).
        Perfect for scanning possible bonding sites or docking.
        """
        def field_at(x: np.ndarray) -> np.ndarray:
            total = np.zeros_like(x, dtype=float)
            for atom in self.atoms:
                total += atom._core.modified_field_contribution(x)
            return total
        return field_at

    # ===================================================================
    # Nice utilities
    # ===================================================================
    def __len__(self):
        return len(self.atoms)

    def __repr__(self):
        return f"<Molecule {len(self.atoms)} atoms, {len(self._core.bond_graph)} bonds>"

    def to_pdb(self, filename: str = "molecule.pdb"):
        """Quick PDB export (you can expand this later)."""
        with open(filename, "w") as f:
            f.write("REMARK   0 HQIV Molecule generated by pyhqiv\n")
            for i, atom in enumerate(self.atoms, 1):
                pos = atom.position_angstrom.magnitude
                f.write(f"ATOM  {i:5d}  CA  {atom.species:3s} A   1    "
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C  \n")


# ===================================================================
# Paper-faithful integral: E_tot = ∫ (ρ c² + ħc/Δx) d³x
# ===================================================================


def _local_density_from_positions(positions: np.ndarray, scale_ang: float = 1.0) -> float:
    """Rough local density (relative): number density from bounding box. scale_ang = 1 Å."""
    positions = np.asarray(positions)
    if positions.size == 0:
        return 1.0
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    n = positions.shape[0]
    lo = np.min(positions, axis=0)
    hi = np.max(positions, axis=0)
    vol_ang3 = np.prod(np.maximum(hi - lo, scale_ang))
    return max(n / vol_ang3, 1e-6)


def hqiv_energy_for_angles(
    phi: float,
    psi: float,
    theta_local_ang: Optional[float] = None,
    temperature: float = 300.0,
    atoms: Optional[List] = None,
    positions: Optional[np.ndarray] = None,
    n_grid: int = 16,
) -> float:
    """
    HQIV energy for a dihedral (φ, ψ) from the paper axiom integral.

    When atoms and positions are given, composes 8×8 field from atoms,
    gets effective_theta_local from the composite, and integrates
    E_tot = ∫ (ρ c² + ħc/Δx) d³x over a small volume around the dihedral.

    When atoms is None, falls back to scalar theta_local_ang (backward compat).
    """
    if atoms is not None and len(atoms) > 0:
        from pyhqiv.energy_field import HQIVEnergyField
        const = get_hqiv_nuclear_constants()
        lattice_base_m = const["LATTICE_BASE_M"]
        pos = np.asarray(positions) if positions is not None else None
        if pos is None or pos.size == 0:
            pos = np.array([getattr(a, "position", np.zeros(3)) for a in atoms], dtype=float)
            if hasattr(atoms[0], "_core"):
                pos = np.array([a._core.position for a in atoms], dtype=float)
        pos = np.asarray(pos, dtype=float)
        pos_ang = pos * 1e10 if np.max(np.abs(pos)) < 1e-3 else pos  # assume m or Å
        local_density = _local_density_from_positions(pos_ang)
        field = HQIVEnergyField.from_atoms(atoms, positions=pos)
        theta_eff_m = field.effective_theta_local(lattice_base_m, local_density)
        theta_eff_ang = theta_eff_m * 1e10
        # Numerical quadrature over small grid (paper's discrete null lattice style)
        grid = np.linspace(-0.5, 0.5, n_grid)
        dV = (theta_eff_ang / n_grid) ** 3 * 1e-30  # m³ per cell (rough)
        mass_density = 1e3  # kg/m³ placeholder (protein ~1 g/cm³)
        E_integrated = 0.0
        for dx in grid:
            local_delta_x = theta_eff_m * (1.0 + 0.01 * np.abs(dx))
            E_integrated += field.total_energy_density(mass_density, local_delta_x) * dV
        return float(E_integrated) / 1.602176634e-19  # J → eV
    # Scalar fallback: no matrix path, return ħc/Θ in eV (backward compat)
    theta_ang = theta_local_ang if theta_local_ang is not None else 1.53
    from pyhqiv.constants import HBAR_SI, C_SI
    theta_m = theta_ang * 1e-10
    hbar_c = HBAR_SI * C_SI
    return float(hbar_c / max(theta_m, 1e-30)) / 1.602176634e-19  # J → eV


# Convenience
__all__ = ["Molecule", "HQIVMolecule", "hqiv_energy_for_angles"]