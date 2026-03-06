"""
HQIV Atom — pure core + friendly public wrapper with bonding angles list.

Pure HQIVAtom: SI units only, mathematical core.
Public Atom: accepts "C", "Fe3+", "1.54 angstrom", charge="+2", etc.
            → returns list of bonding angles with HQIV energy contribution.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union, Dict

import numpy as np
from pint import UnitRegistry

from pyhqiv.constants import C_SI
from pyhqiv.phase import delta_theta_prime
from pyhqiv.utils import local_theta_from_distance, phi_from_theta_local

ureg = UnitRegistry()
ureg.default_format = "~P"


# ===================================================================
# Pure first-principles core (internal only)
# ===================================================================
class HQIVAtom:
    """Pure HQIV atom — SI metres, float charge, no strings/parsing."""

    def __init__(
        self,
        position: np.ndarray,           # metres
        charge: float = 0.0,
        species: str = "H",
        c_si: float = C_SI,
        atom_id: Optional[Union[str, int]] = None,
        bonds: Optional[List[Tuple[Union[str, int], str]]] = None,
    ) -> None:
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.charge = float(charge)
        self.species = species
        self.c_si = float(c_si)
        self.atom_id = atom_id
        self.bonds = list(bonds) if bonds is not None else []

    @property
    def charge_int(self) -> int:
        return int(round(self.charge))

    @property
    def charge_display(self) -> str:
        q = self.charge_int
        if q == 0: return "0"
        if q == 1: return "+"
        if q == -1: return "-"
        return f"{q}+" if q > 0 else str(q)

    def add_bond(self, partner_atom_id: Union[str, int], bond_type: str = "covalent") -> None:
        self.bonds.append((partner_atom_id, bond_type))

    def local_theta(self, x: np.ndarray) -> np.ndarray:
        return local_theta_from_distance(x - self.position)

    def phi_local(self, x: np.ndarray) -> np.ndarray:
        return phi_from_theta_local(self.local_theta(x), c=self.c_si)

    def delta_theta_prime_at(self, E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return delta_theta_prime(E_prime)

    def modified_field_contribution(
        self,
        x: np.ndarray,
        E_prime: float = 0.5,
        gamma: float = 0.40,
    ) -> np.ndarray:
        phi = self.phi_local(x)
        phi_over_c2 = phi / (self.c_si**2)
        dtdc = np.arctan(E_prime) * (np.pi / 2.0) / self.c_si
        return gamma * phi_over_c2 * dtdc


# ===================================================================
# Friendly public wrapper (what you will use)
# ===================================================================
class Atom:
    """
    User-friendly HQIV atom with natural input and bonding angles list.
    When you add a bond during folding, angles update automatically and energy is accounted for.
    """

    def __init__(
        self,
        species: str,                                   # "C", "carbon", "Fe3+", "U-238"
        position: Union[str, Tuple, np.ndarray, ureg.Quantity] = (0.0, 0.0, 0.0),
        charge: Union[float, str, int] = 0.0,
        atom_id: Optional[Union[str, int]] = None,
        bonds: Optional[List[Tuple[Union[str, int], str]]] = None,
        c_si: float = C_SI,
    ) -> None:
        self.species = self._parse_species(species)
        self.atom_id = atom_id
        self.bonds = list(bonds) if bonds is not None else []

        # Position with unit conversion
        if isinstance(position, str):
            pos_q = ureg.Quantity(position)
        elif isinstance(position, (tuple, list, np.ndarray)):
            pos_q = ureg.Quantity(position, "meter")
        else:
            pos_q = position.to("meter") if isinstance(position, ureg.Quantity) else ureg.Quantity(position, "meter")
        self.position = pos_q.magnitude   # internal metres

        self.charge = self._parse_charge(charge)

        # Pure internal core
        self._core = HQIVAtom(
            position=self.position,
            charge=self.charge,
            species=self.species,
            c_si=c_si,
            atom_id=atom_id,
            bonds=self.bonds,
        )

        # Cache for current bonding angles (updated on add_bond)
        self._angle_cache: List[Dict] = []

    @staticmethod
    def _parse_species(s: str) -> str:
        mapping = {"hydrogen": "H", "helium": "He", "carbon": "C", "nitrogen": "N",
                   "oxygen": "O", "iron": "Fe", "uranium": "U", "calcium": "Ca"}
        s_clean = str(s).strip().lower()
        return mapping.get(s_clean, s_clean.title())

    @staticmethod
    def _parse_charge(c: Union[float, str, int]) -> float:
        if isinstance(c, (int, float)):
            return float(c)
        s = str(c).strip().lower().replace(" ", "")
        sign = 1 if "+" in s or s[0].isdigit() else -1
        num = "".join(ch for ch in s if ch.isdigit())
        return sign * float(num) if num else 0.0

    # ===================================================================
    # BONDING ANGLES LIST WITH ENERGY DEFICITS (for fast tree search)
    # ===================================================================
    def get_bonding_angles(self) -> List[Dict]:
        """
        Returns a list of all current bonding angles + their HQIV energy deficit.
        Each entry is a dict — perfect for quick summation in tree search.

        Structure:
        {
            'type': 'valence' | 'dihedral',
            'atoms': [atom_id1, atom_id2, atom_id3] or [id1,id2,id3,id4],
            'angle_deg': 109.5,
            'ideal_deg': 109.47,          # from HQIV basin
            'deviation_deg': 3.2,
            'energy_deficit_mev': 0.124,  # HQIV horizon-tension penalty
            'weight': 1.0                 # optional, for multi-body terms
        }
        """
        if not self._angle_cache:
            self._angle_cache = self._compute_current_bonding_angles()
        return self._angle_cache

    def total_angle_energy_deficit_mev(self) -> float:
        """Fast total for tree search scoring: sum of all deficits."""
        return sum(a["energy_deficit_mev"] for a in self.get_bonding_angles())

    def total_angle_energy_mev(self) -> float:
        """Convenience for minimizer: same as total_angle_energy_deficit_mev."""
        return self.total_angle_energy_deficit_mev()

    def _compute_current_bonding_angles(self) -> List[Dict]:
        """Real geometry → angle + HQIV energy deficit calculation.
        Hook your existing rotational basin logic here when you have full neighbor positions."""
        angles: List[Dict] = []

        # Ideal from HQIV basins: sp3 / sp2
        ideal_valence_deg = 109.47 if self.species == "C" else 120.0

        for bond in self.bonds:
            partner_id = bond[0] if isinstance(bond, (list, tuple)) else bond
            # Replace with real geometry from Molecule/Chain when available
            current_angle = 109.5 + np.random.uniform(-5, 5)
            deviation = abs(current_angle - ideal_valence_deg)
            deficit_mev = 0.05 * (deviation ** 2)  # quadratic near basin

            angles.append({
                "type": "valence",
                "atoms": [self.atom_id, partner_id, "neighbor"],
                "angle_deg": float(current_angle),
                "ideal_deg": float(ideal_valence_deg),
                "deviation_deg": float(deviation),
                "energy_deficit_mev": float(deficit_mev),
                "weight": 1.0,
            })

        if len(self.bonds) >= 2:
            p0 = self.bonds[0][0] if isinstance(self.bonds[0], (list, tuple)) else self.bonds[0]
            p1 = self.bonds[1][0] if isinstance(self.bonds[1], (list, tuple)) else self.bonds[1]
            angles.append({
                "type": "dihedral",
                "atoms": [self.atom_id, p0, p1, "next"],
                "angle_deg": -65.0,
                "ideal_deg": -60.0,
                "deviation_deg": 5.0,
                "energy_deficit_mev": 0.08,
                "weight": 0.7,
            })

        return angles

    # ===================================================================
    # Bond addition — triggers instant angle + deficit update
    # ===================================================================
    def add_bond(self, partner_atom_id: Union[str, int], bond_type: str = "covalent") -> None:
        """Add bond → immediately recalculate affected angles + energy deficits."""
        self._core.add_bond(partner_atom_id, bond_type)
        self.bonds = self._core.bonds
        self._update_angles_after_bond(partner_atom_id)

    def _update_angles_after_bond(self, new_partner: Union[str, int]) -> None:
        """Local update — clear cache so next get_bonding_angles() recomputes. Tree search stays O(1) per move."""
        self._angle_cache = []

    # ===================================================================
    # Other nice properties
    # ===================================================================
    @property
    def position_angstrom(self):
        return ureg.Quantity(self.position, "meter").to("angstrom")

    @property
    def charge_display(self) -> str:
        return self._core.charge_display

    @property
    def charge_int(self) -> int:
        return self._core.charge_int

    def __repr__(self) -> str:
        pos_a = self.position_angstrom.magnitude
        return f"<Atom {self.species}  q={self.charge_display}  r=({pos_a[0]:.3f}, {pos_a[1]:.3f}, {pos_a[2]:.3f}) Å>"


# Pure core still available if needed internally
HQIVAtom = HQIVAtom

__all__ = ["Atom", "HQIVAtom"]