"""
HQIVUniversalSystem: any-scale network effects from single axiom.

Quarks → nucleons → nuclei → neutron stars → galaxies.
Computationally O(N²) for graph (expensive but exact).
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from pyhqiv.horizon_network import HorizonNetwork, relax_nucleon_positions
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.subatomic import quark_nodes_for_nucleon


def _node_from_particle(p: Union[dict, tuple], algebra) -> tuple:
    """(position, state_matrix, mass_mev) from particle dict or tuple."""
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        return (np.asarray(p[0]), np.asarray(p[1]).reshape(8, 8), float(p[2]))
    pos = np.asarray(p.get("position", np.zeros(3)), dtype=float)
    mat = np.asarray(p.get("state_matrix", np.eye(8)), dtype=float).reshape(8, 8)
    mass = float(p.get("mass_mev", p.get("mass", 0.0)))
    return (pos, mat, mass)


class HQIVUniversalSystem:
    """
    Any-scale binding from HorizonNetwork + DiscreteNullLattice.

    Particles: list of dicts with position, state_matrix, mass_mev, and optional type
    ("proton", "neutron", or omitted). When expand_to_quarks=True, each nucleon becomes
    3 quark nodes so the network runs at quark level (e.g. 6 nodes for deuteron).
    """

    def __init__(
        self,
        particles: List[Union[dict, tuple]],
        lattice_base_m: float = None,
        expand_to_quarks: bool = False,
        algebra=None,
    ) -> None:
        if algebra is None:
            from pyhqiv.algebra import OctonionHQIVAlgebra
            algebra = OctonionHQIVAlgebra(verbose=False)
        self.algebra = algebra
        self._particles_raw = list(particles)
        self.expand_to_quarks = expand_to_quarks
        const = get_hqiv_nuclear_constants() if lattice_base_m is None else {}
        self._lattice_base_m = float(lattice_base_m if lattice_base_m is not None else const.get("LATTICE_BASE_M", 1.94e-15))
        self.lattice = DiscreteNullLattice()
        node_list = self._prepare_nodes()
        self._n_particles = len(self._particles_raw)
        self.net = HorizonNetwork(node_list, self._lattice_base_m, algebra=self.algebra)

    def _prepare_nodes(self) -> List[tuple]:
        """List of (position, state_matrix, mass_mev) for HorizonNetwork."""
        nodes: List[tuple] = []
        for p in self._particles_raw:
            pt = p if isinstance(p, dict) else {"position": p[0], "state_matrix": p[1], "mass_mev": p[2]}
            ptype = pt.get("type", "")
            if self.expand_to_quarks and ptype in ("proton", "neutron"):
                center = np.asarray(pt.get("position", np.zeros(3)), dtype=float).reshape(3)
                is_p = ptype == "proton"
                nodes.extend(quark_nodes_for_nucleon(is_p, center, algebra=self.algebra))
            else:
                nodes.append(_node_from_particle(pt, self.algebra))
        return nodes

    def total_energy_mev(self) -> float:
        """Total system energy (MeV). Lattice δE correction already in net.total_energy()."""
        return float(self.net.total_energy())

    def binding_per_particle(self) -> float:
        """(E_free - E_bound) / N where E_free = sum of rest masses of original particles."""
        E_free = 0.0
        for p in self._particles_raw:
            pt = p if isinstance(p, dict) else None
            if pt is not None:
                E_free += float(pt.get("mass_mev", pt.get("mass", 0.0)))
            else:
                E_free += float(p[2])
        E_bound = self.total_energy_mev()
        n = max(1, self._n_particles)
        return float(E_free - E_bound) / n

    def _effective_m(self) -> int:
        """Effective shell index from total energy (for lattice scaling)."""
        return max(1, int(self.total_energy_mev() / 1000.0))

    def relax(self, steps: int = 100, seed: int = 42) -> None:
        """Re-run nucleon relaxation and rebuild network. Only for expand_to_quarks=False."""
        if self.expand_to_quarks:
            raise NotImplementedError("relax() with expand_to_quarks=True not implemented")
        radii = np.array([self.net._radii_m[i] for i in range(len(self.net.nodes))])
        is_proton = []
        for p in self._particles_raw:
            t = (p.get("type", "") if isinstance(p, dict) else "") or "neutron"
            is_proton.append(t == "proton")
        positions = relax_nucleon_positions(radii, is_proton, n_steps=steps, seed=seed)
        nodes = [
            (positions[i], self.net.nodes[i][1], self.net.nodes[i][2])
            for i in range(len(self.net.nodes))
        ]
        self.net = HorizonNetwork(nodes, self._lattice_base_m, algebra=self.algebra)


__all__ = ["HQIVUniversalSystem"]
