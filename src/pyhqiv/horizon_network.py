"""
Horizon network: overlap graph of causal diamonds, Θ_eff from sphere-touching geometry.

When objects (nucleons, residues) have overlapping horizons, they share lattice modes
(horizon monogamy). The connected component's radii give the Pythagorean mode multiplier
μ = Σr_i / sqrt(Σr_i²) ≥ 1 → Θ_eff = L×8×μ → ħc/Δx drops → E_bound < E_free → positive B.

Paper: single axiom E_tot = m c² + ħc/Δx, Δx ≤ Θ_local; Θ_local from geometry only (no Δ in algebra).
"""

from __future__ import annotations

from collections import deque
import math
from typing import List, Optional, Set, Tuple, Union

import numpy as np

from pyhqiv.constants import HBAR_C_MEV_FM
from pyhqiv.subatomic import _sphere_touching_mu

# ħc in MeV·m (radius from mass: r = ħc / (m c²) in length units)
_HBAR_C_MEV_M: float = HBAR_C_MEV_FM * 1e-15

# Coulomb scale for p-p repulsion (hypercharge/geometry); scaled by r_avg² so same order as radii
_COULOMB_RELAX_SCALE: float = 0.001


def relax_nucleon_positions(
    radii_m: np.ndarray,
    is_proton: List[bool],
    n_steps: int = 80,
    seed: int = 42,
) -> np.ndarray:
    """
    Geometric nucleon packing: force-based relaxation to a stable 3D configuration.

    Three forces only (from radii / charge, no new constants):
    - Hard-sphere repulsion if distance < r_i + r_j
    - Soft attraction if distance > r_i + r_j (encourages touching → binding)
    - Coulomb repulsion between protons only ∝ (r_avg/dist)² so scale matches radii

    Returns (A, 3) positions; overlap graph and μ then computed as before.
    Deterministic for given seed. He-4 → tetrahedron; ⁸Be → two alphas; etc.
    """
    radii_m = np.asarray(radii_m, dtype=float)
    A = len(radii_m)
    if A == 0:
        return np.zeros((0, 3))
    r_avg = float(np.mean(radii_m))
    rng = np.random.default_rng(seed)
    r_nuc = (A ** (1.0 / 3.0)) * r_avg * 1.2
    pos = rng.normal(0, r_nuc / 3.0, (A, 3)).astype(float)

    for _ in range(n_steps):
        for i in range(A):
            for j in range(i + 1, A):
                delta = pos[i] - pos[j]
                dist = float(np.linalg.norm(delta)) + 1e-30
                rsum = radii_m[i] + radii_m[j]
                unit = delta / dist

                if dist < rsum:
                    # Hard-sphere repulsion
                    force = (rsum - dist) * unit
                    pos[i] += 0.3 * force
                    pos[j] -= 0.3 * force
                elif dist > rsum * 1.05:
                    # Soft attraction (encourage touching)
                    force = (dist - rsum * 1.05) * unit
                    pos[i] -= 0.05 * force
                    pos[j] += 0.05 * force

                # Coulomb repulsion (protons only); scale (r_avg/dist)² so same order as radii
                if is_proton[i] and is_proton[j]:
                    coul_mag = _COULOMB_RELAX_SCALE * (r_avg / dist) ** 2 * r_avg
                    coul = coul_mag * unit
                    pos[i] += coul
                    pos[j] -= coul

    pos -= np.mean(pos, axis=0)
    return pos


def relax_quark_positions(
    radii_m: np.ndarray,
    charges: np.ndarray,
    n_steps: int = 80,
    seed: int = 42,
) -> np.ndarray:
    """
    Three quark spheres with fractional charge (Q_u = +2/3, Q_d = -1/3).

    Same three forces as nucleon layer: hard-sphere repulsion, soft attraction
    to touching, and Coulomb between all pairs ∝ Q_i Q_j / d² (repulsion for
    like signs, attraction for opposite). No new constants; scale matches radii.
    Returns (3, 3) positions in m; d_ij → r_i + r_j at equilibrium.
    """
    radii_m = np.asarray(radii_m, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = len(radii_m)
    if n != 3 or len(charges) != 3:
        raise ValueError("relax_quark_positions expects 3 radii and 3 charges")
    r_avg = float(np.mean(radii_m))
    rng = np.random.default_rng(seed)
    # Initial blob ~ single nucleon scale
    r_nuc = (3 ** (1.0 / 3.0)) * r_avg * 1.2
    pos = rng.normal(0, r_nuc / 3.0, (3, 3)).astype(float)

    for _ in range(n_steps):
        for i in range(3):
            for j in range(i + 1, 3):
                delta = pos[i] - pos[j]
                dist = float(np.linalg.norm(delta)) + 1e-30
                rsum = radii_m[i] + radii_m[j]
                unit = delta / dist

                if dist < rsum:
                    force = (rsum - dist) * unit
                    pos[i] += 0.3 * force
                    pos[j] -= 0.3 * force
                elif dist > rsum * 1.05:
                    force = (dist - rsum * 1.05) * unit
                    pos[i] -= 0.05 * force
                    pos[j] += 0.05 * force

                # Coulomb: F ∝ Q_i Q_j / d²; like charges repel, opposite attract
                qq = charges[i] * charges[j]
                coul_mag = _COULOMB_RELAX_SCALE * qq * (r_avg / dist) ** 2 * r_avg
                coul = coul_mag * unit  # repulsion when qq > 0 (pos[i] += coul pushes i away from j)
                pos[i] += coul
                pos[j] -= coul

    pos -= np.mean(pos, axis=0)
    return pos


def _default_algebra():
    from pyhqiv.algebra import OctonionHQIVAlgebra
    return OctonionHQIVAlgebra(verbose=False)


def _object_to_node(
    obj: Union[Tuple, dict],
    algebra,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """(position, state_matrix, mass_mev) from object or tuple."""
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        pos, mat, mass = obj[0], obj[1], obj[2]
    elif isinstance(obj, dict):
        pos = obj["position"]
        mat = obj.get("state_matrix", np.eye(8))
        mass = obj.get("mass_mev", obj.get("mass", 0.0))
    else:
        pos = np.asarray(getattr(obj, "position", np.zeros(3)), dtype=float)
        mat = getattr(obj, "state_matrix", None) or getattr(obj, "species_matrix", np.eye(8))
        mass = float(getattr(obj, "mass_mev", getattr(obj, "mass", 0.0)))
    mat = np.asarray(mat, dtype=float).reshape(8, 8)
    return (np.asarray(pos, dtype=float).reshape(3), mat, float(mass))


class HorizonNetwork:
    """
    Overlap graph of causal diamonds; Θ_eff from sphere-touching geometry on the component.

    Nodes: (position, 8×8 state_matrix, mass_mev). Radius per node r_i = ħc/(m_i c²).
    Edges: distance < r_i + r_j (spheres touch). Θ_eff = lattice_base_m * 8 * μ with
    μ = Σr_i / sqrt(Σr_i²) over the component. Binding appears from μ > 1 when N > 1.
    """

    def __init__(
        self,
        objects: List[Union[Tuple, dict, "object"]],
        lattice_base_m: float,
        algebra=None,
    ) -> None:
        self.lattice_base_m = float(lattice_base_m)
        self.algebra = algebra or _default_algebra()
        self.nodes: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for obj in objects:
            self.nodes.append(_object_to_node(obj, self.algebra))

        self._radii_m: List[float] = []
        self.graph: List[List[int]] = [[] for _ in range(len(self.nodes))]
        self._build_overlap_graph()

    def _build_overlap_graph(self) -> None:
        """Edges where distance(i,j) < r_i + r_j (sphere-touching). Radii from mass: r = ħc/(m c²)."""
        n = len(self.nodes)
        self._radii_m = [
            _HBAR_C_MEV_M / max(self.nodes[i][2], 1e-30)
            for i in range(n)
        ]
        for i in range(n):
            self.graph[i] = []
            pos_i = self.nodes[i][0]
            for j in range(n):
                if i == j:
                    continue
                pos_j = self.nodes[j][0]
                d = float(np.linalg.norm(pos_i - pos_j))
                if d < self._radii_m[i] + self._radii_m[j]:
                    self.graph[i].append(j)

    def _connected_component_containing(self, position: np.ndarray) -> Set[int]:
        """BFS: indices of nodes in the same connected component as the node at/near position."""
        pos = np.asarray(position, dtype=float).reshape(3)
        if not self.nodes:
            return set()
        # Node closest to position
        start = int(np.argmin([np.linalg.norm(self.nodes[i][0] - pos) for i in range(len(self.nodes))]))
        return self._connected_component_from_index(start)

    def _connected_component_from_index(self, start: int) -> Set[int]:
        """BFS: indices of nodes in the same connected component as `start`."""
        if not self.nodes:
            return set()
        seen: Set[int] = {start}
        q: deque = deque([start])
        while q:
            i = q.popleft()
            for j in self.graph[i]:
                if j not in seen:
                    seen.add(j)
                    q.append(j)
        return seen

    def _mu_for_indices(self, indices: Set[int]) -> float:
        """Sphere-touching mode multiplier on a specific index set."""
        if len(indices) <= 1:
            return 1.0
        radii = np.array([self._radii_m[i] for i in sorted(indices)], dtype=float)
        return _sphere_touching_mu(radii)

    def effective_theta_local(self, position: np.ndarray) -> float:
        """
        Θ_local from sphere-touching geometry on the full connected component.

        μ = Σr_i / sqrt(Σr_i²) over component; Θ_eff = lattice_base_m * 8 * μ.
        Single node → μ = 1 → Θ = L×8. Cluster (e.g. 4 nucleons) → μ > 1 → binding.
        """
        component = self._connected_component_containing(position)
        if not component or len(component) <= 1:
            return self.lattice_base_m * 8.0
        mu = self._mu_for_indices(component)
        return self.lattice_base_m * 8.0 * max(mu, 1e-30)

    def effective_theta_for_index(self, index: int) -> float:
        """
        Node-local horizon from the same overlap graph.

        Each nucleon inherits both the coherence of the full connected component and
        the overlap valence of its immediate neighbourhood. The geometric mean avoids
        double-counting while preserving the single-graph construction at every scale.
        """
        if index < 0 or index >= len(self.nodes):
            raise IndexError("node index out of range")
        component = self._connected_component_from_index(index)
        component_mu = self._mu_for_indices(component)
        local_indices: Set[int] = {index, *self.graph[index]}
        local_mu = self._mu_for_indices(local_indices)
        return self.lattice_base_m * 8.0 * math.sqrt(max(component_mu * local_mu, 1e-30))

    def effective_theta_array(self) -> np.ndarray:
        """Per-node local horizons derived from the same overlap graph."""
        if not self.nodes:
            return np.array([])
        return np.array([self.effective_theta_for_index(i) for i in range(len(self.nodes))], dtype=float)

    def total_energy(self) -> float:
        """Paper axiom: E = Σ (mass_mev + ħc/Δx), Δx = Θ_local."""
        E = 0.0
        for i, (pos, _, mass_mev) in enumerate(self.nodes):
            theta = self.effective_theta_local(pos)
            E += mass_mev + _HBAR_C_MEV_M / max(theta, 1e-30)
        return float(E)


__all__ = ["HorizonNetwork", "relax_nucleon_positions", "relax_quark_positions"]
