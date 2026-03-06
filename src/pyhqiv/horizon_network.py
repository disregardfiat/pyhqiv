"""
Horizon network: overlap graph of causal diamonds, Θ_eff from sphere-touching geometry.

When objects (nucleons, residues) have overlapping horizons, they share lattice modes
(horizon monogamy). The connected component's radii give the Pythagorean mode multiplier
μ = Σr_i / sqrt(Σr_i²) ≥ 1 → Θ_eff = L×8×μ → ħc/Δx drops → E_bound < E_free → positive B.

Paper: single axiom E_tot = m c² + ħc/Δx, Δx ≤ Θ_local; Θ_local from geometry only (no Δ in algebra).

Effective potential for the balanced well (pure algebra, no new constants):
  V_eff(r) = V_rep(r) + V_attr(r)
  V_rep(r) = A / r^12   (hard-core Pauli/horizon exclusion)
  V_attr(r) = -B φ(r)/r^6 - C·tr(M1@M2@Δ)·exp(-r/λ_coh)
A, B, C and λ_coh are fixed by ħc, lattice_base, and algebra. Minimum at dV_eff/dr = 0
gives r_eq ≈ 1.4 fm for nucleon pairs (alpha-particle rms radius scale).
"""

from __future__ import annotations

from collections import deque
import math
from typing import List, Optional, Set, Tuple, Union

import numpy as np

from pyhqiv.constants import C_SI, HBAR_C_MEV_FM
from pyhqiv.subatomic import _sphere_touching_mu

# ħc in MeV·m (radius from mass: r = ħc / (m c²) in length units)
_HBAR_C_MEV_M: float = HBAR_C_MEV_FM * 1e-15

# Coulomb scale for p-p repulsion (hypercharge/geometry); scaled by r_avg² so same order as radii
_COULOMB_RELAX_SCALE: float = 0.001

# Coherence length for trace term: multiple of pair contact scale (fm → m)
_LAMBDA_COH_FACTOR: float = 2.0

# Balanced well: connect when d < r_eq (min-energy state). r_eq ≈ scale * (r_i + r_j);
# nucleon r_sum ~ 1.2 fm, alpha rms ~ 1.4 fm => scale = 1.4/1.2
R_EQ_SCALE: float = 1.4 / 1.2



def effective_potential_pair(
    r_m: float,
    r1_m: float,
    r2_m: float,
    lattice_base_m: float,
    trace_M1_M2_Delta: float = 0.0,
    lambda_coh_m: Optional[float] = None,
) -> float:
    """
    Effective potential between two horizons (pure algebra, no new constants).

    V_eff(r) = A/r^12 + (-B φ(r)/r^6) + (-C·tr(M1@M2@Δ)·exp(-r/λ_coh)).
    A from Pauli/horizon exclusion; B from φ = 2c²/Θ(r) and ħc; C from ħc/(r1+r2).
    Universally applicable: any two horizon radii r1_m, r2_m (metres). Returns energy in MeV.
    """
    r = max(r_m, 1e-20)
    r1 = max(r1_m, 1e-20)
    r2 = max(r2_m, 1e-20)
    r_sum = r1 + r2
    L8 = lattice_base_m * 8.0

    # A: V_rep = A/r^12, A = ħc (r1+r2)^11 so repulsion scale at contact is ħc/(r1+r2)
    A = _HBAR_C_MEV_M * (r_sum ** 11)
    V_rep = A / (r ** 12)

    # μ(r): two spheres. Touching μ_touch = (r1+r2)/sqrt(r1²+r2²); beyond r_sum use decay
    rad = np.array([r1, r2], dtype=float)
    mu_touch = _sphere_touching_mu(rad)
    if r <= r_sum:
        # Overlap: μ grows as r decreases (so Θ grows, attraction stronger); avoid μ→0
        mu_r = max(mu_touch * (r / r_sum) ** 0.5, 0.5)
    else:
        lam = lambda_coh_m if lambda_coh_m is not None else _LAMBDA_COH_FACTOR * r_sum
        mu_r = 1.0 + (mu_touch - 1.0) * math.exp(-(r - r_sum) / max(lam, 1e-30))

    # φ(r) = 2c²/Θ(r), Θ(r) = L*8*μ(r). B so that -B φ/r^6 is in MeV: B*φ/r^6 = ħc/(μ r^6) * (r_sum)^6
    # => B = ħc (r_sum)^6 (L*8) / (2 c²)
    theta_r = L8 * max(mu_r, 1e-30)
    phi_r = 2.0 * (C_SI ** 2) / theta_r
    B = _HBAR_C_MEV_M * (r_sum ** 6) * L8 / (2.0 * (C_SI ** 2))
    V_attr_vdw = -B * phi_r / (r ** 6)

    # Trace term: -C·tr·exp(-r/λ_coh), C = ħc/r_sum
    C = _HBAR_C_MEV_M / r_sum
    lam = lambda_coh_m if lambda_coh_m is not None else _LAMBDA_COH_FACTOR * r_sum
    V_attr_trace = -C * float(trace_M1_M2_Delta) * math.exp(-r / max(lam, 1e-30))

    return float(V_rep + V_attr_vdw + V_attr_trace)


def equilibrium_separation_two_horizons(
    r1_m: float,
    r2_m: float,
    lattice_base_m: float,
    trace_M1_M2_Delta: float = 0.0,
    lambda_coh_m: Optional[float] = None,
) -> float:
    """
    Equilibrium separation r_eq where dV_eff/dr = 0 (balanced well).
    Universally applicable for any two horizon radii (metres). Returns r_eq in metres.
    For nucleon pairs gives ~1.4 fm (alpha-particle rms radius scale).
    """
    from scipy.optimize import minimize_scalar

    r_sum = max(r1_m + r2_m, 1e-20)
    # Bracket: minimum is between contact and a few times contact
    lo = r_sum * 0.6
    hi = r_sum * 3.0

    def obj(r_m: float) -> float:
        return effective_potential_pair(
            r_m, r1_m, r2_m, lattice_base_m,
            trace_M1_M2_Delta=trace_M1_M2_Delta,
            lambda_coh_m=lambda_coh_m,
        )

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
    return float(res.x)


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
        """
        Edges where distance(i,j) < r_eq(i,j) (min-energy state / balanced well).
        r_eq = R_EQ_SCALE * (r_i + r_j) so connectivity models the potential minimum,
        not strict touch. Radii from mass: r = ħc/(m c²).
        """
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
                r_eq_ij = R_EQ_SCALE * (self._radii_m[i] + self._radii_m[j])
                if d < r_eq_ij:
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


__all__ = [
    "HorizonNetwork",
    "relax_nucleon_positions",
    "relax_quark_positions",
    "effective_potential_pair",
    "equilibrium_separation_two_horizons",
    "R_EQ_SCALE",
]
