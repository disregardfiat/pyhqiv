"""
Phase-lifted fanoplane fusion: pure algebraic entanglement and binding (no positions/radii).

Uses only algebra.py objects: L(e_i), Δ, matrix multiplication, trace with Δ.
Entanglement: M12 = M1 + M2 + [M1,M2]_Δ  with  [M1,M2]_Δ = M1 Δ M2 - M2 Δ M1.
Binding: B = E_free - E_bound = -tr([M1,M2]_Δ @ Δ).
Holding distance: σ = ‖[M1,M2]_Δ‖_F (dimensionless, fanoplane separation).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def phase_lifted_commutator(M1: np.ndarray, M2: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """[M1, M2]_Δ = M1 Δ M2 - M2 Δ M1 (entanglement term)."""
    return M1 @ Delta @ M2 - M2 @ Delta @ M1


def entangle_particles(M1: np.ndarray, M2: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """Entangled composite M12 = M1 + M2 + [M1, M2]_Δ (non-separable)."""
    comm = phase_lifted_commutator(M1, M2, Delta)
    return M1 + M2 + comm


def holding_distance(M1: np.ndarray, M2: np.ndarray, Delta: np.ndarray) -> float:
    """Algebraic separation σ = ‖[M1, M2]_Δ‖_F (dimensionless)."""
    comm = phase_lifted_commutator(M1, M2, Delta)
    return float(np.linalg.norm(comm, "fro"))


def binding_energy_pair(M1: np.ndarray, M2: np.ndarray, Delta: np.ndarray) -> float:
    """B = E_free - E_bound = -tr([M1,M2]_Δ @ Δ) in trace units."""
    comm = phase_lifted_commutator(M1, M2, Delta)
    return float(-np.trace(comm @ Delta))


def iterated_fusion(matrices: List[np.ndarray], Delta: np.ndarray) -> np.ndarray:
    """Fuse all particles by repeated phase-lifted entanglement: M_12..A from M_1,...,M_A."""
    if not matrices:
        return np.zeros((8, 8))
    if len(matrices) == 1:
        return np.asarray(matrices[0], dtype=float).copy()
    M = matrices[0]
    for i in range(1, len(matrices)):
        M = entangle_particles(M, matrices[i], Delta)
    return M


def binding_energy_algebraic(
    matrices: List[np.ndarray],
    Delta: np.ndarray,
    lattice_base_m: float,
    hbar_c_mev_m: float,
) -> Tuple[float, np.ndarray]:
    """
    Binding energy B (MeV) and per-particle bound Θ (m) from pure algebra.

    B = (E_free - E_bound) * scale_mev. Per-particle σ_i = mean over j of holding_distance(Mi, Mj);
    theta_i = lattice_base * 8 / (1 + σ_i) so smaller σ → larger Θ (more bound).
    """
    A = len(matrices)
    if A == 0:
        return 0.0, np.array([])
    if A == 1:
        return 0.0, np.array([lattice_base_m * 8.0])

    E_free = sum(float(np.trace(M @ Delta)) for M in matrices)
    M_bound = iterated_fusion(matrices, Delta)
    E_bound = float(np.trace(M_bound @ Delta))
    B_trace = E_free - E_bound

    # Scale to MeV: one trace unit → ħc/L energy
    scale_mev = hbar_c_mev_m / max(lattice_base_m, 1e-30)
    B_mev = B_trace * scale_mev

    # Per-particle σ (mean holding distance to others) → theta
    sigmas = np.zeros(A)
    for i in range(A):
        if A == 2:
            sigmas[i] = holding_distance(matrices[i], matrices[1 - i], Delta)
        else:
            others = [matrices[j] for j in range(A) if j != i]
            M_rest = iterated_fusion(others, Delta)
            sigmas[i] = holding_distance(matrices[i], M_rest, Delta)
    theta_bound = lattice_base_m * 8.0 / (1.0 + np.maximum(sigmas, 1e-30))
    return float(B_mev), np.asarray(theta_bound, dtype=float)


__all__ = [
    "phase_lifted_commutator",
    "entangle_particles",
    "holding_distance",
    "binding_energy_pair",
    "iterated_fusion",
    "binding_energy_algebraic",
]
