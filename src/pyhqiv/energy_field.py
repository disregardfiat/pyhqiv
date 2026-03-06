"""
HQIV energy field: 8×8 matrix-valued state (universal from quark to macro scale).

The informational vacuum at any point is an 8-component object (octonion) or its 8×8
matrix representation. The local energy density is the paper axiom (scalar projection)
plus an algebraic contribution from the full matrix. Total energy of a system defines
its horizon: E_tot = ∫ (ρ c² + ħc/Δx) d³x and Θ_system = ħc / E_tot (or from composite invariant).

Single merge component: merge_constituents() composes 8×8 states (left action, optional
singlet projection) and is used at all scales — subatomic (quarks→nucleon), nuclear
(nucleons→nucleus), molecular (atoms→molecule), and beyond (solar, etc.). No tuning.

- Octonion left-multiplications L(e_i) and commutators + Δ close to so(8).
- SU(3)_c = g₂ generators preserving e₇; hypercharge Y in 4×4 block; triality → generations.

See docs/binding_energy_walkthrough.md §6.1 (full matrix, color vs Coulomb).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from pyhqiv.constants import C_SI, HBAR_SI, HBAR_C_MEV_FM

# ħc in J·m for energy density (scalar part: ħc/Δx with Δx in m)
_HBAR_C_SI: float = HBAR_SI * C_SI
# ħc in MeV·m for horizon-from-energy: Θ = ħc / E_mev
_HBAR_C_MEV_M: float = HBAR_C_MEV_FM * 1e-15


def _default_algebra():
    from pyhqiv.algebra import OctonionHQIVAlgebra
    return OctonionHQIVAlgebra(verbose=False)


def _singlet_projector(algebra=None) -> np.ndarray:
    """
    Projector onto SU(3)_c singlet (e7 direction: colour preferred axis).
    Paper's g₂ color generators preserve e7; singlet = projection onto e7.
    """
    if algebra is None:
        algebra = _default_algebra()
    e7 = np.zeros(8)
    e7[7] = 1.0
    return np.outer(e7, e7)


def merge_constituents(
    constituents: List[Union[np.ndarray, "HQIVEnergyField"]],
    project_singlet: bool = True,
    algebra=None,
) -> "HQIVEnergyField":
    """
    Single merge component for all scales: subatomic → nuclear → molecular → solar and beyond.

    Composes constituent 8×8 states via left action (octonion multiplication). Optionally
    projects onto the invariant subspace (singlet). The composite's total energy defines
    its effective horizon (use effective_theta_local or effective_horizon_from_energy_mev).

    Parameters
    ----------
    constituents : list of (8, 8) arrays or HQIVEnergyField
        Each element is either an 8×8 state matrix or an object with .state_matrix.
    project_singlet : bool
        If True, project composed product onto SU(3)_c singlet (e7). Use True for
        color-singlet composites (nucleon, nucleus, molecule).
    algebra : OctonionHQIVAlgebra, optional
        Default: OctonionHQIVAlgebra(verbose=False).

    Returns
    -------
    HQIVEnergyField
        Composite state. Use .effective_theta_local(lattice_base_m, density) for Θ from
        algebraic invariant, or effective_horizon_from_energy_mev(E_tot) when E_tot
        comes from an integral over the system.
    """
    alg = algebra or _default_algebra()
    total = np.eye(8)
    for c in constituents:
        M = np.asarray(c, dtype=float) if isinstance(c, np.ndarray) else c.state_matrix
        if M.shape != (8, 8):
            M = np.asarray(M, dtype=float).reshape(8, 8)
        total = total @ M
    if project_singlet:
        P = _singlet_projector(alg)
        total = P @ total @ P.T
    return HQIVEnergyField(algebra=alg, state_matrix=total)


def effective_horizon_from_energy_mev(energy_mev: float) -> float:
    """
    Effective horizon Θ (m) when total energy of the system defines it: Θ = ħc / E_tot.

    Use when E_tot comes from integrating the axiom over the system (several integrals).
    No tuning: same relation at every scale (subatomic to solar).
    """
    if energy_mev <= 0:
        return 1e-30
    return _HBAR_C_MEV_M / energy_mev


def species_matrix_for_species(species: str, algebra=None) -> np.ndarray:
    """
    Eight-by-eight state matrix for an element (H, C, N, O, S, ...).
    Uses algebra: identity + small perturbation from Delta/L so that
    composition and projection give species-dependent effective Θ.
    """
    if algebra is None:
        algebra = _default_algebra()
    # Distinct perturbation per species (hypercharge/block style)
    seed = hash(species.strip().upper() or "X") % 2147483647
    np.random.seed(seed)
    # Use Delta and one L so different species get different matrices
    idx = abs(seed) % max(1, len(algebra.L))
    gen = algebra.L[idx] if algebra.L else algebra.Delta
    scale = 0.05 + 0.05 * (abs(seed) % 10) / 10.0
    M = np.eye(8) + scale * gen
    np.random.seed(None)
    return M


class HQIVEnergyField:
    """
    Lightweight 8×8 matrix-valued energy field (universal equation).

    Same equation applies from quark scale (Δx ~ 10⁻¹⁸ m) to macro liquid He
    (Δx ~ coherence length). Backward compatibility: project_scalar_phi() yields
    scalar φ for existing code (atom.py, system.py, lattice.py, thermo.py, fluid.py).
    """

    def __init__(
        self,
        algebra: Optional["OctonionHQIVAlgebra"] = None,
        state_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.algebra = algebra or _default_algebra()
        if state_matrix is not None:
            state_matrix = np.asarray(state_matrix, dtype=float)
            if state_matrix.shape != (8, 8):
                raise ValueError("state_matrix must be 8×8")
            self.state_matrix = state_matrix.copy()
        else:
            self.state_matrix = np.eye(8)  # vacuum = no excitation

    @classmethod
    def from_atoms(
        cls,
        atoms: List,
        positions: Optional[np.ndarray] = None,
        algebra=None,
    ) -> "HQIVEnergyField":
        """
        Molecular scale: merge atoms into one composite via merge_constituents.
        Each atom contributes its species 8×8 matrix; same merge as subatomic/nuclear.
        """
        alg = algebra or _default_algebra()
        matrices = []
        for atom in atoms:
            species = getattr(atom, "species", None) or getattr(atom, "species_symbol", "X")
            if hasattr(atom, "_core"):
                species = getattr(atom._core, "species", species)
            matrices.append(species_matrix_for_species(str(species), alg))
        return merge_constituents(matrices, project_singlet=True, algebra=alg)

    def energy_density(
        self,
        mass_density: float,
        delta_x: float,
        hbar_c: Optional[float] = None,
    ) -> float:
        """
        Universal full-matrix E_tot density (paper axiom + algebraic contribution).

        scalar_part = ρ c² + ħc/Δx  (Δx ≤ Θ_local).
        algebraic_part = (ħc/Δx) × normalized trace(state_matrix @ Δ) so that
        identity state adds zero; internal dof (color, generations) add to energy.

        Parameters
        ----------
        mass_density : float
            Mass density (kg/m³).
        delta_x : float
            Local resolution / horizon scale (m).
        hbar_c : float, optional
            ħc in J·m; default from constants.

        Returns
        -------
        float
            Energy density (J/m³).
        """
        hc = hbar_c if hbar_c is not None else _HBAR_C_SI
        dx = max(float(delta_x), 1e-35)
        scalar_part = mass_density * (C_SI ** 2) + hc / dx
        # Algebraic part: same dimension as ħc/Δx; identity gives trace(I@Δ)=0
        alg = np.trace(self.state_matrix @ self.algebra.Delta)
        # Normalize so typical perturbations give O(1) correction
        algebraic_part = (hc / dx) * (alg / 8.0)
        return float(scalar_part + algebraic_part)

    def project_scalar_phi(self, c: Optional[float] = None) -> float:
        """
        Backward-compatible scalar φ = 2c²/Θ_eff for existing code.

        Effective scale from state_matrix trace: φ = 2c² / trace(M).
        For vacuum (identity) trace=8 so φ = c²/4.

        Parameters
        ----------
        c : float, optional
            Speed of light (m/s); default from constants.

        Returns
        -------
        float
            φ in (m/s)².
        """
        tr = max(float(np.trace(self.state_matrix)), 1e-30)
        speed = c if c is not None else C_SI
        return 2.0 * (speed ** 2) / tr

    def effective_theta_scale(self) -> float:
        """
        Dimensionless scale from state matrix trace: trace(M)/8.
        Vacuum (identity) gives 1.0. For Θ in metres use L_ref * effective_theta_scale().
        """
        tr = max(float(np.trace(self.state_matrix)), 1e-30)
        return tr / 8.0

    def effective_theta_local(
        self,
        lattice_base_m: float,
        local_density: float = 1.0,
    ) -> float:
        """
        Paper-faithful Θ_local from algebraic invariant (trace of state @ Delta).
        Binding increases effective modes via coherence → larger Θ_local than free.
        Θ_local = lattice_base_m * effective_modes / (1 + 0.1*local_density).
        """
        algebraic_shift = float(np.trace(self.state_matrix @ self.algebra.Delta))
        effective_modes = 8.0 + algebraic_shift
        return lattice_base_m * max(effective_modes, 1e-30) / (1.0 + 0.1 * max(local_density, 0.0))

    def total_energy_density(
        self,
        mass_density: float,
        delta_x: float,
        hbar_c: Optional[float] = None,
    ) -> float:
        """
        Exact paper axiom: E_tot density = ρ c² + ħc/Δx (Δx ≤ Θ_local).
        Same as energy_density(); name for integral E_tot = ∫ (ρ c² + ħc/Δx) d³x.
        """
        return self.energy_density(mass_density, delta_x, hbar_c=hbar_c)

    def coherence(self) -> float:
        """
        Coherence from state matrix (e.g. for superfluid: det or eigenvalue alignment).

        Returns abs(det(state_matrix)); for identity det=1. For aligned BEC-like state
        coherence → 1; for de-aligned (e.g. QGP) coherence < 1.
        """
        return float(np.abs(np.linalg.det(self.state_matrix)))

    @staticmethod
    def energy_mev_from_theta_m(theta_m: float) -> float:
        """
        Total energy E (MeV) when horizon defines it: E = ħc/Θ. Inverse of effective_horizon_from_energy_mev.
        """
        if theta_m <= 0:
            return 0.0
        return _HBAR_C_MEV_M / theta_m


__all__ = [
    "HQIVEnergyField",
    "merge_constituents",
    "effective_horizon_from_energy_mev",
    "species_matrix_for_species",
]
