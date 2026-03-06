"""
Layer 0 of the binding-energy hierarchy: sub-nucleon constituents → nucleon.

Protons and neutrons are each made of sub-atomic constituents (in HQIV: horizon
modes or lattice dof at that scale). This module computes the energy of a single
proton or neutron as the bound state of its constituents, E = ħc Σ(1/Θ_i), and
exposes effective horizons Θ_proton, Θ_neutron for use by nuclear.py (layer 1).

**Design note:** The correct formulation is not only the color (confinement) force
but **color vs Coulomb**: the full matrix of all energy states (e.g. from
pyhqiv.algebra: 8×8, SU(3)_c, U(1)_Y) should be used so binding is the balance of
both. Current code uses a placeholder (constituent count + equal Θ); replace with
algebra-derived state matrix when implementing color–Coulomb competition.

Hierarchy:
  Layer 0 (here): constituents → E_proton, E_neutron, Θ_proton, Θ_neutron.
  Layer 1 (nuclear): E_free = P×E_proton + N×E_neutron.
  Layer 2 (nuclear): B_nuclear = E_free − E_nucleus.

See docs/binding_energy_walkthrough.md (§6.1 full matrix, color vs Coulomb).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pyhqiv.constants import ALPHA_EM_INV, HBAR_C_MEV_FM, M_D_MEV_QCD, M_U_MEV_QCD
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants

# ħc in MeV·m (same as nuclear.py)
_HBAR_C_MEV_M: float = HBAR_C_MEV_FM * 1e-15

# Placeholder: number of constituents per nucleon (e.g. quark-like; replace when
# paper specifies sub-nucleon structure).
CONSTITUENTS_PROTON: int = 3
CONSTITUENTS_NEUTRON: int = 3


def quark_flavors_for_nucleon(is_proton: bool) -> Tuple[str, str, str]:
    """Valence-quark content used by the layer-0 nucleon ladder."""
    if is_proton:
        return ("u", "u", "d")
    return ("u", "d", "d")


def _quark_radii_m_from_masses() -> Tuple[float, float]:
    """
    Effective quark horizon radii (m) from their masses via inverse-frequency scaling.

    r_q ∝ ħc / (m_q c²); the absolute scale cancels in the sphere-touching μ so we
    use ħc/(m_q c²) directly. In full HQIV the quark masses come from the mass
    equation at now; here we use PDG-like reference values M_U_MEV_QCD, M_D_MEV_QCD.
    """
    # Use MeV energies; c factors cancel in the proportionality.
    r_u = _HBAR_C_MEV_M / max(M_U_MEV_QCD, 1e-30)
    r_d = _HBAR_C_MEV_M / max(M_D_MEV_QCD, 1e-30)
    return (r_u, r_d)


def _sphere_touching_mu(radii: np.ndarray) -> float:
    """
    Sphere-touching mode multiplier μ for a set of radii.

    For radii r_i, μ = (Σ r_i) / sqrt(Σ r_i²) ≥ 1 encodes the Pythagorean
    "Casimir deficit" when spheres touch: more overlap → larger μ.
    """
    r = np.asarray(radii, dtype=float)
    num = float(np.sum(r))
    den = float(np.sqrt(np.sum(r * r)))
    if den <= 0.0:
        return 1.0
    return max(num / den, 1.0)


_R_U_M, _R_D_M = _quark_radii_m_from_masses()

# Fractional charges from hypercharge (OctonionHQIVAlgebra 4×4 block): Q_u = +2/3, Q_d = -1/3
_Q_U: float = 2.0 / 3.0
_Q_D: float = -1.0 / 3.0


def _quark_charges(flavor_content: str) -> np.ndarray:
    """Charges (in units of e) for three quarks from flavor string, e.g. 'uud', 'udd'."""
    return np.array([_Q_U if q == "u" else _Q_D for q in flavor_content.strip().lower()])


def _quark_radii_for_flavor(flavor_content: str) -> np.ndarray:
    """(3,) radii in m for flavor_content."""
    return np.array([_R_U_M if q == "u" else _R_D_M for q in flavor_content.strip().lower()])


def _quark_binding_angles(flavor_content: str) -> np.ndarray:
    """
    Three quarks arrange via fractional charge + horizon spheres (same relaxation as nuclei).

    Returns (3,) bond angles in radians: angle at vertex 0 (between 1-0-2), at 1, at 2.
    Proton (uud): u-u repulsion → ~109° tetrahedral preference. Neutron (udd): d-d attraction → more acute d-d angle.
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavor_content = flavor_content.strip().lower()
    if len(flavor_content) != 3:
        raise ValueError("flavor_content must be 3 characters (e.g. 'uud', 'udd')")
    radii = _quark_radii_for_flavor(flavor_content)
    charges = _quark_charges(flavor_content)
    positions = relax_quark_positions(radii, charges)
    angles = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        v1 = positions[j] - positions[i]
        v2 = positions[k] - positions[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-30 or n2 <= 1e-30:
            angles.append(0.0)
            continue
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        angles.append(float(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return np.array(angles)


def _quark_coulomb_energy_mev(flavor_content: str) -> float:
    """
    Electrostatic energy of the equilibrium 3-quark configuration.

    E_Coul = (α ℏc) Σ_{i<j} Q_i Q_j / d_ij (d in m → E in MeV). No new constants; α from ALPHA_EM_INV.
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavor_content = flavor_content.strip().lower()
    radii = _quark_radii_for_flavor(flavor_content)
    charges = _quark_charges(flavor_content)
    positions = relax_quark_positions(radii, charges)
    alpha = 1.0 / max(ALPHA_EM_INV, 1e-30)
    e_mev = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = max(float(np.linalg.norm(positions[i] - positions[j])), 1e-30)
            e_mev += alpha * _HBAR_C_MEV_M * charges[i] * charges[j] / d
    return float(e_mev)


def _quark_geometry_theta_m(flavor_content: str, t_cmb: float = 2.725) -> float:
    """
    Free-nucleon effective horizon from charge-driven geometry only.

    μ = Σr_i / √(Σr_i²); Θ = L×8×μ. Equilibrium from relax_quark_positions (Coulomb + touching).
    """
    from pyhqiv.horizon_network import relax_quark_positions

    flavor_content = flavor_content.strip().lower()
    radii = _quark_radii_for_flavor(flavor_content)
    relax_quark_positions(radii, _quark_charges(flavor_content))  # equilibrium; μ depends only on radii
    mu = _sphere_touching_mu(radii)
    L = get_hqiv_nuclear_constants(t_cmb)["LATTICE_BASE_M"]
    return L * 8.0 * max(mu, 1e-30)


def _nucleon_matrix_invariants(is_proton: bool, algebra=None) -> Tuple[float, float]:
    """
    Two geometric invariants of the unprojected 3-quark composite.

    `coherence` measures how efficiently the composite packs tension into the same
    8x8 support. Larger coherence means a lower confinement-energy cost. `span`
    measures the apparent size of the merged state. Replacing a `u` by a `d`
    reduces coherence but increases span, so the neutron comes out heavier while
    carrying a larger effective horizon.
    """
    from pyhqiv.energy_field import merge_constituents

    mats = quark_state_matrices_for_nucleon(is_proton, algebra=algebra)
    composite = merge_constituents(mats, project_singlet=False, algebra=algebra)
    tr = float(np.trace(composite.state_matrix))
    fro_sq = float(np.linalg.norm(composite.state_matrix) ** 2)
    coherence = (tr * tr) / max(fro_sq, 1e-30)
    span = fro_sq / max(abs(tr), 1e-30)
    return (coherence, span)


def _constituent_horizons_m(
    n_constituents: int,
    t_cmb: float = 2.725,
    base_scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Horizon Θ_i (m) for each constituent in a bound nucleon.

    Placeholder: sub-nucleon scale derived from nuclear LATTICE_BASE; tuned so
    E = ħc Σ(1/Θ_i) is on the order of nucleon mass (~938 MeV). Replace with
    paper-derived constituent structure when available.
    """
    const = get_hqiv_nuclear_constants(t_cmb)
    L_nuclear = const["LATTICE_BASE_M"]
    # Sub-nucleon scale: much smaller than nuclear so confinement gives ~GeV.
    # E ~ ħc * n / Θ => Θ ~ ħc * n / E. For E ~ 938 MeV, n=3: Θ ~ 6.3e-19 m.
    # Use a factor below nuclear scale (1e-4 gives ~1.9e-19 m per constituent).
    sub_scale = L_nuclear * 1e-4 * base_scale_factor
    # Equal horizons per constituent (symmetric bound state)
    theta = sub_scale / max(n_constituents, 1)
    return np.full(n_constituents, theta, dtype=float)


def energy_from_constituents_mev(theta_i_m: np.ndarray) -> float:
    """E = ħc Σ(1/Θ_i) (MeV) for a set of constituent horizons."""
    if len(theta_i_m) == 0:
        return 0.0
    inv = 1.0 / np.maximum(np.asarray(theta_i_m, dtype=float), 1e-30)
    return float(_HBAR_C_MEV_M * np.sum(inv))


def effective_theta_m(energy_mev: float) -> float:
    """Effective horizon Θ_eff such that E = ħc/Θ_eff (m)."""
    if energy_mev <= 0:
        return 1e-30
    return _HBAR_C_MEV_M / energy_mev


def proton_energy_mev(t_cmb: float = 2.725) -> float:
    """
    Rest energy of a single proton (MeV) from charge-driven layer-0 geometry.

    E = ħc/Θ + E_Coul; uud has u-u repulsion so E_Coul > 0. Same sphere-touching + Coulomb as nuclei.
    """
    theta = _quark_geometry_theta_m("uud", t_cmb)
    e_tension = _HBAR_C_MEV_M / max(theta, 1e-30)
    e_coul = _quark_coulomb_energy_mev("uud")
    return e_tension + e_coul


def neutron_energy_mev(t_cmb: float = 2.725) -> float:
    """
    Rest energy of a single neutron (MeV) from charge-driven layer-0 geometry.

    E = ħc/Θ + E_Coul; udd has d-d attraction so E_Coul < 0. Neutron ends up heavier (Θ_n > Θ_p, E_Coul less positive).
    """
    theta = _quark_geometry_theta_m("udd", t_cmb)
    e_tension = _HBAR_C_MEV_M / max(theta, 1e-30)
    e_coul = _quark_coulomb_energy_mev("udd")
    return e_tension + e_coul


def proton_effective_theta_m(t_cmb: float = 2.725) -> float:
    """
    Effective horizon Θ_proton (m) from uud charge-driven geometry.

    Equilibrium from relax_quark_positions (fractional Q + touching); Θ = L×8×μ, μ = Σr/√(Σr²).
    """
    return _quark_geometry_theta_m("uud", t_cmb)


def neutron_effective_theta_m(t_cmb: float = 2.725) -> float:
    """
    Effective horizon Θ_neutron (m) from udd charge-driven geometry.

    Same rule; udd gives different μ and bond angles (d-d attraction → more acute angles).
    """
    return _quark_geometry_theta_m("udd", t_cmb)


def nucleon_energies_mev(t_cmb: float = 2.725) -> Tuple[float, float]:
    """(E_proton, E_neutron) in MeV from layer 0. For use by layer 1."""
    return (proton_energy_mev(t_cmb), neutron_energy_mev(t_cmb))


def nucleon_effective_theta_m(t_cmb: float = 2.725) -> Tuple[float, float]:
    """(Θ_proton, Θ_neutron) in m from layer 0. For use by layer 1."""
    return (proton_effective_theta_m(t_cmb), neutron_effective_theta_m(t_cmb))


# ---------------------------------------------------------------------------
# 8×8 matrix ladder: color singlet → proton from quark states (energy_field)
# ---------------------------------------------------------------------------


def color_singlet_projector(algebra=None) -> np.ndarray:
    """
    Projector onto SU(3)_c singlet (e7 direction: colour preferred axis in paper).

    Returns 8×8 matrix P such that P @ state projects onto the colour-singlet
    subspace. Uses e7 (index 7) as the preferred axis preserved by g₂ color generators.
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    e7 = np.zeros(8)
    e7[7] = 1.0
    P = np.outer(e7, e7)
    return P


def make_proton_from_quark_states(
    quark_state_matrices: List[np.ndarray],
    algebra=None,
):
    """
    Subatomic scale: merge three quark 8×8 states into one nucleon (proton) via
    the same merge_constituents used at all scales. Returns colour-singlet composite.
    """
    from pyhqiv.energy_field import merge_constituents
    return merge_constituents(
        list(quark_state_matrices),
        project_singlet=True,
        algebra=algebra,
    )


def make_neutron_from_quark_states(
    quark_state_matrices: List[np.ndarray],
    algebra=None,
):
    """Build neutron as colour-singlet 8×8 from three quark matrices (e.g. d,d,u). Same as proton but flavor differs."""
    return make_proton_from_quark_states(quark_state_matrices, algebra=algebra)


def quark_state_matrix(flavor: str = "u", color_index: int = 0, algebra=None) -> np.ndarray:
    """
    8×8 state for one quark (flavor u/d, color 0,1,2).

    Pure color (g₂) + flavor-dependent scale only. No Δ admixture — binding comes from
    sphere-touching μ applied on the full connected component in HorizonNetwork.
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    color_gens = algebra._identify_color_generators()
    idx = min(color_index, len(color_gens) - 1) if color_gens else 0
    gen = color_gens[idx] if color_gens else np.zeros((8, 8))
    scale = 0.1 if flavor == "u" else 0.12
    return np.eye(8) + scale * gen


def quark_state_matrices_for_nucleon(is_proton: bool, algebra=None) -> List[np.ndarray]:
    """Three 8x8 quark states in the physical valence ordering for a nucleon."""
    return [
        quark_state_matrix(flavor, color_index=i, algebra=algebra)
        for i, flavor in enumerate(quark_flavors_for_nucleon(is_proton))
    ]


def quark_binding_angles(flavor_content: str) -> np.ndarray:
    """
    Binding angles (rad) for a 3-quark configuration from fractional charge + sphere-touching.

    flavor_content: "uud" (proton) or "udd" (neutron). Returns (3,) angles at each vertex.
    """
    return _quark_binding_angles(flavor_content)


__all__ = [
    "CONSTITUENTS_PROTON",
    "CONSTITUENTS_NEUTRON",
    "quark_flavors_for_nucleon",
    "quark_binding_angles",
    "energy_from_constituents_mev",
    "effective_theta_m",
    "proton_energy_mev",
    "neutron_energy_mev",
    "proton_effective_theta_m",
    "neutron_effective_theta_m",
    "nucleon_energies_mev",
    "nucleon_effective_theta_m",
    "color_singlet_projector",
    "make_proton_from_quark_states",
    "make_neutron_from_quark_states",
    "quark_state_matrix",
    "quark_state_matrices_for_nucleon",
]
