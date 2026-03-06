"""
HQIV first-principles nuclear decay.

Binding energy must be computed hierarchically (bottom to top):
  Layer 0 (sub-nucleon): constituents → bound proton/neutron (E_proton, E_neutron).
  Layer 1 (nucleon): free nucleon energy from layer 0 (or Θ_free_p, Θ_free_n until layer 0 exists).
  Layer 2 (nucleus): B_nuclear = E(P free protons + N free neutrons) − E(nucleus).
Currently only layers 1–2 are implemented; layer 0 (constituents) is not yet in code.

E_tot = Σ m c² + Σ ħc / Θ_i   (Θ_i ≤ horizon radius from causal monogamy)
φ_i = 2 c² / Θ_i
P_snap = exp(-ΔE_info / (ħc / Θ_avg)) × (φ / (φ + φ_crit))
λ = P_snap / τ_tick   → macroscopic λ = λ / scale   (scale from T_CMB lapse)
t_1/2 = ln(2) / λ

All constants derived from T_CMB via get_hqiv_nuclear_constants.
No SEMF, no empirical B, no hard-coded "now" values.
See docs/binding_energy_walkthrough.md for the full hierarchical walkthrough.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

import numpy as np

from scipy.optimize import minimize

from pyhqiv.constants import (
    C_SI,
    E_PL_MEV,
    HBAR_C_MEV_FM,
    L_PLANCK_M,
    M_D_MEV_QCD,
    M_NEUTRON_MEV,
    M_PROTON_MEV,
    M_U_MEV_QCD,
)
from pyhqiv.fluid import f_inertia
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.horizon_network import (
    HorizonNetwork,
    equilibrium_separation_two_horizons,
    relax_nucleon_positions,
)

# Element symbol → (P, N_default) for naming and Nuclide parsing only
_ELEMENT_PN: dict = {
    "H": (1, 0), "He": (2, 2), "Li": (3, 4), "Be": (4, 5), "B": (5, 6), "C": (6, 6),
    "N": (7, 7), "O": (8, 8), "F": (9, 10), "Ne": (10, 10), "Na": (11, 12), "Mg": (12, 12),
    "Al": (13, 14), "Si": (14, 14), "P": (15, 16), "S": (16, 16), "Cl": (17, 18),
    "Ar": (18, 22), "K": (19, 20), "Ca": (20, 20), "Fe": (26, 30), "Cu": (29, 34),
    "Zn": (30, 34), "U": (92, 146),
}
# Long names → symbol for Nuclide identifier parsing (e.g. "carbon-14")
_ELEMENT_NAME_TO_SYMBOL: dict = {
    "hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be", "boron": "B",
    "carbon": "C", "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne",
    "sodium": "Na", "magnesium": "Mg", "aluminum": "Al", "aluminium": "Al", "silicon": "Si",
    "phosphorus": "P", "sulfur": "S", "sulphur": "S", "chlorine": "Cl", "argon": "Ar",
    "potassium": "K", "calcium": "Ca", "iron": "Fe", "copper": "Cu", "zinc": "Zn",
    "uranium": "U",
}


def nuclide_from_symbol(symbol: str, N: Optional[int] = None) -> Tuple[int, int]:
    """(P, N) from element symbol. If N given, use (P, N); else use default isotope from table."""
    sym = symbol.strip().title()
    P, N_default = _ELEMENT_PN.get(sym, (0, 0))
    n = N if N is not None else N_default
    return (P, n)


def _bound_theta_from_matrix_composition(
    P: int,
    N: int,
    lattice_base_m: float,
    algebra=None,
) -> float:
    """
    Nuclear scale: merge P protons + N neutrons via merge_constituents (same component
    as subatomic/molecular). Bound Θ from composite invariant: effective_modes = 8 + trace(M @ Δ).
    No tuning.
    """
    if P <= 0 and N <= 0:
        return lattice_base_m * 8.0
    from pyhqiv.subatomic import (
        make_proton_from_quark_states,
        make_neutron_from_quark_states,
        quark_state_matrices_for_nucleon,
    )
    from pyhqiv.energy_field import merge_constituents

    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)

    constituents = []
    for _ in range(P):
        mats = quark_state_matrices_for_nucleon(True, algebra=algebra)
        constituents.append(make_proton_from_quark_states(mats, algebra=algebra))
    for _ in range(N):
        mats = quark_state_matrices_for_nucleon(False, algebra=algebra)
        constituents.append(make_neutron_from_quark_states(mats, algebra=algebra))

    # Unprojected product for invariant (projection loses trace(M@Δ)); use merge with project_singlet=False to get product, then read invariant.
    composite_field = merge_constituents(constituents, project_singlet=False, algebra=algebra)
    algebraic_shift = float(np.trace(composite_field.state_matrix @ algebra.Delta))
    effective_modes = 8.0 + algebraic_shift
    return float(lattice_base_m * max(effective_modes, 1e-30))


def _free_nucleon_thetas_m(lattice_base_m: float, algebra=None) -> Tuple[float, float]:
    """First principles: proton and neutron effective Θ (m) from merge(3 quarks) each."""
    from pyhqiv.subatomic import neutron_effective_theta_m, proton_effective_theta_m

    return (proton_effective_theta_m(), neutron_effective_theta_m())


def _nucleon_state_matrix_unprojected(is_proton: bool, algebra) -> np.ndarray:
    """Unprojected 8×8 nucleon state (product of 3 quarks, no singlet projection) for network invariant."""
    from pyhqiv.subatomic import quark_state_matrices_for_nucleon
    from pyhqiv.energy_field import merge_constituents
    quarks = quark_state_matrices_for_nucleon(is_proton, algebra=algebra)
    composite = merge_constituents(quarks, project_singlet=False, algebra=algebra)
    return composite.state_matrix


def build_nucleon_matrix_with_phase(
    is_proton: bool,
    lattice_base_m: float,
    algebra=None,
) -> np.ndarray:
    """
    Nucleon matrix M = M_base + θ Δ with axiom-derived phase-lift.

    θ = (π/2) arctan(E/E_Pl_eff) from local rapidity; E_Pl_eff = E_Pl × (L_Pl/L)
    so lattice shell sets the effective Planck scale (paper lattice-shell index).
    Ensures tr(M @ Δ) ≠ 0 so the algebraic binding path yields non-zero B.
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)
    M_base = _nucleon_state_matrix_unprojected(is_proton, algebra)
    E_mev = M_PROTON_MEV if is_proton else M_NEUTRON_MEV
    # E_Pl_eff from lattice: at nuclear L, E_Pl_eff ~ 10 MeV so θ ~ O(1)
    E_Pl_eff = E_PL_MEV * (L_PLANCK_M / max(lattice_base_m, 1e-35))
    theta = (np.pi / 2.0) * np.arctan(E_mev / max(E_Pl_eff, 1e-30))
    return np.asarray(M_base, dtype=float) + theta * np.asarray(algebra.Delta, dtype=float)


def _initial_guess_positions(
    radii_m: np.ndarray,
    is_proton: List[bool],
    lattice_base_m: float,
) -> np.ndarray:
    """Initial positions (m) for minimizer: tetrahedron for A=4 at balanced-well scale; else force-based relax."""
    from pyhqiv.horizon_network import R_EQ_SCALE

    A = len(radii_m)
    if A == 4:
        r_avg = float(np.mean(radii_m))
        # Min-energy state: edge slightly inside r_eq so graph sees one component (d < r_eq)
        edge_m = R_EQ_SCALE * (2.0 * r_avg) * 0.98
        scale = edge_m / (2.0 * np.sqrt(2.0))
        verts = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=float) * scale
        return verts
    return relax_nucleon_positions(radii_m, is_proton)


def minimize_nucleon_configuration(
    radii_m: np.ndarray,
    is_proton: List[bool],
    lattice_base_m: float,
    algebra=None,
    initial_guess: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Find equilibrium positions where repulsion + attraction balance (balanced well).

    Uses HorizonNetwork.total_energy() as objective; constants A, B, C from algebra
    and lattice. Returns positions (m) of nucleons. Universally applicable to any
    number of nucleons; for A=4 seeds tetrahedron at r_eq ~ 1.4 fm.
    """
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)

    A = len(radii_m)
    if A == 0:
        return np.zeros((0, 3))
    if A == 1:
        return np.zeros((1, 3))

    M_p = _nucleon_state_matrix_unprojected(True, algebra)
    M_n = _nucleon_state_matrix_unprojected(False, algebra)

    def objective(pos_flat: np.ndarray) -> float:
        positions = pos_flat.reshape(-1, 3)
        nodes = [
            (positions[i], M_p if is_proton[i] else M_n, M_PROTON_MEV if is_proton[i] else M_NEUTRON_MEV)
            for i in range(A)
        ]
        net = HorizonNetwork(nodes, lattice_base_m, algebra=algebra)
        return net.total_energy()

    if initial_guess is None:
        initial_guess = _initial_guess_positions(radii_m, is_proton, lattice_base_m)
    x0 = np.asarray(initial_guess, dtype=float).reshape(A, 3)
    # Bounds in metres (~ ±5 fm)
    bound_fm = 5.0 * 1e-15
    bounds = [(-bound_fm, bound_fm)] * (A * 3)

    res = minimize(
        objective,
        x0.flatten(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )
    positions = res.x.reshape(-1, 3)
    # Centre at origin
    positions -= np.mean(positions, axis=0)
    return positions


def _quark_level_bound_thetas(
    P: int,
    N: int,
    lattice_base_m: float,
    algebra=None,
) -> np.ndarray:
    """
    Per-nucleon bound Θ from a quark-level HorizonNetwork (¹H, ²H, ³H).
    Quark interactions drive nucleon attraction and decay; not just macroscopically accurate.
    Returns theta_bound array of length A = P+N (one effective Θ per nucleon from its 3 quarks).
    """
    from pyhqiv.horizon_network import relax_quark_positions
    from pyhqiv.subatomic import (
        _quark_charges,
        _quark_radii_for_flavor,
        quark_state_matrices_for_nucleon,
    )

    A = P + N
    if A <= 0 or A > 3:
        return np.array([])
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)

    # Nucleon positions (min-energy state)
    hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
    r_p = hbar_c_mev_m / M_PROTON_MEV
    r_n = hbar_c_mev_m / M_NEUTRON_MEV
    radii_m = np.array([r_p] * P + [r_n] * N)
    is_proton = [True] * P + [False] * N
    nucleon_positions = minimize_nucleon_configuration(
        radii_m, is_proton, lattice_base_m, algebra
    )

    # Build 3*A quark nodes: (position, 8x8, mass_mev)
    nodes: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for n in range(A):
        is_p = is_proton[n]
        flavor = "uud" if is_p else "udd"
        radii_q = _quark_radii_for_flavor(flavor)
        charges = _quark_charges(flavor)
        local_q = relax_quark_positions(radii_q, charges)  # (3,3) m, centred
        mats = quark_state_matrices_for_nucleon(is_p, algebra=algebra)
        masses = [M_U_MEV_QCD if f == "u" else M_D_MEV_QCD for f in flavor]
        for q in range(3):
            pos = nucleon_positions[n] + local_q[q]
            nodes.append((pos, mats[q], masses[q]))

    net = HorizonNetwork(nodes, lattice_base_m, algebra=algebra)
    theta_per_quark = net.effective_theta_array()  # length 3*A

    # Per-nucleon Θ = geometric mean of its 3 quarks' Θ (quark-driven tension)
    theta_bound = np.zeros(A)
    for n in range(A):
        t0 = theta_per_quark[3 * n]
        t1 = theta_per_quark[3 * n + 1]
        t2 = theta_per_quark[3 * n + 2]
        theta_bound[n] = (t0 * t1 * t2) ** (1.0 / 3.0)
    return theta_bound


def _binding_energy_via_network(
    P: int,
    N: int,
    lattice_base_m: float,
    algebra=None,
) -> Tuple[float, np.ndarray]:
    """
    Binding energy and per-nucleon bound Θ from HorizonNetwork (overlap graph + composite invariant).

    E_free = sum of single-nucleon network energies; E_bound = one network of P+N nucleons.
    Returns (B_mev, theta_bound_array) with theta_bound_array of length A = P+N.
    """
    if P <= 0 and N <= 0:
        return (0.0, np.array([]))
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)

    M_p = _nucleon_state_matrix_unprojected(True, algebra)
    M_n = _nucleon_state_matrix_unprojected(False, algebra)
    origin = np.zeros(3)
    E_free = 0.0
    for _ in range(P):
        net = HorizonNetwork(
            [(origin, M_p, M_PROTON_MEV)],
            lattice_base_m,
            algebra=algebra,
        )
        E_free += net.total_energy()
    for _ in range(N):
        net = HorizonNetwork(
            [(origin, M_n, M_NEUTRON_MEV)],
            lattice_base_m,
            algebra=algebra,
        )
        E_free += net.total_energy()

    A = P + N
    hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
    r_p = hbar_c_mev_m / M_PROTON_MEV
    r_n = hbar_c_mev_m / M_NEUTRON_MEV
    radii_m = np.array([r_p] * P + [r_n] * N)
    is_proton = [True] * P + [False] * N
    positions = minimize_nucleon_configuration(radii_m, is_proton, lattice_base_m, algebra)
    nodes = (
        [(positions[i], M_p, M_PROTON_MEV) for i in range(P)]
        + [(positions[P + j], M_n, M_NEUTRON_MEV) for j in range(N)]
    )
    net_bound = HorizonNetwork(nodes, lattice_base_m, algebra=algebra)
    E_bound = net_bound.total_energy()
    theta_bound = net_bound.effective_theta_array()
    return (float(E_free - E_bound), theta_bound)


def _binding_energy_via_algebra(
    P: int,
    N: int,
    lattice_base_m: float,
    algebra=None,
) -> Tuple[float, np.ndarray]:
    """
    Binding energy and per-nucleon bound Θ from phase-lifted fanoplane fusion (pure algebra).

    M12 = M1 + M2 + [M1,M2]_Δ; B = -tr([M1,M2]_Δ @ Δ); σ = ‖[M1,M2]_Δ‖_F.
    Uses build_nucleon_matrix_with_phase so tr(M@Δ) ≠ 0 (θ = (π/2) arctan(E/E_Pl) from axiom).
    Returns (B_mev, theta_bound_array).
    """
    from pyhqiv.entanglement import binding_energy_algebraic

    if P <= 0 and N <= 0:
        return (0.0, np.array([]))
    if algebra is None:
        from pyhqiv.algebra import OctonionHQIVAlgebra
        algebra = OctonionHQIVAlgebra(verbose=False)

    matrices = [build_nucleon_matrix_with_phase(True, lattice_base_m, algebra) for _ in range(P)]
    matrices += [build_nucleon_matrix_with_phase(False, lattice_base_m, algebra) for _ in range(N)]
    hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
    return binding_energy_algebraic(matrices, algebra.Delta, lattice_base_m, hbar_c_mev_m)


class NuclearConfig:
    """
    Single source of truth for a nuclide (P, N) in pure HQIV.
    All horizons from first principles: free = merge(quarks), bound = merge(nucleons).
    No preset constants, no opt-in paths.
    """

    def __init__(
        self,
        P: int,
        N: int,
        name: str = "",
        is_bare: bool = False,
        excitation_energy: float = 0.0,
        t_cmb: float = 2.725,
    ) -> None:
        self.P = int(P)
        self.N = int(N)
        self.A = self.P + self.N
        self.name = name or f"{self.P}-{self.A}"
        self.is_bare = is_bare
        self.excitation_energy = float(excitation_energy)
        self.t_cmb = float(t_cmb)

        const = get_hqiv_nuclear_constants(self.t_cmb)
        self._tau_tick = const["TAU_TICK_OBSERVER_S"]
        self._macro_scale = const["MACROSCOPIC_DECAY_SCALE"]
        self._phi_crit = const["PHI_CRIT_SI"]
        self._lattice_base = const["LATTICE_BASE_M"]

        self._theta_free_p, self._theta_free_n = _free_nucleon_thetas_m(self._lattice_base)
        from pyhqiv.subatomic import nucleon_energies_mev

        self._free_proton_energy_mev, self._free_neutron_energy_mev = nucleon_energies_mev(self.t_cmb)
        self._is_proton_bound = np.array([True] * self.P + [False] * self.N, dtype=bool)
        if self.A == 1:
            theta_free = self._theta_free_p if self.P == 1 else self._theta_free_n
            self._theta_bound = np.array([theta_free], dtype=float)
            self._binding_energy_mev = 0.0
        elif self.A > 1:
            self._binding_energy_mev, self._theta_bound = _binding_energy_via_network(
                self.P, self.N, self._lattice_base
            )
            if self.A <= 3:
                self._theta_bound = _quark_level_bound_thetas(
                    self.P, self.N, self._lattice_base
                )
        else:
            self._theta_bound = np.array([])
            self._binding_energy_mev = 0.0
        _, theta_alpha_arr = _binding_energy_via_network(2, 2, self._lattice_base)
        self._theta_alpha = float(theta_alpha_arr[0]) if len(theta_alpha_arr) > 0 else self._lattice_base * 8.0

        self._E_info_mev = self._compute_E_info_mev()

    def _compute_E_info_mev(self) -> float:
        """E_info = ħc Σ(1/Θ_i) + excitation (MeV)."""
        if self.A <= 0 or len(self._theta_bound) == 0:
            return self.excitation_energy
        hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
        tension = hbar_c_mev_m * np.sum(1.0 / self._theta_bound)
        return float(tension + self.excitation_energy)

    def theta_stable_m(self) -> float:
        """Effective stable horizon (harmonic mean of bound Θ — direct from 1/Θ energy weighting)."""
        if self.A <= 0 or len(self._theta_bound) == 0:
            return self._lattice_base * 8.0  # so(8) dimension, degenerate case
        return float(self.A / np.sum(1.0 / self._theta_bound))

    def theta_unstable_m(self, source: Optional[str] = None) -> float:
        """Maximum-tension horizon (min Θ), optionally restricted to convertible nucleons."""
        if self.A <= 0 or len(self._theta_bound) == 0:
            return self._lattice_base * 8.0
        if source == "β-":
            mask = ~self._is_proton_bound
            if np.any(mask):
                return float(np.min(self._theta_bound[mask]))
        if source == "β+":
            mask = self._is_proton_bound
            if np.any(mask):
                return float(np.min(self._theta_bound[mask]))
        return float(np.min(self._theta_bound))

    def phi_si(self) -> float:
        """φ = 2 c² / Θ_avg (m²/s²). Paper: φ(x) = 2c²/Θ_local(x)."""
        theta_avg = (self.theta_unstable_m() + self.theta_stable_m()) / 2.0
        return 2.0 * (C_SI**2) / max(theta_avg, 1e-30)

    def _lapse_f(self) -> float:
        """
        Modified inertia f(a_loc, φ) = a_loc / (a_loc + φ/6). Paper particle action.

        Use acceleration scale at the nucleus horizon: a_loc = c²/Θ_avg (so φ = 2a_loc).
        When φ is large (compact nucleus), f < 1 → effective thermal energy for barrier crossing is reduced.
        """
        theta_avg = (self.theta_unstable_m() + self.theta_stable_m()) / 2.0
        theta_avg = max(theta_avg, 1e-30)
        a_loc = (C_SI**2) / theta_avg
        phi = 2.0 * (C_SI**2) / theta_avg
        return float(f_inertia(a_loc, phi))

    @property
    def binding_energy_mev(self) -> float:
        return self._binding_energy_mev

    @property
    def E_info_mev(self) -> float:
        return self._E_info_mev

    # ── Snap energy calculations (pure ΔE = ħc (1/Θ_u – 1/Θ_d …)) ──
    def _delta_E_mev(
        self,
        theta_daughter_stable: float,
        theta_cluster: float = None,
        theta_source: Optional[float] = None,
        conversion_gap_mev: float = 0.0,
    ) -> float:
        """General ΔE_info = ħc (1/Θ_unstable – 1/Θ_daughter – 1/Θ_cluster if present)."""
        theta_u = theta_source if theta_source is not None else self.theta_unstable_m()
        hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
        de = conversion_gap_mev + hbar_c_mev_m * (
            1.0 / max(theta_u, 1e-30) - 1.0 / max(theta_daughter_stable, 1e-30)
        )
        if theta_cluster is not None:
            de -= hbar_c_mev_m / max(theta_cluster, 1e-30)
        return max(0.0, float(de))

    # ── Allowed relational snaps ──
    def allowed_snaps(self) -> List[Tuple["NuclearConfig", float, str]]:
        snaps = []
        theta_stable = self.theta_stable_m()
        # β⁻: A==1 (free neutron) or quark-driven light nuclei (A≤3) or one neutron with lower Θ
        theta_beta_minus = self.theta_unstable_m("β-")
        has_beta_minus_weak_site = (
            self.A == 1
            or (self.A <= 3 and self.N > 0)  # quark-level: decay driven by quark interactions
            or theta_beta_minus < theta_stable * (1.0 - 1e-9)
        )
        if self.N > 0 and has_beta_minus_weak_site:
            dau = NuclearConfig(self.P + 1, self.N - 1, t_cmb=self.t_cmb)
            de = self._delta_E_mev(
                dau.theta_stable_m(),
                theta_source=theta_beta_minus,
                conversion_gap_mev=self._free_neutron_energy_mev - self._free_proton_energy_mev,
            )
            if de > 0:
                snaps.append((dau, de, "β-"))

        # β⁺
        theta_beta_plus = self.theta_unstable_m("β+")
        has_beta_plus_weak_site = self.A == 1 or theta_beta_plus < theta_stable * (1.0 - 1e-9)
        if self.P > 0 and has_beta_plus_weak_site:
            dau = NuclearConfig(self.P - 1, self.N + 1, t_cmb=self.t_cmb)
            de = self._delta_E_mev(
                dau.theta_stable_m(),
                theta_source=theta_beta_plus,
                conversion_gap_mev=self._free_proton_energy_mev - self._free_neutron_energy_mev,
            )
            if de > 0:
                snaps.append((dau, de, "β+"))

        # α
        if self.P >= 2 and self.N >= 2:
            de = self._delta_E_mev(NuclearConfig(self.P - 2, self.N - 2, t_cmb=self.t_cmb).theta_stable_m(),
                                   self._theta_alpha)
            if de > 0:
                dau = NuclearConfig(self.P - 2, self.N - 2, t_cmb=self.t_cmb)
                snaps.append((dau, de, "α"))

        # Spontaneous fission (symmetric split)
        if self.A >= 60:
            frag = NuclearConfig(self.P // 2, self.N // 2, t_cmb=self.t_cmb)
            de = self._delta_E_mev(frag.theta_stable_m())   # approximate for two fragments
            if de > 0:
                snaps.append(((self.P // 2, self.N // 2), de, "SF"))

        # Cluster (¹⁴C example — add more as needed)
        if self.P >= 6 and self.N >= 8 and self.A >= 180:
            de = self._delta_E_mev(
                NuclearConfig(self.P - 6, self.N - 8, t_cmb=self.t_cmb).theta_stable_m(),
                NuclearConfig(6, 8, t_cmb=self.t_cmb).theta_stable_m()
            )
            if de > 0:
                dau = NuclearConfig(self.P - 6, self.N - 8, t_cmb=self.t_cmb)
                snaps.append((dau, de, "cluster(¹⁴C)"))

        return snaps

    def snap_probability(self, delta_E_info_mev: float) -> float:
        """
        Core HQIV snap probability. Paper: Boltzmann × φ-damping × modified inertia.

        P_snap = exp(-ΔE / kT_eff) × (φ/(φ+φ_crit)), with kT_eff = ħc/Θ × f(a_loc, φ).
        Modified inertia f = a_loc/(a_loc+φ/6) reduces effective thermal energy when φ is large
        (compact nucleus), so barrier crossing is harder — same f as in particle action S = -m c ∫ f ds.
        """
        if delta_E_info_mev <= 0:
            return 0.0
        theta_avg = (self.theta_unstable_m() + self.theta_stable_m()) / 2.0
        theta_avg = max(theta_avg, 1e-30)
        hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
        kT_horizon = hbar_c_mev_m / theta_avg
        f = self._lapse_f()
        kT_eff = kT_horizon * max(f, 0.01)
        boltzmann = np.exp(-delta_E_info_mev / kT_eff)
        damping = self.phi_si() / max(self.phi_si() + self._phi_crit, 1e-30)
        return float(boltzmann * damping)

    def decay_rate_per_s(self) -> float:
        """
        Total macroscopic decay constant. Paper: λ_obs = (P_snap/τ_tick) × f / scale.

        Observer-time rate includes lapse f (S_particle = -m c ∫ f ds ⇒ dτ = f dt for rate scaling).
        """
        snaps = self.allowed_snaps()
        if not snaps:
            return 0.0
        total_p = sum(self.snap_probability(de) for _, de, _ in snaps)
        f = self._lapse_f()
        lam_raw = total_p / max(self._tau_tick, 1e-50)
        return lam_raw * f / max(self._macro_scale, 1e-30)

    def half_life_s(self) -> Optional[float]:
        """Half-life in seconds (None if stable)."""
        lam = self.decay_rate_per_s()
        return np.log(2.0) / lam if lam > 0 and np.isfinite(lam) else None


# ── Minimal public API (just thin wrappers around NuclearConfig) ──
def delta_E_info_mev(theta_unstable_m: float, theta_stable_m: float) -> float:
    """ΔE_info = ħc (1/Θ_unstable – 1/Θ_stable) in MeV. Θ in metres."""
    if theta_unstable_m <= 0 or theta_stable_m <= 0:
        return 0.0
    hbar_c_mev_m = HBAR_C_MEV_FM * 1e-15
    return hbar_c_mev_m * (1.0 / theta_unstable_m - 1.0 / theta_stable_m)


def binding_energy_mev(P: int, N: int, t_cmb: float = 2.725) -> float:
    """Nuclear binding energy B = E_free − E_bound (MeV). First principles only (merge path)."""
    return NuclearConfig(P, N, t_cmb=t_cmb).binding_energy_mev


def binding_energy_mev_algebraic(P: int, N: int, t_cmb: float = 2.725) -> float:
    """Binding energy (MeV) from phase-lifted fanoplane fusion only (no positions/radii)."""
    const = get_hqiv_nuclear_constants(t_cmb)
    return _binding_energy_via_algebra(P, N, const["LATTICE_BASE_M"], None)[0]


def theta_nuclear_stable_m(P: int, N: int, t_cmb: float = 2.725) -> float:
    return NuclearConfig(P, N, t_cmb=t_cmb).theta_stable_m()


def theta_nuclear_unstable_m(P: int, N: int, t_cmb: float = 2.725) -> float:
    return NuclearConfig(P, N, t_cmb=t_cmb).theta_unstable_m()


def half_life_nuclide_hqiv(P: int, N: int, t_cmb: float = 2.725) -> Optional[float]:
    return NuclearConfig(P, N, t_cmb=t_cmb).half_life_s()


def decay_chain(P: int, N: int, max_steps: int = 20, t_cmb: float = 2.725) -> List[Tuple[int, int]]:
    chain = [(P, N)]
    current = NuclearConfig(P, N, t_cmb=t_cmb)
    for _ in range(max_steps - 1):
        snaps = current.allowed_snaps()
        if not snaps:
            break
        probs = [current.snap_probability(de) for _, de, _ in snaps]
        idx = int(np.argmax(probs))
        daughter, _, _ = snaps[idx]
        if isinstance(daughter, tuple):
            chain.append(daughter)
            current = NuclearConfig(*daughter, t_cmb=t_cmb)
        else:
            chain.append((daughter.P, daughter.N))
            current = daughter
    return chain


def decay_chain_nuclide_hqiv(
    P: int, N: int, max_steps: int = 20, t_cmb: float = 2.725
) -> Tuple[Optional[float], str, Optional[int], Optional[int], List[Tuple[int, int]]]:
    cfg = NuclearConfig(P, N, t_cmb=t_cmb)
    t_half = cfg.half_life_s()
    chain = decay_chain(P, N, max_steps, t_cmb)
    if len(chain) < 2:
        return t_half, "stable", None, None, chain
    dP, dN = chain[1]
    snaps = cfg.allowed_snaps()
    if not snaps:
        return t_half, "?", dP, dN, chain
    probs = [cfg.snap_probability(de) for _, de, _ in snaps]
    idx = int(np.argmax(probs))
    mode = snaps[idx][2]
    return t_half, mode, dP, dN, chain


# ── Optional pint for public API unit conversion (internal units stay natural: s, MeV) ──
try:
    from pint import UnitRegistry
    _ureg = UnitRegistry()
    _ureg.default_format = "~P"
except ImportError:
    _ureg = None


class Nuclide:
    """
    User-facing wrapper around HQIV NuclearConfig.
    Accepts natural-language identifiers; returns quantities in human-readable units when pint is installed.
    Internal package units are natural (s, MeV); unit conversion happens at this API boundary.
    """

    def __init__(
        self,
        identifier: Union[str, int, Tuple[int, int]],
        t_cmb: float = 2.725,
        neutrino_flux: float = 6.5e10,
    ) -> None:
        self.t_cmb = float(t_cmb)
        self.neutrino_flux = float(neutrino_flux)
        P, N = self._parse_identifier(identifier)
        self.P = P
        self.N = N
        self.A = P + N
        self.symbol = self._get_symbol(P) or f"Z{P}"
        self._cfg = NuclearConfig(
            P, N,
            name=f"{self.symbol}-{self.A}",
            t_cmb=self.t_cmb,
        )

    @staticmethod
    def _parse_identifier(s: Union[str, int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(s, tuple) and len(s) == 2:
            return int(s[0]), int(s[1])
        if isinstance(s, int):
            A = s
            for sym, (p, ndef) in _ELEMENT_PN.items():
                if p + ndef == A:
                    return p, ndef
            raise ValueError(f"Unknown mass number {A}. Use e.g. 'U-238' or (92, 146).")
        s_clean = str(s).strip().replace(" ", "")
        # P-A with hyphen: "92-238"
        m = re.match(r"^(\d+)-(\d+)$", s_clean)
        if m:
            P, A = int(m.group(1)), int(m.group(2))
            N = A - P
            if N >= 0:
                return P, N
        s = s_clean.lower().replace("-", "")
        # Element + mass: "u238", "238u", "helium4", "carbon14"
        m = re.match(r"^(\d+)?([a-z]+)(\d+)?$", s)
        if m:
            num1, elem, num2 = m.groups()
            mass = int(num1 or num2 or 0)
            sym = _ELEMENT_NAME_TO_SYMBOL.get(elem, elem.title())
            P, N_default = _ELEMENT_PN.get(sym, (0, 0))
            if mass == 0:
                mass = P + N_default
            N = mass - P
            if N >= 0:
                return P, N
        raise ValueError(
            f"Could not parse nuclide '{s}'. Try 'U-238', 'He-4', 'carbon-14', 238, or (P, N)."
        )

    @staticmethod
    def _get_symbol(Z: int) -> Optional[str]:
        for sym, (p, _) in _ELEMENT_PN.items():
            if p == Z:
                return sym
        return None

    @property
    def half_life(self):
        """Half-life. With pint: Quantity (e.g. .to('yr')); without pint: float seconds or inf."""
        hl_s = self._cfg.half_life_s()
        if hl_s is None or hl_s <= 0:
            if _ureg is not None:
                return _ureg.Quantity(float("inf"), "s")
            return float("inf")
        if _ureg is not None:
            return _ureg.Quantity(hl_s, "s")
        return hl_s

    @property
    def binding_energy(self):
        """Binding energy. With pint: Quantity (MeV); without pint: float MeV."""
        if _ureg is not None:
            return _ureg.Quantity(self._cfg.binding_energy_mev, "MeV")
        return self._cfg.binding_energy_mev

    @property
    def binding_energy_per_nucleon(self):
        """Binding energy per nucleon. With pint: Quantity; without pint: float MeV."""
        be = self.binding_energy
        if _ureg is not None and hasattr(be, "magnitude"):
            return be / self.A
        return be / self.A

    @property
    def E_info(self):
        """E_info from lattice. With pint: Quantity (MeV); without pint: float MeV."""
        if _ureg is not None:
            return _ureg.Quantity(self._cfg.E_info_mev, "MeV")
        return self._cfg.E_info_mev

    def decay_chain(self, max_steps: int = 20) -> List["Nuclide"]:
        """Chain of Nuclide instances (same t_cmb, neutrino_flux)."""
        raw = decay_chain(self.P, self.N, max_steps, t_cmb=self.t_cmb)
        sym = self._get_symbol
        return [
            Nuclide((p, n), t_cmb=self.t_cmb, neutrino_flux=self.neutrino_flux)
            for p, n in raw
        ]

    def __repr__(self) -> str:
        hl = self.half_life
        if _ureg is not None and hasattr(hl, "to"):
            yr = hl.to("yr").magnitude
        else:
            yr = hl / (365.25 * 24 * 3600) if hl != float("inf") else float("inf")
        unit = "yr"
        return f"<Nuclide {self.symbol}-{self.A}  t½≈{yr:.3g} {unit}>"


__all__ = [
    "nuclide_from_symbol",
    "NuclearConfig",
    "Nuclide",
    "delta_E_info_mev",
    "binding_energy_mev",
    "binding_energy_mev_algebraic",
    "build_nucleon_matrix_with_phase",
    "theta_nuclear_stable_m",
    "theta_nuclear_unstable_m",
    "half_life_nuclide_hqiv",
    "decay_chain",
    "decay_chain_nuclide_hqiv",
    "minimize_nucleon_configuration",
]