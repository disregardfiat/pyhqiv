"""Utility functions: grid generation, broadcasting, reproducibility, molecular Θ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

# Z (nuclear charge) by element symbol for molecular Θ
_ELEMENT_Z: dict = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "Fe": 26}
# Default mass (amu) per symbol for isotope scaling; C12 → 12, etc.
_ELEMENT_MASS_AMU: dict = {"H": 1, "C": 12, "N": 14, "O": 16, "S": 32, "P": 31, "Fe": 56}


def make_grid_3d(
    lo: Union[float, Tuple[float, float, float]],
    hi: Union[float, Tuple[float, float, float]],
    n: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Return (N, 3) array of grid points. lo/hi are bounds per axis; n is count per axis.
    """
    if np.isscalar(lo):
        lo = (lo, lo, lo)
    if np.isscalar(hi):
        hi = (hi, hi, hi)
    if np.isscalar(n):
        n = (n, n, n)
    mg = np.mgrid[
        lo[0] : hi[0] : complex(n[0]),
        lo[1] : hi[1] : complex(n[1]),
        lo[2] : hi[2] : complex(n[2]),
    ]
    return mg.reshape(3, -1).T


def local_theta_from_distance(r: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Θ_local ∝ r (or scale); φ = 2c²/Θ_local. Simple radial model."""
    r = np.asarray(r, dtype=float)
    return np.maximum(np.linalg.norm(r, axis=-1), 1e-30) * scale


def phi_from_theta_local(theta_local: np.ndarray, c: float = 1.0) -> np.ndarray:
    """φ(x) = 2c²/Θ_local(x). c in natural units can be 1."""
    return 2.0 * (c**2) / np.maximum(np.asarray(theta_local), 1e-30)


def set_seed(seed: Optional[int] = 42) -> None:
    """Set NumPy and Python RNG seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)


# -----------------------------------------------------------------------------
# Molecular / diamond lattice (PROtien: Θ from Z and coordination)
# -----------------------------------------------------------------------------
# Θ(Z, coord) uses one reference (theta_ref_ang) and paper α; bond_length_from_theta
# is purely HQIV (r_eq = min(Θ_i, Θ_j) × monogamy_factor). No empirical bond constants.
#
# Reference length can be derived from local conditions via theta_ref_from_environment
# (ρ, T, M → mean interparticle spacing in Å), so coupling length is not fixed 1.53 Å.

# Metres to Å for environment-derived Θ
_M_TO_ANG: float = 1e10


def theta_ref_from_environment(
    rho_kg_m3: Union[float, np.ndarray],
    molar_mass_kg: float,
    T_K: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Reference horizon scale Θ_ref (Å) from local conditions — no empirical bond length.

    Same axiom as thermo: Θ = (M / (ρ N_A))^{1/3} (mean interparticle spacing).
    Optional T softens at high T. Use this when analyzing different places (different
    P, T, phase) so the coupling length is derived, not fixed.

    Parameters
    ----------
    rho_kg_m3 : float or array
        Mass density in kg/m³.
    molar_mass_kg : float
        Molar mass in kg/mol.
    T_K : float or array, optional
        Temperature in K; if provided, Θ gets a mild T-dependent factor.

    Returns
    -------
    Θ_ref in Å. Use as theta_ref_ang in theta_local() / theta_for_atom().
    """
    from pyhqiv.constants import N_A

    try:
        from pyhqiv.constants import T_PL_K
    except ImportError:
        T_PL_K = 1.416808e32  # fallback

    rho = np.maximum(np.asarray(rho_kg_m3, dtype=float), 1e-30)
    n_m3 = rho * N_A / molar_mass_kg
    L_m = (1.0 / np.maximum(n_m3, 1e-30)) ** (1.0 / 3.0)
    if T_K is not None:
        T = np.asarray(T_K, dtype=float)
        L_m = L_m * (1.0 + 0.1 * np.sqrt(np.minimum(T / T_PL_K, 1.0)))
    return np.maximum(L_m, 1e-30) * _M_TO_ANG


def theta_local(
    z_shell: int,
    coordination: int = 1,
    alpha: float = 0.91,
    theta_ref_ang: float = 1.53,
    mass_amu: Optional[float] = None,
    mass_ref_amu: float = 12.0,
) -> float:
    """
    Diamond size Θ_local (Å) at a lattice node from nuclear charge and monogamy.

    Θ = theta_ref * (6^alpha * 2^(1/3)) * Z^{-alpha} / coordination^{1/3}.
    If mass_amu is given, Θ is scaled by (mass_amu / mass_ref_amu)^{1/3} so that
    heavier isotopes (e.g. C14) get a larger Θ and fold differently than C12.

    Parameters
    ----------
    z_shell : int
        Nuclear charge (e.g. C=6, N=7, O=8).
    coordination : int
        Coordination number (monogamy).
    alpha : float
        Exponent for Z scaling (default 0.91).
    theta_ref_ang : float
        Reference length (Å). Default 1.53 matches carbon at coord=2 under standard
        conditions. For conditions in different places use theta_ref_from_environment(ρ, T, M).
    mass_amu : float, optional
        Mass in amu (e.g. 14 for C14). If given, isotope-dependent Θ.
    mass_ref_amu : float
        Reference mass for scaling (default 12 for carbon). Used when mass_amu is set.

    Returns
    -------
    float
        Θ in Å.
    """
    z = max(1, int(z_shell))
    coord = max(1, int(coordination))
    norm = (6.0**alpha) * (2.0 ** (1.0 / 3.0))
    theta = theta_ref_ang * norm * (z ** (-alpha)) / (coord ** (1.0 / 3.0))
    if mass_amu is not None and mass_ref_amu > 0:
        theta = theta * (float(mass_amu) / mass_ref_amu) ** (1.0 / 3.0)
    return float(theta)


def theta_for_atom(
    symbol: str,
    coordination: int = 1,
    alpha: float = 0.91,
    theta_ref_ang: float = 1.53,
    mass_amu: Optional[float] = None,
) -> float:
    """
    Θ_local (Å) for an element by symbol. Uses Z map: H=1, C=6, N=7, O=8, S=16, P=15, Fe=26.

    Optional mass_amu (e.g. 14 for C14-alpha) makes Θ isotope-dependent so that
    C14-alpha folds differently than C12-alpha (heavier → larger Θ → different
    bond length and torsion landscape).

    Parameters
    ----------
    symbol : str
        Element symbol (e.g. "C", "N", "O").
    coordination : int
        Coordination number.
    alpha, theta_ref_ang
        As in theta_local().
    mass_amu : float, optional
        Mass in amu (e.g. 14 for C14). If None, uses default for symbol (e.g. 12 for C).

    Returns
    -------
    float
        Θ in Å.
    """
    sym = symbol.strip().title()
    z = _ELEMENT_Z.get(sym, 6)  # default C
    mass_ref = _ELEMENT_MASS_AMU.get(sym, 12.0)
    return theta_local(
        z,
        coordination,
        alpha=alpha,
        theta_ref_ang=theta_ref_ang,
        mass_amu=mass_amu,
        mass_ref_amu=mass_ref,
    )


# -----------------------------------------------------------------------------
# Universal nuclide API: (P, N) only — no element symbols or mass tables.
# Θ, radius, coupling scale, and decay stubs from proton and neutron count.
# -----------------------------------------------------------------------------


def theta_local_nuclide(
    P: int,
    N: int,
    coordination: int = 1,
    alpha: float = 0.91,
    theta_ref_ang: float = 1.53,
) -> float:
    """
    Θ_local (Å) for nuclide (P, N) from bound energy: Θ = ħc/B. No radius constant.

    B(P,N) is computed via semi-empirical mass formula in the nuclear module;
    horizon scale is set directly by the binding energy of the system.
    coordination, alpha, theta_ref_ang are ignored (kept for API compatibility).
    """
    try:
        from pyhqiv.nuclear import theta_nuclear_stable_m
        theta_m = theta_nuclear_stable_m(int(P), int(N))
        return float(theta_m * _M_TO_ANG)
    except Exception:
        # Fallback if nuclear not available: same formula as before
        p, n = max(1, int(P)), max(0, int(N))
        coord = max(1, int(coordination))
        norm = (6.0**alpha) * (2.0 ** (1.0 / 3.0))
        mass_factor = (1.0 + n / max(p, 1)) ** (1.0 / 3.0)
        return float(theta_ref_ang * norm * (p ** (-alpha)) / (coord ** (1.0 / 3.0)) * mass_factor)


def radius_nuclide(
    P: int,
    N: int,
    coordination: int = 1,
    alpha: float = 0.91,
    theta_ref_ang: float = 1.53,
) -> float:
    """
    Characteristic radius (Å) for nuclide (P, N). Θ = ħc/B from binding energy.

    Same as theta_local_nuclide: horizon scale derived from B(P,N), not constants.
    """
    return theta_local_nuclide(P, N, coordination=coordination, alpha=alpha, theta_ref_ang=theta_ref_ang)


@dataclass
class NuclideDecay:
    """Placeholder for half-life and decay chain from (P, N)."""

    P: int
    N: int
    half_life_seconds: Optional[float]  # None = stable or unknown
    decay_mode: str  # e.g. "stable", "β-", "β+", "α", "SF", "unknown"
    daughter_P: Optional[int]
    daughter_N: Optional[int]
    decay_chain: List[Tuple[int, int]]  # [(P,N), ...] until stable


def half_life_nuclide(P: int, N: int) -> Optional[float]:
    """
    Half-life (seconds) for nuclide (P, N) from HQIV first-principles decay.

    P_snap = exp(– ΔE_info/(ħc/Θ_avg)) × (φ/(φ+φ_crit)); λ = P_snap/τ_tick;
    t_1/2 = ln(2)/λ with macroscopic lapse scaling. Returns None if stable.
    """
    try:
        from pyhqiv.nuclear import half_life_nuclide_hqiv
        return half_life_nuclide_hqiv(int(P), int(N))
    except Exception:
        return None


def decay_chain_nuclide(
    P: int,
    N: int,
    max_steps: int = 20,
) -> NuclideDecay:
    """
    Decay chain for nuclide (P, N) from HQIV snap formula.

    Returns NuclideDecay with half_life_seconds, decay_mode, daughter (P,N),
    and decay_chain list. All from (P, N) and paper constants.
    """
    try:
        from pyhqiv.nuclear import decay_chain_nuclide_hqiv
        t_half, mode, dP, dN, chain = decay_chain_nuclide_hqiv(int(P), int(N), max_steps=max_steps)
        return NuclideDecay(
            P=int(P),
            N=int(N),
            half_life_seconds=t_half,
            decay_mode=mode,
            daughter_P=dP,
            daughter_N=dN,
            decay_chain=chain,
        )
    except Exception:
        return NuclideDecay(
            P=int(P),
            N=int(N),
            half_life_seconds=None,
            decay_mode="stable",
            daughter_P=None,
            daughter_N=None,
            decay_chain=[(int(P), int(N))],
        )


def bond_length_from_theta(
    theta_i: float,
    theta_j: float,
    monogamy_factor: float = 1.0,
) -> float:
    """
    Equilibrium separation (Å) between two nodes: r_eq = min(Θ_i, Θ_j) * monogamy_factor.

    Causal diamond containing both atoms (paper lattice / monogamy).

    Parameters
    ----------
    theta_i, theta_j : float
        Θ at each node (Å).
    monogamy_factor : float
        Scale factor (default 1.0).

    Returns
    -------
    float
        r_eq in Å.
    """
    return float(min(theta_i, theta_j) * monogamy_factor)


def damping_force_magnitude(
    phi: Union[float, np.ndarray],
    grad_phi: Union[float, np.ndarray],
    a_loc: float = 1.0,
    gamma: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    |f_φ| = γ * φ * |∇φ| / (a_loc + φ/6)² (paper geometric damping).

    From modified inertia f(a_loc, φ) = a_loc / (a_loc + φ/6) and horizon term.
    Units: φ and a_loc consistent (e.g. φ = 2/Θ with Θ in Å); grad_phi = |∇φ|; return = force magnitude.

    Parameters
    ----------
    phi : float or array
        Auxiliary field φ at point(s).
    grad_phi : float or array
        |∇φ| at point(s).
    a_loc : float
        Local acceleration scale in denominator (default 1.0).
    gamma : float, optional
        Thermodynamic coefficient; default from pyhqiv.constants.GAMMA (0.40).

    Returns
    -------
    float or array
        |f_φ| magnitude.
    """
    if gamma is None:
        from pyhqiv.constants import GAMMA

        gamma = GAMMA
    phi = np.asarray(phi, dtype=float)
    grad_phi = np.asarray(grad_phi, dtype=float)
    denom = np.maximum(a_loc + phi / 6.0, 1e-30)
    return gamma * phi * np.abs(grad_phi) / (denom**2)
