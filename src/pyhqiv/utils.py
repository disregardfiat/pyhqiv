"""Utility functions: grid generation, broadcasting, reproducibility, molecular Θ."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# Z (nuclear charge) by element symbol for molecular Θ
_ELEMENT_Z: dict = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "Fe": 26}


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
    return 2.0 * (c ** 2) / np.maximum(np.asarray(theta_local), 1e-30)


def set_seed(seed: Optional[int] = 42) -> None:
    """Set NumPy and Python RNG seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)


# -----------------------------------------------------------------------------
# Molecular / diamond lattice (PROtien: Θ from Z and coordination)
# -----------------------------------------------------------------------------


def theta_local(
    z_shell: int,
    coordination: int = 1,
    alpha: float = 0.91,
    theta_ref_ang: float = 1.53,
) -> float:
    """
    Diamond size Θ_local (Å) at a lattice node from nuclear charge and monogamy.

    Θ = theta_ref * (6^alpha * 2^(1/3)) * Z^{-alpha} / coordination^{1/3}.
    Reference set so that Θ_C(coord=2) ≈ 1.53 Å, Θ_N(coord=2) ≈ 1.33 Å.

    Parameters
    ----------
    z_shell : int
        Nuclear charge (e.g. C=6, N=7, O=8).
    coordination : int
        Coordination number (monogamy).
    alpha : float
        Exponent for Z scaling (default 0.91).
    theta_ref_ang : float
        Reference length (Å); Θ_C(coord=2) = theta_ref_ang (default 1.53).

    Returns
    -------
    float
        Θ in Å.
    """
    z = max(1, int(z_shell))
    coord = max(1, int(coordination))
    norm = (6.0 ** alpha) * (2.0 ** (1.0 / 3.0))
    return theta_ref_ang * norm * (z ** (-alpha)) / (coord ** (1.0 / 3.0))


def theta_for_atom(
    symbol: str,
    coordination: int = 1,
    alpha: float = 0.91,
    theta_ref_ang: float = 1.53,
) -> float:
    """
    Θ_local (Å) for an element by symbol. Uses Z map: H=1, C=6, N=7, O=8, S=16, P=15, Fe=26.

    Parameters
    ----------
    symbol : str
        Element symbol (e.g. "C", "N", "O").
    coordination : int
        Coordination number.
    alpha, theta_ref_ang
        As in theta_local().

    Returns
    -------
    float
        Θ in Å.
    """
    z = _ELEMENT_Z.get(symbol.strip().title(), 6)  # default C
    return theta_local(z, coordination, alpha=alpha, theta_ref_ang=theta_ref_ang)


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
    return gamma * phi * np.abs(grad_phi) / (denom ** 2)
