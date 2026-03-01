"""
Modified Navier-Stokes (HQIV fluid): f(a_loc, φ), g_vac, ν_eddy.

From the paper (Ettinger, Feb 2026): modified inertia f = a_loc/(a_loc + φ/6),
vacuum source g_vac = -γ ∇(φ δ̇θ′)/6, and eddy viscosity ν_eddy = γ Θ_local |δ̇θ′| ℓ_coh² C.
Laminar limit |a| ≫ φ/6 → f→1, g_vac→0 → standard Navier-Stokes.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from pyhqiv.constants import GAMMA


def f_inertia(
    a_loc: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    f_min: float = 0.01,
) -> Union[float, np.ndarray]:
    """
    Modified inertia factor f(a_loc, φ) = a_loc / (a_loc + φ/6). Paper particle action.

    In momentum equation: ρ f Dv/Dt = rhs ⇒ Dv/Dt = rhs / (ρ f).
    Laminar limit |a| ≫ φ/6 ⇒ f → 1.

    Parameters
    ----------
    a_loc : float or array
        Magnitude of local acceleration |a| (or scale).
    phi : float or array
        Auxiliary field φ = 2c²/Θ_local.
    f_min : float
        Floor for f (default 0.01) to avoid division by zero.

    Returns
    -------
    float or array
        f ∈ [f_min, 1].
    """
    a = np.asarray(a_loc, dtype=float)
    p = np.asarray(phi, dtype=float)
    denom = np.maximum(a + p / 6.0, 1e-30)
    f = a / denom
    return np.maximum(np.minimum(f, 1.0), f_min)


def g_vac_vector(
    phi: Union[float, np.ndarray],
    dot_delta_theta: Union[float, np.ndarray],
    grad_phi: np.ndarray,
    grad_dot_delta_theta: np.ndarray,
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    Vacuum source g_vac = -γ ∇(φ δ̇θ′) / 6 (per unit mass) for momentum equation.

    ∇(φ δ̇θ′) = φ ∇δ̇θ′ + δ̇θ′ ∇φ. So g_vac = -γ/6 * (φ * grad_dot_delta_theta + dot_delta_theta * grad_phi).

    Parameters
    ----------
    phi : float or array
        φ at point(s).
    dot_delta_theta : float or array
        δ̇θ′ at point(s).
    grad_phi : array
        ∇φ, shape (..., 3) or (3,).
    grad_dot_delta_theta : array
        ∇δ̇θ′, shape (..., 3) or (3,).

    Returns
    -------
    array
        g_vac vector, same shape as grad_phi.
    """
    phi = np.asarray(phi, dtype=float)
    dot = np.asarray(dot_delta_theta, dtype=float)
    g_phi = np.asarray(grad_phi, dtype=float)
    g_dot = np.asarray(grad_dot_delta_theta, dtype=float)
    # Ensure broadcastable: phi and dot can be scalars or (..., 1)
    term = phi * g_dot + dot * g_phi
    return (-gamma / 6.0) * term


def eddy_viscosity(
    Theta_local: Union[float, np.ndarray],
    dot_delta_theta: Union[float, np.ndarray],
    l_coh: Union[float, np.ndarray],
    coherence_factor: float = 1.0,
    gamma: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    HQIV eddy viscosity ν_eddy = γ Θ_local |δ̇θ′| ℓ_coh² C.

    τ_total = τ_mol + ρ ν_eddy S. High coherence (plasma): C≈1; entropic turbulence: C smaller.

    Parameters
    ----------
    Theta_local : float or array
        Local causal-horizon radius (Θ), same units as ℓ_coh.
    dot_delta_theta : float or array
        |δ̇θ′| (phase-lift clock, ≈ H in homogeneous limit).
    l_coh : float or array
        Coherence length (e.g. Debye length, or integral scale).
    coherence_factor : float
        C ∈ [0, 1]; default 1.0.
    gamma : float, optional
        From constants.GAMMA if None.

    Returns
    -------
    float or array
        ν_eddy (same units as Θ * (1/s) * length²).
    """
    if gamma is None:
        gamma = GAMMA
    Theta = np.asarray(Theta_local, dtype=float)
    dot = np.asarray(dot_delta_theta, dtype=float)
    lc = np.asarray(l_coh, dtype=float)
    return gamma * Theta * np.abs(dot) * (lc ** 2) * coherence_factor


def modified_momentum_rhs(
    grad_p: np.ndarray,
    div_tau_mol: np.ndarray,
    g_ext: np.ndarray,
    g_vac: np.ndarray,
    rho: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    RHS of modified momentum (before dividing by ρ f): -∇p/ρ + ∇·τ/ρ + g_ext + g_vac.

    Then a_modified = this_rhs / f (with f = f_inertia(|a|, φ)).
    """
    rho = np.asarray(rho, dtype=float)
    if rho.shape != grad_p.shape and np.ndim(rho) < np.ndim(grad_p):
        rho = np.broadcast_to(rho, grad_p.shape)
    return -grad_p / np.maximum(rho, 1e-30) + div_tau_mol / np.maximum(rho, 1e-30) + g_ext + g_vac
