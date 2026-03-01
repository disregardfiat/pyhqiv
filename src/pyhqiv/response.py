"""
HQIV linear response: conductivity tensor from phase-horizon corrected Maxwell
and f(a, φ) inertia. Non-local conductivity from horizon monogamy (γ ≈ 0.40).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from pyhqiv.constants import C_SI, GAMMA


def compute_conductivity(
    omega: float,
    sigma_0: Union[float, np.ndarray],
    phi_avg: float = 0.0,
    gamma: float = GAMMA,
) -> Union[float, np.ndarray]:
    """
    Phase-horizon corrected conductivity: σ(ω) = σ_0(ω) * (1 + γ φ_avg / (ω + ε)).
    Linear-response (Kubo/Drude) baseline σ_0 modified by horizon factor.
    Non-local conductivity from horizon monogamy. omega in rad/s; phi_avg in same
    units as ω² (e.g. SI or natural).
    """
    sigma_0 = np.asarray(sigma_0, dtype=float)
    eps = 1e-30
    factor = 1.0 + gamma * phi_avg / (omega + eps)
    return sigma_0 * factor


def response_tensor_diagonal(
    omega: float,
    dim: int = 3,
    sigma_0: float = 1.0,
    phi_avg: float = 0.0,
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    Diagonal response tensor (conductivity) σ_ij = σ δ_ij with HQIV correction.
    Returns (dim, dim) array.
    """
    sigma = compute_conductivity(omega, sigma_0, phi_avg=phi_avg, gamma=gamma)
    return np.eye(dim) * sigma
