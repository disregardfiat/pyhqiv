"""
HQIV phase lift: δθ′(E′), ˙δθ′, homogeneous limit ˙δθ′≈H, ADM lapse compression,
and modified Maxwell lift terms (γ(φ/c²)(˙δθ′/c)). Paper Sec. 2 & 5.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import C_SI, GAMMA, LAPSE_COMPRESSION_PAPER


def delta_theta_prime(E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Phase-horizon angle δθ′(E′) = arctan(E′)×(π/2).
    E′ is normalized energy in [0, 1] (or same units as scale); paper Sec. 2.
    """
    return np.arctan(np.asarray(E_prime, dtype=float)) * (math.pi / 2.0)


def delta_theta_prime_dot_homogeneous(H: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Homogeneous limit: ˙δθ′ ≈ H (natural units). In SI: ˙δθ′ has units 1/s.
    """
    return np.asarray(H, dtype=float)


def adm_lapse_compression_factor(
    phi_over_c2: Union[float, np.ndarray],
    delta_theta_dot_over_c: Union[float, np.ndarray],
    gamma: float = GAMMA,
) -> Union[float, np.ndarray]:
    """
    Effective lapse factor from γ(φ/c²)(˙δθ′/c) term.
    Homogeneous: φ = cH, ˙δθ′ = H ⇒ γ H²/c²; compression factor ≈ 3.96 (paper).
    """
    phi = np.asarray(phi_over_c2, dtype=float)
    dtdc = np.asarray(delta_theta_dot_over_c, dtype=float)
    return 1.0 + gamma * phi * dtdc


def apparent_age_from_wall_clock(
    age_wall_yr: Union[float, np.ndarray],
    lapse_compression: float = LAPSE_COMPRESSION_PAPER,
) -> Union[float, np.ndarray]:
    """Apparent age (local chronometers) from wall-clock age and lapse compression."""
    return np.asarray(age_wall_yr, dtype=float) / lapse_compression


class HQIVPhaseLift:
    """
    Phase-horizon lift: δθ′(E′), ˙δθ′ = u^μ ∇_μ δθ′, homogeneous ˙δθ′≈H,
    ADM lapse compression, and modified Maxwell terms.
    """

    def __init__(self, gamma: float = GAMMA, c_si: float = C_SI) -> None:
        self.gamma = gamma
        self.c_si = c_si

    def delta_theta_prime(self, E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """δθ′(E′) = arctan(E′)×(π/2)."""
        return delta_theta_prime(E_prime)

    def delta_theta_prime_dot(
        self,
        u_mu: Optional[np.ndarray] = None,
        grad_delta_theta: Optional[np.ndarray] = None,
        H_homogeneous: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """
        ˙δθ′ = u^μ ∇_μ δθ′. If grad_delta_theta and u_mu not provided, use homogeneous limit ˙δθ′ ≈ H.
        """
        if H_homogeneous is not None:
            return delta_theta_prime_dot_homogeneous(H_homogeneous)
        if u_mu is not None and grad_delta_theta is not None:
            return np.dot(np.asarray(u_mu).ravel(), np.asarray(grad_delta_theta).ravel())
        raise ValueError("Provide either H_homogeneous or (u_mu, grad_delta_theta)")

    def lapse_compression(
        self,
        phi_local: Union[float, np.ndarray],
        delta_theta_dot: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Lapse factor from γ(φ/c²)(˙δθ′/c). phi_local in (m/s²) or natural; delta_theta_dot in 1/s."""
        phi_over_c2 = np.asarray(phi_local, dtype=float) / (self.c_si ** 2)
        dtdc = np.asarray(delta_theta_dot, dtype=float) / self.c_si
        return adm_lapse_compression_factor(phi_over_c2, dtdc, gamma=self.gamma)

    def maxwell_lift_coefficient(
        self,
        phi_over_c2: Union[float, np.ndarray],
        delta_theta_dot_over_c: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Coefficient γ(φ/c²)(˙δθ′/c) for modified Maxwell D/Dt = ∂/∂t′ + ˙δθ′ ∂/∂δθ′."""
        return self.gamma * np.asarray(phi_over_c2) * np.asarray(delta_theta_dot_over_c)
