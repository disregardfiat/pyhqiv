"""
Protocols and abstract base classes for extensible HQIV components.

Implement these protocols to plug in custom lattices or phase lifts
while remaining compatible with helpers that expect the standard interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Union

import numpy as np

from pyhqiv.constants import (
    AGE_APPARENT_GYR_PAPER,
    AGE_WALL_GYR_PAPER,
    LAPSE_COMPRESSION_PAPER,
    T_CMB_K,
)


class NullLatticeProtocol(Protocol):
    """
    Protocol for a discrete null lattice: shell temperatures, curvature imprint δE,
    mode counts, Ω_k^true, and evolve_to_cmb.
    Use this (or inherit from NullLatticeBase) for custom lattice implementations.
    """

    m_trans: int
    gamma: float
    alpha: float

    def shell_temperature(self, m: np.ndarray, E_0_factor: float = 1.0) -> np.ndarray:
        """T(m) for shell indices m."""
        ...

    def delta_E(self, m: np.ndarray, E_0_factor: float = 1.0) -> np.ndarray:
        """Curvature imprint δE(m) for shell indices m."""
        ...

    def mode_count_per_shell(self, m: np.ndarray) -> np.ndarray:
        """New modes per shell (vectorized over m)."""
        ...

    def omega_k_true(self, E_0_factor: float = 1.0, use_jax: bool = False) -> float:
        """Ω_k^true from shell integral."""
        ...

    def evolve_to_cmb(
        self,
        T0_K: float = T_CMB_K,
        E_0_factor: float = 1.0,
        use_jax: bool = False,
    ) -> Dict[str, Any]:
        """Evolve to CMB hypersurface; return dict with Omega_true_k, ages, lapse, etc."""
        ...

    def get_delta_E_grid(self, E_0_factor: float = 1.0) -> np.ndarray:
        """δE(m) for m = 0, 1, ..., m_trans-1."""
        ...

    def get_cumulative_mode_counts(self) -> np.ndarray:
        """Cumulative mode count at each shell 0..m_trans."""
        ...


class NullLatticeBase(ABC):
    """
    Abstract base class for custom null lattices.
    Subclass and implement the abstract methods to define alternative combinatorics
    or shell integrals while keeping the same interface as DiscreteNullLattice.
    """

    def __init__(
        self,
        m_trans: int,
        gamma: float,
        alpha: float,
        **kwargs: Any,
    ) -> None:
        self.m_trans = m_trans
        self.gamma = gamma
        self.alpha = alpha

    @abstractmethod
    def shell_temperature(self, m: np.ndarray, E_0_factor: float = 1.0) -> np.ndarray:
        """T(m) for shell indices m."""
        ...

    @abstractmethod
    def delta_E(self, m: np.ndarray, E_0_factor: float = 1.0) -> np.ndarray:
        """Curvature imprint δE(m) for shell indices m."""
        ...

    @abstractmethod
    def mode_count_per_shell(self, m: np.ndarray) -> np.ndarray:
        """New modes per shell (vectorized over m)."""
        ...

    @abstractmethod
    def omega_k_true(self, E_0_factor: float = 1.0, use_jax: bool = False) -> float:
        """Ω_k^true from shell integral."""
        ...

    def evolve_to_cmb(
        self,
        T0_K: float = T_CMB_K,
        E_0_factor: float = 1.0,
        use_jax: bool = False,
    ) -> Dict[str, Any]:
        """Default: use omega_k_true and paper values for ages/lapse."""
        omega_k = self.omega_k_true(E_0_factor=E_0_factor, use_jax=use_jax)
        return {
            "Omega_true_k": omega_k,
            "age_wall_Gyr": AGE_WALL_GYR_PAPER,
            "age_apparent_Gyr": AGE_APPARENT_GYR_PAPER,
            "lapse_compression": LAPSE_COMPRESSION_PAPER,
            "T_CMB_K": T0_K,
            "m_trans": self.m_trans,
            "gamma": self.gamma,
        }

    def get_delta_E_grid(self, E_0_factor: float = 1.0) -> np.ndarray:
        """δE(m) for m = 0, 1, ..., m_trans-1."""
        m_arr = np.arange(0, self.m_trans, dtype=float)
        return self.delta_E(m_arr, E_0_factor=E_0_factor)

    @abstractmethod
    def get_cumulative_mode_counts(self) -> np.ndarray:
        """Cumulative mode count at each shell 0..m_trans."""
        ...


class PhaseLiftProtocol(Protocol):
    """
    Protocol for a phase-horizon lift: δθ′(E′), ˙δθ′, lapse compression,
    and Maxwell lift coefficient.
    Use this (or inherit from PhaseLiftBase) for custom phase lifts.
    """

    gamma: float

    def delta_theta_prime(self, E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """δθ′(E′) = arctan(E′)×(π/2)."""
        ...

    def delta_theta_prime_dot(
        self,
        u_mu: Optional[np.ndarray] = None,
        grad_delta_theta: Optional[np.ndarray] = None,
        H_homogeneous: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """˙δθ′ = u^μ ∇_μ δθ′ or homogeneous limit ˙δθ′ ≈ H."""
        ...

    def lapse_compression(
        self,
        phi_local: Union[float, np.ndarray],
        delta_theta_dot: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Lapse factor from γ(φ/c²)(˙δθ′/c)."""
        ...

    def maxwell_lift_coefficient(
        self,
        phi_over_c2: Union[float, np.ndarray],
        delta_theta_dot_over_c: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Coefficient γ(φ/c²)(˙δθ′/c) for modified Maxwell."""
        ...


class PhaseLiftBase(ABC):
    """
    Abstract base class for custom phase lifts.
    Subclass and implement the abstract methods to define alternative
    δθ′(E′) or lapse relations.
    """

    def __init__(self, gamma: float, c_si: float = 2.99792458e8) -> None:
        self.gamma = gamma
        self.c_si = c_si

    @abstractmethod
    def delta_theta_prime(self, E_prime: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """δθ′(E′) (e.g. arctan(E′)×(π/2))."""
        ...

    @abstractmethod
    def delta_theta_prime_dot(
        self,
        u_mu: Optional[np.ndarray] = None,
        grad_delta_theta: Optional[np.ndarray] = None,
        H_homogeneous: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """˙δθ′ = u^μ ∇_μ δθ′ or homogeneous limit."""
        ...

    def lapse_compression(
        self,
        phi_local: Union[float, np.ndarray],
        delta_theta_dot: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Lapse factor from γ(φ/c²)(˙δθ′/c). Default implementation."""
        phi_over_c2 = np.asarray(phi_local, dtype=float) / (self.c_si**2)
        dtdc = np.asarray(delta_theta_dot, dtype=float) / self.c_si
        return 1.0 + self.gamma * phi_over_c2 * dtdc

    def maxwell_lift_coefficient(
        self,
        phi_over_c2: Union[float, np.ndarray],
        delta_theta_dot_over_c: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """γ(φ/c²)(˙δθ′/c)."""
        return self.gamma * np.asarray(phi_over_c2) * np.asarray(delta_theta_dot_over_c)
