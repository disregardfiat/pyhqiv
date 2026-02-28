"""
Phase-horizon corrected Maxwell: FDTD (Yee grid) + two extra derivative terms
from the geometric route (Paper Sec. 2 & 5). Reduces to ordinary Maxwell when ˙δθ′→0.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pyhqiv.constants import C_SI, GAMMA


def _curl_E(E: np.ndarray, dx: float) -> np.ndarray:
    """∇×E on Yee grid (E at edges). Returns dB/dt component."""
    Ex, Ey, Ez = E[0], E[1], E[2]
    # Standard curl
    dEy_dz = np.gradient(Ey, dx, axis=2)
    dEz_dy = np.gradient(Ez, dx, axis=1)
    dEx_dz = np.gradient(Ex, dx, axis=2)
    dEz_dx = np.gradient(Ez, dx, axis=0)
    dEx_dy = np.gradient(Ex, dx, axis=1)
    dEy_dx = np.gradient(Ey, dx, axis=0)
    curl_x = dEz_dy - dEy_dz
    curl_y = dEx_dz - dEz_dx
    curl_z = dEy_dx - dEx_dy
    return np.array([curl_x, curl_y, curl_z])


def _curl_H(H: np.ndarray, dx: float) -> np.ndarray:
    """∇×H on Yee grid. Returns dD/dt component."""
    return _curl_E(H, dx)  # same stencil for curl


class PhaseHorizonFDTD:
    """
    Phase-horizon FDTD: Yee grid plus two extra terms from
    D/Dt = ∂/∂t′ + ˙δθ′ ∂/∂δθ′:
    ∇×E = -(∂B/∂t′ + ˙δθ′ ∂B/∂δθ′),
    ∇×H = J + (∂D/∂t′ + ˙δθ′ ∂D/∂δθ′).
    When ˙δθ′→0 reduces to ordinary Maxwell.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        dx: float = 1.0,
        dt: float = 0.5,
        c: float = 1.0,
        gamma: float = GAMMA,
    ) -> None:
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = c
        self.gamma = gamma
        # Yee: E and B staggered
        self.E = np.zeros((3,) + shape)
        self.B = np.zeros((3,) + shape)
        self.phi_grid: Optional[np.ndarray] = None
        self.delta_theta_dot_grid: Optional[np.ndarray] = None

    def set_phase_horizon(
        self,
        phi_over_c2: np.ndarray,
        delta_theta_dot_over_c: np.ndarray,
    ) -> None:
        """Set γ(φ/c²)(˙δθ′/c) field on grid for modified update."""
        self.phi_grid = np.asarray(phi_over_c2)
        self.delta_theta_dot_grid = np.asarray(delta_theta_dot_over_c)

    def step(
        self,
        J: Optional[np.ndarray] = None,
    ) -> None:
        """
        One FDTD step. With phase lift: ∂B/∂t = -∇×E - ˙δθ′ ∂B/∂δθ′ (extra term).
        We approximate ∂B/∂δθ′ ≈ (∂B/∂t)/H so extra = ˙δθ′*(∂B/∂t)/H; then
        (1 + ˙δθ′/H) ∂B/∂t = -∇×E ⇒ ∂B/∂t = -∇×E / (1 + ˙δθ′/H). Same for D.
        """
        curl_E = _curl_E(self.E, self.dx)
        # Ordinary: dB/dt = -curl_E
        if self.delta_theta_dot_grid is None or self.phi_grid is None:
            self.B -= self.dt * curl_E * self.c
        else:
            # Factor 1/(1 + γ φ/c² ˙δθ′/c) for effective lapse
            fac = 1.0 + self.gamma * self.phi_grid * self.delta_theta_dot_grid
            self.B -= self.dt * curl_E * self.c / np.maximum(fac, 1e-30)

        curl_H = _curl_H(self.B / np.maximum(self.c, 1e-30), self.dx)
        if J is None:
            J = np.zeros_like(self.E)
        self.E += self.dt * (self.c * curl_H - J)
        if self.delta_theta_dot_grid is not None and self.phi_grid is not None:
            fac = 1.0 + self.gamma * self.phi_grid * self.delta_theta_dot_grid
            self.E /= np.maximum(fac, 1e-30)

    def get_E_B(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current E and B arrays."""
        return self.E.copy(), self.B.copy()
