"""
HQIV Orbital Mechanics: lapse on proper time, effective inertia f, clock desynchronization.

Core axiom: f(a_loc, φ) = a_loc / (a_loc + φ/6); φ = 2 c²/Θ_local.
Integrates with REBOUND/Gala-style stepping: at each step compute φ(r), f(r), advance τ with dτ = dt/f.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import C_SI, GAMMA
from pyhqiv.fluid import f_inertia
from pyhqiv.utils import phi_from_theta_local

# SI constants for orbits
G_SI: float = 6.67430e-11  # m³/(kg·s²)
M_SUN_KG: float = 1.9884e30
AU_M: float = 1.495978707e11  # m


def theta_local_from_r(r: Union[float, np.ndarray], r_min: float = 1e3) -> np.ndarray:
    """Θ_local ∝ r (radial); floor r_min to avoid φ → ∞. Units: same as r (m)."""
    r = np.asarray(r)
    return np.maximum(np.abs(r), r_min)


def phi_from_r(r: Union[float, np.ndarray], c: float = C_SI, r_min: float = 1e3) -> np.ndarray:
    """φ(r) = 2 c²/Θ_local(r) with Θ = max(|r|, r_min)."""
    theta = theta_local_from_r(r, r_min)
    return phi_from_theta_local(theta, c=c)


def proper_time_rate(f: Union[float, np.ndarray]) -> np.ndarray:
    """dτ/dt = f (local proper time rate vs coordinate time when lapse is from f)."""
    return np.asarray(f, dtype=float)


def clock_desync_ratio(
    f_obs: Union[float, np.ndarray],
    f_emit: Union[float, np.ndarray],
) -> np.ndarray:
    """Δτ_emit / Δτ_obs = f_emit / f_obs (ratio of proper time intervals)."""
    return np.asarray(f_emit, dtype=float) / np.maximum(np.asarray(f_obs, dtype=float), 1e-30)


class HQIVOrbit:
    """
    Orbital mechanics with HQIV: φ(r), f(r) = a_loc/(a_loc+φ/6), proper-time rate f,
    and optional force scaling. Can wrap REBOUND or Gala; here we provide a simple
    Kepler + lapse integrator.
    """

    def __init__(
        self,
        M_central_kg: float = M_SUN_KG,
        gamma: float = GAMMA,
        c_si: float = C_SI,
        r_min_m: float = 1e3,
    ) -> None:
        self.M_central = M_central_kg
        self.gamma = gamma
        self.c_si = c_si
        self.r_min = r_min_m
        self.G = G_SI

    def theta_local(self, r_vec: np.ndarray) -> np.ndarray:
        """Θ at position r_vec (shape (..., 3)); returns (...,)."""
        r = np.linalg.norm(r_vec, axis=-1)
        return np.maximum(r, self.r_min)

    def phi(self, r_vec: np.ndarray) -> np.ndarray:
        """φ at position r_vec. r_vec in m, φ in m²/s²."""
        theta = self.theta_local(r_vec)
        return phi_from_theta_local(theta, c=self.c_si)

    def a_grav_mag(self, r_vec: np.ndarray) -> np.ndarray:
        """Newtonian |a| = G M / r² (m/s²)."""
        r = np.linalg.norm(r_vec, axis=-1)
        return self.G * self.M_central / np.maximum(r**2, 1e-30)

    def lapse_f(self, r_vec: np.ndarray) -> np.ndarray:
        """f(a_loc, φ) at position r_vec; a_loc = G M/r²."""
        phi = self.phi(r_vec)
        a_loc = self.a_grav_mag(r_vec)
        return f_inertia(a_loc, phi)

    def proper_time_step(
        self,
        r_vec: np.ndarray,
        dt_coordinate: float,
    ) -> float:
        """Proper time increment dτ = f(r) * dt (for a particle at r_vec)."""
        f = self.lapse_f(np.asarray(r_vec).reshape(1, 3)).reshape(())
        return float(f * dt_coordinate)

    def acceleration_hqiv(
        self,
        r_vec: np.ndarray,
        scale_force_by_1_over_f: bool = False,
    ) -> np.ndarray:
        """
        Newtonian a = -G M r̂/r². If scale_force_by_1_over_f, effective a_eff = a/f
        (modified inertia: m_eff a_eff = F ⇒ a_eff = F/(m f)).
        """
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r = np.maximum(r, 1e-30)
        r_hat = r_vec / r
        a_newt = -self.G * self.M_central / (r**2) * r_hat
        if not scale_force_by_1_over_f:
            return a_newt
        f = self.lapse_f(r_vec)
        f = np.expand_dims(np.asarray(f), axis=-1)
        return a_newt / np.maximum(f, 0.01)

    def integrate_kepler_with_lapse(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
        return_proper_time: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Simple Euler (or symplectic) integration of d²r/dt² = -G M r/r³.
        Returns t (coord), r (n_steps+1, 3), v (n_steps+1, 3), and τ (proper time) if requested.
        """
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        t = np.zeros(n_steps + 1)
        r = np.zeros((n_steps + 1, 3))
        v = np.zeros((n_steps + 1, 3))
        tau = np.zeros(n_steps + 1) if return_proper_time else None

        t[0] = t0
        r[0] = np.asarray(r0, dtype=float).reshape(3)
        v[0] = np.asarray(v0, dtype=float).reshape(3)
        if return_proper_time:
            tau[0] = 0.0

        for i in range(n_steps):
            a = self.acceleration_hqiv(r[i : i + 1], scale_force_by_1_over_f=False).reshape(3)
            v[i + 1] = v[i] + a * dt
            r[i + 1] = r[i] + v[i + 1] * dt
            t[i + 1] = t[i] + dt
            if return_proper_time:
                dtau = self.proper_time_step(r[i + 1], dt)
                tau[i + 1] = tau[i] + dtau

        return t, r, v, tau

    def earth_sun_example(
        self,
        n_steps: int = 500,
        n_orbits: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Earth-like orbit: a ≈ 1 au, T ≈ 1 yr. Returns t, r, v, τ.
        """
        a_au = 1.0
        a_m = a_au * AU_M
        # v_circ = sqrt(G M / a)
        v_circ = np.sqrt(self.G * self.M_central / a_m)
        T_orb = 2.0 * np.pi * a_m / v_circ  # period in s
        r0 = np.array([a_m, 0.0, 0.0])
        v0 = np.array([0.0, v_circ, 0.0])
        t_span = (0.0, n_orbits * T_orb)
        t, r, v, tau = self.integrate_kepler_with_lapse(
            r0, v0, t_span, n_steps=n_steps, return_proper_time=True
        )
        return t, r, v, tau  # type: ignore


def parker_perihelion_lapse(R_perihelion_au: float = 0.05) -> float:
    """Lapse factor f at Parker Solar Probe perihelion (au). φ and f at that r."""
    r_m = R_perihelion_au * AU_M
    phi = float(phi_from_r(r_m))
    a_loc = G_SI * M_SUN_KG / (r_m**2)
    return float(f_inertia(a_loc, phi))
