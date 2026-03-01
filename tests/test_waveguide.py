"""Tests for HQIV waveguide: k_c², geometry rules, mode solver."""

import numpy as np

from pyhqiv.constants import C_SI
from pyhqiv.waveguide import (
    distance_to_boundary_circle,
    distance_to_boundary_rect,
    dot_delta_theta_from_phi,
    hyperbolic_boundary_r,
    kc_squared_hqiv,
    rectangular_cutoff_kc_squared,
    waveguide_radius_constant_phi,
    waveguide_taper_slope,
    waveguide_te11_cutoff_beta,
)


def test_dot_delta_theta_from_phi():
    """δ̇θ′ = γ φ / c."""
    phi = 1e5
    d = dot_delta_theta_from_phi(phi, gamma=0.4, c=C_SI)
    expected = 0.4 * phi / C_SI
    assert abs(d - expected) < 1e-10


def test_kc_squared_hqiv_m0():
    """m_phase=0 => k_c² = ω²/c² - β² (real)."""
    omega = 2 * np.pi * 1e9
    beta = 10.0
    kc2 = kc_squared_hqiv(omega, beta, 0, 0.0, c=C_SI)
    expected = (omega / C_SI) ** 2 - beta**2
    assert abs(kc2 - expected) < 1e-10
    assert abs(np.imag(kc2)) < 1e-15


def test_kc_squared_hqiv_m1_complex():
    """m_phase=1 with δ̇θ′ ≠ 0 => complex k_c²."""
    omega = 2 * np.pi * 1e9
    beta = 10.0
    dot = 1e-18
    kc2 = kc_squared_hqiv(omega, beta, 1, dot, c=C_SI)
    assert not np.isrealobj(kc2)
    assert np.imag(kc2) != 0


def test_waveguide_radius_constant_phi():
    """a = 2c²/φ_target."""
    phi = 1e10
    a = waveguide_radius_constant_phi(phi, c=C_SI)
    expected = 2 * C_SI**2 / phi
    assert abs(a - expected) < 0.01


def test_waveguide_te11_cutoff_beta():
    """β for TE11 with 1.841/a cutoff."""
    omega = 2 * np.pi * 10e9
    a = 0.01
    beta = waveguide_te11_cutoff_beta(omega, a, 0, 0.0, c=C_SI)
    kc = 1.841 / a
    expected_sq = (omega / C_SI) ** 2 - kc**2
    assert abs(beta**2 - expected_sq) < 1e-6


def test_waveguide_taper_slope():
    """da/dz = -a/Θ * dΘ/dz."""
    da_dz = waveguide_taper_slope(a=1.0, Theta_local=0.5, dTheta_dz=-0.1)
    assert abs(da_dz - (1.0 * 0.1 / 0.5)) < 1e-10


def test_hyperbolic_boundary_r():
    """r(θ) = Θ0 cosh(κ θ)."""
    r = hyperbolic_boundary_r(0.0, Theta0=1.0, kappa=0.1)
    assert abs(r - 1.0) < 1e-10
    r2 = hyperbolic_boundary_r(1.0, Theta0=1.0, kappa=0.1)
    assert r2 > 1.0


def test_rectangular_cutoff_kc_squared():
    """k_c,mn² = (mπ/w)² + (nπ/h)² + phase terms."""
    kc2 = rectangular_cutoff_kc_squared(1, 0, 1.0, 0.5, 2 * np.pi * 1e9, 10.0, 0, 0.0, c=C_SI)
    expected_geom = (np.pi / 1.0) ** 2
    assert abs(kc2.real - expected_geom) < 1e-6


def test_distance_to_boundary_rect():
    """Distance to nearest wall in [0,w]×[0,h]."""
    x = np.array([[0.5, 1.0], [0.5, 1.0]])
    y = np.array([[0.25, 0.25], [0.75, 0.75]])
    d = distance_to_boundary_rect(x, y, w=2.0, h=1.0)
    assert d.shape == (2, 2)
    assert np.all(d > 0)
    assert abs(d[0, 0] - 0.25) < 0.01  # min(0.5, 1.0, 0.25, 0.75) = 0.25


def test_distance_to_boundary_circle():
    """Distance to circle boundary = max(0, a - r)."""
    d = distance_to_boundary_circle(0.0, 0.0, cx=0, cy=0, radius=1.0)
    assert abs(d - 1.0) < 0.01
    d_edge = distance_to_boundary_circle(1.0, 0.0, cx=0, cy=0, radius=1.0)
    assert d_edge < 0.1


def test_waveguide_mode_solver_import():
    """Mode solver requires scipy."""
    from pyhqiv import waveguide

    assert hasattr(waveguide, "hqiv_waveguide_mode_solver")


def test_waveguide_mode_solver_small_grid():
    """Run mode solver on small grid (constant Theta)."""
    from pyhqiv.waveguide import hqiv_waveguide_mode_solver

    # 5x5 so dx, dy are non-zero
    gx, gy = np.mgrid[0:1:5j, 0:1:5j]
    omega = 2 * np.pi * 1e9
    beta = 0.0
    evals, evecs, mask = hqiv_waveguide_mode_solver(
        gx, gy, omega, beta, m_phase=0, Theta_grid=None, n_modes=2
    )
    assert evals.size <= 2
    assert mask.shape == gx.shape
