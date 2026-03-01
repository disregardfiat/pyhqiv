"""Tests for modified Navier-Stokes (HQIV fluid): f_inertia, g_vac, ν_eddy."""

import numpy as np
import pytest

from pyhqiv.fluid import (
    eddy_viscosity,
    f_inertia,
    g_vac_vector,
    modified_momentum_rhs,
)
from pyhqiv.constants import GAMMA


def test_f_inertia_laminar_limit():
    """Large |a| ⇒ f → 1."""
    assert abs(f_inertia(100.0, 1.0) - 1.0) < 0.01
    assert abs(f_inertia(1e6, 1.0) - 1.0) < 1e-6


def test_f_inertia_small_a():
    """Small |a| ⇒ f < 1, f = a/(a + φ/6)."""
    f = f_inertia(0.1, 1.0)
    expected = 0.1 / (0.1 + 1.0 / 6.0)
    assert abs(f - expected) < 1e-10
    assert 0 < f < 1


def test_f_inertia_floor():
    """f never below f_min."""
    f = f_inertia(0.0, 1.0, f_min=0.01)
    assert f >= 0.01


def test_f_inertia_array():
    """Vectorized a and phi."""
    a = np.array([0.1, 1.0, 10.0])
    phi = np.array([1.0, 1.0, 1.0])
    f = f_inertia(a, phi)
    assert f.shape == (3,)
    assert f[0] < f[1] < f[2]
    assert abs(f[2] - 1.0) < 0.1


def test_g_vac_vector():
    """g_vac = -γ/6 * (φ ∇δ̇θ′ + δ̇θ′ ∇φ)."""
    phi = 1.0
    dot = 0.5
    grad_phi = np.array([1.0, 0.0, 0.0])
    grad_dot = np.array([0.0, 0.1, 0.0])
    g = g_vac_vector(phi, dot, grad_phi, grad_dot, gamma=0.4)
    expected = -0.4 / 6.0 * (phi * grad_dot + dot * grad_phi)
    np.testing.assert_array_almost_equal(g, expected)
    assert g.shape == (3,)


def test_g_vac_homogeneous_gradient_zero():
    """If ∇φ and ∇δ̇θ′ are zero, g_vac = 0."""
    g = g_vac_vector(1.0, 0.5, np.zeros(3), np.zeros(3))
    np.testing.assert_array_almost_equal(g, np.zeros(3))


def test_eddy_viscosity():
    """ν_eddy = γ Θ |δ̇θ′| ℓ_coh² C."""
    nu = eddy_viscosity(
        Theta_local=1.0,
        dot_delta_theta=1e-18,
        l_coh=1e-3,
        coherence_factor=0.5,
        gamma=0.4,
    )
    expected = 0.4 * 1.0 * 1e-18 * (1e-3 ** 2) * 0.5
    assert abs(nu - expected) < 1e-30
    assert nu >= 0


def test_eddy_viscosity_default_gamma():
    """Uses GAMMA when gamma=None."""
    nu = eddy_viscosity(1.0, 1e-18, 1e-3, 1.0, gamma=None)
    assert nu == GAMMA * 1.0 * 1e-18 * 1e-6


def test_modified_momentum_rhs():
    """RHS = -∇p/ρ + div_τ/ρ + g_ext + g_vac."""
    grad_p = np.array([1.0, 0.0, 0.0])
    div_tau = np.zeros(3)
    g_ext = np.array([0.0, 0.0, -10.0])
    g_vac = np.zeros(3)
    rhs = modified_momentum_rhs(grad_p, div_tau, g_ext, g_vac, rho=1.0)
    np.testing.assert_array_almost_equal(rhs, [-1.0, 0.0, -10.0])


def test_laminar_collapse():
    """When φ/6 negligible vs |a|, f≈1; when ∇(φ δ̇θ′)=0, g_vac=0 → standard NS."""
    f = f_inertia(100.0, 0.01)
    assert abs(f - 1.0) < 0.01
    g = g_vac_vector(1.0, 0.5, np.zeros(3), np.zeros(3))
    assert np.allclose(g, 0.0)
