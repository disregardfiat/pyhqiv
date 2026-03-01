"""Tests for conductivity / response API."""

from pyhqiv.response import compute_conductivity, response_tensor_diagonal


def test_compute_conductivity_no_phi():
    """When phi_avg=0, σ = σ_0."""
    s = compute_conductivity(omega=1e10, sigma_0=1.0, phi_avg=0.0)
    assert abs(s - 1.0) < 1e-10


def test_compute_conductivity_with_phi():
    """σ = σ_0 * (1 + γ φ/(ω+ε))."""
    s = compute_conductivity(omega=1e10, sigma_0=2.0, phi_avg=1e5, gamma=0.4)
    expected = 2.0 * (1.0 + 0.4 * 1e5 / 1e10)
    assert abs(s - expected) < 1e-6


def test_response_tensor_diagonal():
    """Returns 3×3 diagonal matrix."""
    sig = response_tensor_diagonal(omega=1e10, dim=3, sigma_0=1.0)
    assert sig.shape == (3, 3)
    assert abs(sig[0, 0] - sig[1, 1]) < 1e-10
    assert abs(sig[0, 1]) < 1e-10
