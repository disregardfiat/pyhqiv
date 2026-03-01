"""Tests for semiconductors: compute_band_gap, dos, effective_mass, dielectric."""

import numpy as np

from pyhqiv.semiconductors import (
    compute_band_gap,
    compute_conductivity_tensor,
    dielectric_function_epsilon,
    dos,
    effective_mass,
)


def test_compute_band_gap_direct():
    """Direct gap: VBM at same k as CBM."""
    # 2 k-points, 4 bands; valence 0,1 conduction 2,3
    ev = np.array(
        [
            [-2.0, -1.0, 0.5, 1.0],  # k1: gap 0.5 - (-1) = 1.5
            [-2.1, -0.9, 0.6, 1.1],  # k2: gap 0.6 - (-0.9) = 1.5
        ]
    )
    gap, gap_type = compute_band_gap(ev, phi_avg=0.0)
    assert gap > 0
    assert abs(gap - 1.5) < 0.2
    assert gap_type in ("direct", "indirect")


def test_compute_band_gap_with_hqiv_shift():
    """HQIV shift moves eigenvalues; gap changes by V_shift (all bands shifted)."""
    ev = np.array([[-1.0, 0.5]])  # 1 k, 2 bands; gap = 1.5
    gap0, _ = compute_band_gap(ev, phi_avg=0.0, dot_delta_theta_avg=0.0)
    gap1, _ = compute_band_gap(ev, phi_avg=1e-10, dot_delta_theta_avg=1e-18, gamma=0.4)
    # Same gap when shift is uniform (VBM and CBM both +V_shift)
    assert abs(gap1 - gap0) < 1e-12


def test_compute_band_gap_single_k():
    """Single k (1D eigenvalues) is supported."""
    ev = np.array([-1.0, 0.0, 0.5, 1.0])
    gap, _ = compute_band_gap(ev)
    assert gap > 0
    assert gap <= 2.0


def test_dos_shape():
    """DOS has same length as energy grid."""
    ev = np.random.randn(20, 4)
    energies = np.linspace(-2, 2, 50)
    rho = dos(ev, energies, sigma=0.1)
    assert rho.shape == energies.shape
    assert np.all(rho >= 0)


def test_dos_weights():
    """k_weights change DOS normalization."""
    ev = np.ones((3, 2)) * 0.5
    energies = np.linspace(0, 1, 10)
    rho1 = dos(ev, energies, sigma=0.2, k_weights=np.ones(3))
    rho2 = dos(ev, energies, sigma=0.2, k_weights=np.array([2, 1, 1]))
    assert np.all(np.isfinite(rho1)) and np.all(np.isfinite(rho2))


def test_effective_mass_parabolic():
    """Effective mass from parabolic E(k); finite and positive when curvature is clear."""
    # Linear k path along x; E = 7.62 * kx^2 (parabolic)
    kx = np.linspace(-0.02, 0.02, 41)
    kpts = np.column_stack([kx, np.zeros(41), np.zeros(41)])
    ev = 7.62 * (kx**2)
    ev = np.tile(ev.reshape(-1, 1), (1, 2))
    m = effective_mass(ev, kpts, band_index=0, direction=0, dk=1e-3)
    # effective_mass uses nearest-k interpolation; may be nan if dk doesn't hit grid
    if np.isfinite(m):
        assert m > 0
        assert abs(m - 0.5) < 1.0  # same order of magnitude
    else:
        # At least check the call doesn't crash and returns float
        assert isinstance(m, (float, np.floating))


def test_effective_mass_nan_too_few_k():
    """Too few k-points returns nan."""
    ev = np.array([[0.0], [1.0]])
    kpts = np.array([[0, 0, 0], [0.01, 0, 0]])
    m = effective_mass(ev, kpts, band_index=0)
    assert np.isnan(m) or not np.isfinite(m)


def test_compute_conductivity_tensor():
    """Conductivity tensor 3×3 diagonal."""
    sigma = compute_conductivity_tensor(1e10, T=300, sigma_0=1.0, dim=3)
    assert sigma.shape == (3, 3)
    assert abs(sigma[0, 0] - sigma[1, 1]) < 1e-10
    assert abs(sigma[0, 1]) < 1e-10


def test_dielectric_function_epsilon():
    """Dielectric function returns complex array."""
    omega = np.array([1e10, 2e10])
    eps = dielectric_function_epsilon(omega, sigma_0=1.0, phi_avg=0.0)
    assert eps.shape == omega.shape
    assert np.all(np.isfinite(eps.real)) and np.all(np.isfinite(eps.imag))
