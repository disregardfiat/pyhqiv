"""Tests for DiscreteNullLattice and curvature imprint."""

import numpy as np
import pytest

from pyhqiv.constants import ALPHA, T_PL_GEV
from pyhqiv.lattice import (
    DiscreteNullLattice,
    curvature_imprint_delta_E,
    discrete_mode_count,
    omega_k_from_shell_integral,
)


def test_lattice_omega_k_reproducible():
    np.random.seed(42)
    lat = DiscreteNullLattice(m_trans=500, seed=42)
    o1 = lat.omega_k_true()
    o2 = lat.omega_k_true()
    assert o1 == o2
    assert abs(o1 - 0.0098) < 1e-5


def test_delta_E_grid_shape():
    lat = DiscreteNullLattice(m_trans=100)
    de = lat.get_delta_E_grid()
    assert de.shape == (100,)


def test_shell_temperature():
    lat = DiscreteNullLattice()
    m = np.array([0, 10, 100])
    T = lat.shell_temperature(m)
    assert np.all(T > 0)
    assert T[0] > T[1] > T[2]


def test_mode_count_per_shell():
    lat = DiscreteNullLattice(m_trans=20)
    m = np.arange(20)
    dN = lat.mode_count_per_shell(m)
    assert dN[0] == discrete_mode_count(0)
    assert dN[1] == discrete_mode_count(1)


def test_cumulative_mode_count_hockey_stick():
    """Vectorized cumulative_mode_count matches sum of discrete_mode_count."""
    from pyhqiv.lattice import cumulative_mode_count

    for m_max in [1, 2, 10, 100, 501]:
        total = sum(discrete_mode_count(m) for m in range(m_max))
        assert abs(cumulative_mode_count(m_max) - total) < 1e-9


def test_get_cumulative_mode_counts_vectorized():
    """get_cumulative_mode_counts matches cumulative_mode_count at each shell."""
    from pyhqiv.lattice import cumulative_mode_count

    lat = DiscreteNullLattice(m_trans=50)
    counts = lat.get_cumulative_mode_counts()
    assert len(counts) == lat.m_trans + 1
    assert counts[0] == 0.0
    for k in range(1, lat.m_trans + 1):
        assert abs(counts[k] - cumulative_mode_count(k)) < 1e-9


@pytest.mark.parametrize("m_trans", [50, 100, 500])
def test_omega_k_positive_for_various_m_trans(m_trans):
    """omega_k_true is positive and in a reasonable range for different m_trans."""
    lat = DiscreteNullLattice(m_trans=m_trans)
    omega = lat.omega_k_true()
    assert omega > 0
    assert omega < 0.1


@pytest.mark.parametrize("gamma,alpha", [(0.40, 0.60), (0.35, 0.55)])
def test_evolve_to_cmb_parametrized(gamma, alpha):
    """evolve_to_cmb returns consistent structure for different gamma, alpha."""
    lat = DiscreteNullLattice(m_trans=100, gamma=gamma, alpha=alpha)
    out = lat.evolve_to_cmb(T0_K=2.725)
    assert "Omega_true_k" in out
    assert out["gamma"] == gamma
    assert out["m_trans"] == 100


def test_omega_k_jax_matches_numpy():
    """When JAX is available, use_jax=True gives same result as use_jax=False."""
    pytest.importorskip("jax")
    omega_np = omega_k_from_shell_integral(m_trans=200, use_jax=False)
    omega_jax = omega_k_from_shell_integral(m_trans=200, use_jax=True)
    # JAX may use float32; relax tolerance
    assert abs(omega_np - omega_jax) < 1e-6


def test_curvature_imprint_no_nan_extreme_T():
    """δE(m,T) produces no NaNs for very small or very large T (inf allowed)."""
    m = np.array([0.0, 1.0, 10.0, 100.0])
    T_tiny = np.array([1e-200, 1e-100, 1e-50, 1e-20])
    T_large = np.array([1e20, 1e50, 1e100, 1e200])
    de_tiny = curvature_imprint_delta_E(m, T_tiny, T_Pl=T_PL_GEV, alpha=ALPHA)
    de_large = curvature_imprint_delta_E(m, T_large, T_Pl=T_PL_GEV, alpha=ALPHA)
    assert np.all(~np.isnan(de_tiny))
    assert np.all(~np.isnan(de_large))
    assert np.all(de_tiny > 0)
    assert np.all(de_large > 0)
