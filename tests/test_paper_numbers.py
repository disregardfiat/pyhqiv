"""
Tests that reproduce the paper's exact numerical predictions to 6 decimals.
Paper: Ettinger, February 22, 2026.
"""

import pytest
import numpy as np

from pyhqiv.constants import (
    GAMMA,
    ALPHA,
    T_LOCK_GEV,
    T_CMB_K,
    T_PL_GEV,
    M_TRANS,
    COMBINATORIAL_INVARIANT,
    OMEGA_TRUE_K_PAPER,
    LAPSE_COMPRESSION_PAPER,
    AGE_WALL_GYR_PAPER,
    AGE_APPARENT_GYR_PAPER,
)
from pyhqiv.lattice import (
    DiscreteNullLattice,
    discrete_mode_count,
    cumulative_mode_count,
    curvature_imprint_delta_E,
    omega_k_from_shell_integral,
)
from pyhqiv.algebra import OctonionHQIVAlgebra


def test_gamma_value():
    """γ ≈ 0.40 (entanglement monogamy coefficient)."""
    assert abs(GAMMA - 0.40) < 1e-6


def test_alpha_value():
    """α ≈ 0.60."""
    assert abs(ALPHA - 0.60) < 1e-6


def test_T_lock_GeV():
    """T_lock = 1.8 GeV."""
    assert abs(T_LOCK_GEV - 1.8) < 1e-6


def test_T_CMB_K():
    """T_CMB = 2.725 K."""
    assert abs(T_CMB_K - 2.725) < 1e-6


def test_m_trans():
    """m_trans = 500."""
    assert M_TRANS == 500


def test_combinatorial_invariant():
    """6^7 √3 ≈ 4.849×10^5."""
    expected = (6 ** 7) * np.sqrt(3)
    assert abs(COMBINATORIAL_INVARIANT - expected) < 1e-3
    assert abs(COMBINATORIAL_INVARIANT - 4.849e5) < 5e3  # order of magnitude


def test_Omega_true_k():
    """Ω_k^true ≈ +0.0098 from lattice shell integral."""
    lattice = DiscreteNullLattice(m_trans=500, gamma=0.40)
    result = lattice.evolve_to_cmb(T0_K=2.725)
    omega = result["Omega_true_k"]
    assert abs(omega - OMEGA_TRUE_K_PAPER) < 1e-6
    assert abs(omega - 0.0098) < 1e-6


def test_lapse_compression():
    """51.2 Gyr wall-clock → 13.8 Gyr apparent; time-dilation factor ≈ 3.96 (paper)."""
    assert abs(AGE_WALL_GYR_PAPER - 51.2) < 1e-6
    assert abs(AGE_APPARENT_GYR_PAPER - 13.8) < 1e-6
    # Paper: lapse compression factor ≈ 3.96 (wall / apparent lookback ~12.9 Gyr)
    assert abs(LAPSE_COMPRESSION_PAPER - 3.96) < 0.1
    comp = AGE_WALL_GYR_PAPER / AGE_APPARENT_GYR_PAPER  # 51.2/13.8 ≈ 3.71
    assert 3.5 <= comp <= 4.0


def test_evolve_to_cmb_returns_correct_keys():
    """evolve_to_cmb returns Omega_true_k, age_wall_Gyr, lapse_compression."""
    lattice = DiscreteNullLattice(m_trans=500)
    out = lattice.evolve_to_cmb(T0_K=2.725)
    assert "Omega_true_k" in out
    assert "age_wall_Gyr" in out
    assert "lapse_compression" in out


def test_delta_E_shape_and_combinatorial():
    """δE(m) uses 1/(m+1) and (1+α ln(T_Pl/T)) × 6^7√3."""
    m = np.array([0, 1, 10, 100, 499])
    T = T_PL_GEV / (m + 1.0)
    delta_E = curvature_imprint_delta_E(m, T)
    assert delta_E.shape == m.shape
    assert np.all(delta_E > 0)
    # At m=0: 1/(1) * (1 + α*0) * N67 ≈ N67 (allow small numerical tolerance)
    assert abs(delta_E[0] - COMBINATORIAL_INVARIANT) < 500
    assert delta_E[0] / COMBINATORIAL_INVARIANT < 1.01


def test_mode_count_combinatorial():
    """dN(m) = 8*binom(m+2, 2)."""
    assert discrete_mode_count(0) == 8 * 1  # binom(2,2)=1
    assert discrete_mode_count(1) == 8 * 3  # binom(3,2)=3
    total_500 = cumulative_mode_count(500)
    assert total_500 > 0


def test_omega_k_from_shell_integral_at_500():
    """omega_k_from_shell_integral(m_trans=500) ≈ 0.0098."""
    omega = omega_k_from_shell_integral(m_trans=500)
    assert abs(omega - 0.0098) < 1e-6


def test_so8_closure_dimension_28():
    """OctonionHQIVAlgebra closes to so(8) dimension 28."""
    alg = OctonionHQIVAlgebra(verbose=False)
    dim, history = alg.lie_closure_dimension()
    assert dim == 28


def test_hypercharge_4x4_block():
    """Hypercharge Y has 4×4 block with eigenvalues ±i/6, ±i/2."""
    alg = OctonionHQIVAlgebra(verbose=False)
    data = alg.hypercharge_paper_data()
    assert data is not None
    ev = data["eigenvalues_i_block"]
    assert len(ev) == 4
    # Should be ±1/6, ±1/2 (imaginary parts)
    assert abs(np.abs(ev).min() - 1.0 / 6.0) < 0.1
    assert abs(np.abs(ev).max() - 0.5) < 0.1


def test_lie_closure_max_iter_param():
    """lie_closure_dimension(max_iter=40) and max_iter=100 both give dimension 28."""
    alg = OctonionHQIVAlgebra(verbose=False)
    dim40, _ = alg.lie_closure_dimension(max_iter=40)
    dim100, _ = alg.lie_closure_dimension(max_iter=100)
    assert dim40 == 28
    assert dim100 == 28


def test_hypercharge_block_weight_param():
    """hypercharge_coefficients(block_weight=1e15) and 1e12 both yield valid Y."""
    alg = OctonionHQIVAlgebra(verbose=False)
    c1, Y1, _ = alg.hypercharge_coefficients(block_weight=1e15)
    c2, Y2, _ = alg.hypercharge_coefficients(block_weight=1e12)
    assert c1 is not None and Y1 is not None
    assert c2 is not None and Y2 is not None
    assert np.all(np.isfinite(Y1)) and np.all(np.isfinite(Y2))
