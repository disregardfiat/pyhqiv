"""Tests for DiscreteNullLattice and curvature imprint."""

import numpy as np
import pytest

from pyhqiv.lattice import (
    DiscreteNullLattice,
    discrete_mode_count,
    curvature_imprint_delta_E,
    omega_k_from_shell_integral,
)
from pyhqiv.constants import M_TRANS, T_PL_GEV, ALPHA, COMBINATORIAL_INVARIANT


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
