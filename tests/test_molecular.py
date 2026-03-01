"""Tests for molecular / PROtien helpers: theta_local, bond_length, damping."""

import numpy as np
import pytest

from pyhqiv.constants import A_LOC_ANG, HBAR_C_EV_ANG
from pyhqiv.utils import (
    bond_length_from_theta,
    damping_force_magnitude,
    theta_for_atom,
    theta_local,
)


def test_theta_local_c_n_reference():
    """Θ_C(coord=2) ≈ 1.53 Å, Θ_N(coord=2) ≈ 1.33 Å."""
    assert abs(theta_local(6, 2) - 1.53) < 0.01
    assert abs(theta_local(7, 2) - 1.33) < 0.05


def test_theta_for_atom():
    """theta_for_atom uses Z map and matches theta_local."""
    assert abs(theta_for_atom("C", 2) - 1.53) < 0.01
    assert abs(theta_for_atom("N", 2) - theta_local(7, 2)) < 1e-10
    assert theta_for_atom("O", 1) > theta_for_atom("O", 2)


def test_bond_length_from_theta():
    """r_eq = min(Θ_i, Θ_j) * monogamy_factor."""
    assert bond_length_from_theta(1.53, 1.33, 1.0) == 1.33
    assert bond_length_from_theta(1.0, 2.0, 2.0) == 2.0


def test_damping_force_magnitude():
    """|f_φ| = γ φ |∇φ| / (a_loc + φ/6)²."""
    f = damping_force_magnitude(1.0, 0.5, a_loc=1.0)
    assert f > 0
    denom = (1.0 + 1.0 / 6.0) ** 2
    expected = 0.40 * 1.0 * 0.5 / denom
    assert abs(f - expected) < 1e-10


def test_constants_molecular():
    """HBAR_C_EV_ANG and A_LOC_ANG present."""
    assert HBAR_C_EV_ANG > 1970 and HBAR_C_EV_ANG < 1980
    assert A_LOC_ANG == 1.0
