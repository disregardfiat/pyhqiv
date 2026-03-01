"""Tests for ASE interface: hqiv_energy_at_positions, forces, stress, HQIVCalculator."""

import numpy as np
import pytest

from pyhqiv.ase_interface import (
    hqiv_energy_at_positions,
    hqiv_forces_analytic,
    hqiv_stress_virial,
)
from pyhqiv.system import HQIVSystem
from pyhqiv.constants import GAMMA, C_SI


def test_hqiv_energy_at_positions_two_atoms():
    """E = (1/2) sum φ with φ = 2c²/r; two atoms at (0,0,0) and (1,0,0)."""
    sys = HQIVSystem.from_atoms([(0, 0, 0), (1.0, 0, 0)], charges=[1, -1])
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    E = hqiv_energy_at_positions(sys, positions, energy_scale=1.0)
    assert np.isfinite(E)
    assert E > 0  # repulsive-like sum of φ from other atom
    # φ ∝ 1/r so at r=1 scale ~ 2*c²; rough order
    assert E < 1e30 and E > 1e-10


def test_hqiv_energy_at_positions_scale():
    """energy_scale multiplies the result."""
    sys = HQIVSystem.from_atoms([(0, 0, 0), (2.0, 0, 0)])
    pos = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    E1 = hqiv_energy_at_positions(sys, pos, energy_scale=1.0)
    E2 = hqiv_energy_at_positions(sys, pos, energy_scale=2.0)
    assert abs(E2 - 2.0 * E1) < abs(E1) * 1e-10


def test_hqiv_forces_analytic_shape():
    """Forces (N, 3), symmetric pair contributions."""
    n = 3
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, 1.0, 0]])
    charges = np.zeros(n)
    forces = hqiv_forces_analytic(positions, charges, c_si=C_SI, energy_scale=1.0)
    assert forces.shape == (n, 3)
    assert np.all(np.isfinite(forces))


def test_hqiv_forces_analytic_sum_near_zero():
    """Sum of forces should be small (internal consistency) for symmetric config."""
    positions = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
    charges = np.array([0.0, 0.0])
    forces = hqiv_forces_analytic(positions, charges, energy_scale=1.0)
    # Two equal masses: F1 = -F2 so sum = 0
    np.testing.assert_array_almost_equal(forces.sum(axis=0), np.zeros(3), decimal=5)


def test_hqiv_stress_virial_shape():
    """Stress in Voigt order [xx, yy, zz, yz, xz, xy]."""
    positions = np.array([[0, 0, 0], [1, 0, 0]])
    forces = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    volume = 10.0
    stress = hqiv_stress_virial(positions, forces, volume)
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))


def test_hqiv_stress_virial_zero_volume():
    """Zero volume returns zeros (no div-by-zero)."""
    stress = hqiv_stress_virial(np.zeros((2, 3)), np.zeros((2, 3)), 0.0)
    np.testing.assert_array_almost_equal(stress, np.zeros(6))


def test_hqiv_calculator_energy_forces_stress():
    """When ASE is installed: HQIVCalculator returns energy, forces, stress."""
    try:
        from ase import Atoms
        from pyhqiv.ase_interface import HQIVCalculator
    except ImportError:
        pytest.skip("ASE not installed (pip install pyhqiv[ase])")
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4])
    calc = HQIVCalculator(gamma=GAMMA, energy_scale=1.0)
    atoms.calc = calc
    assert "energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties
    assert "stress" in calc.implemented_properties
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    S = atoms.get_stress()
    assert np.isfinite(E)
    assert F.shape == (2, 3)
    assert S.shape == (6,)
    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(S))
