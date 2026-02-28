"""Tests for HQIVSystem and HQIVAtom."""

import numpy as np
import pytest

from pyhqiv.atom import HQIVAtom
from pyhqiv.system import HQIVSystem
from pyhqiv.constants import GAMMA


def test_atom_phi_local():
    at = HQIVAtom(position=(0, 0, 0), charge=1)
    x = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    phi = at.phi_local(x)
    assert phi.shape == (2,)
    assert phi[0] > phi[1]


def test_system_from_atoms():
    sys = HQIVSystem.from_atoms([(0, 0, 0), (1.5, 0, 0)], charges=[1, -1], gamma=0.40)
    assert len(sys.atoms) == 2
    assert sys.gamma == 0.40
    np.testing.assert_array_almost_equal(sys.positions[0], [0, 0, 0])
    np.testing.assert_array_almost_equal(sys.positions[1], [1.5, 0, 0])


def test_compute_fields_shape():
    sys = HQIVSystem.from_atoms([(0, 0, 0), (1, 0, 0)], charges=[1, -1])
    grid = np.mgrid[-2:2:5j, -2:2:5j, -2:2:5j].reshape(3, -1).T
    E, B = sys.compute_fields(grid, t=0.0)
    assert E.shape == (grid.shape[0], 3)
    assert B.shape == (grid.shape[0], 3)


def test_compute_fields_reproducible():
    sys = HQIVSystem.from_atoms([(0, 0, 0)], charges=[1])
    grid = np.array([[1.0, 0.0, 0.0]])
    E1, _ = sys.compute_fields(grid, t=0.0)
    E2, _ = sys.compute_fields(grid, t=0.0)
    np.testing.assert_array_almost_equal(E1, E2)
