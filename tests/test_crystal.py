"""Tests for HQIVCrystal (PBC, bloch_sum) and band-gap / potential shift."""

import numpy as np
import pytest

from pyhqiv.crystal import HQIVCrystal, hqiv_potential_shift, high_symmetry_k_path
from pyhqiv.atom import HQIVAtom


def test_hqiv_crystal_lattice_vectors():
    """HQIVCrystal stores a1, a2, a3."""
    a1, a2, a3 = [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]
    atoms = [HQIVAtom([0, 0, 0], charge=1), HQIVAtom([0.5, 0.5, 0], charge=-1)]
    lat = np.array([a1, a2, a3])
    cry = HQIVCrystal(atoms, lattice_vectors=lat, supercell_shape=(1, 1, 1))
    np.testing.assert_array_almost_equal(cry.a1, a1)
    np.testing.assert_array_almost_equal(cry.supercell_positions().shape[0], 2)


def test_bloch_sum():
    """bloch_sum at k=0 returns sum of charges over supercell."""
    atoms = [HQIVAtom([0, 0, 0], charge=1), HQIVAtom([0.5, 0, 0], charge=-1)]
    lat = np.eye(3)
    cry = HQIVCrystal(atoms, lattice_vectors=lat, supercell_shape=(1, 1, 1))
    s = cry.bloch_sum([0, 0, 0])
    assert abs(s.real - 0.0) < 0.01 and abs(s.imag) < 0.01


def test_supercell_replication():
    """Supercell (2,1,1) doubles the number of positions."""
    atoms = [HQIVAtom([0, 0, 0], charge=0)]
    lat = np.eye(3)
    cry = HQIVCrystal(atoms, lattice_vectors=lat, supercell_shape=(2, 1, 1))
    pos = cry.supercell_positions()
    assert pos.shape[0] == 2
    np.testing.assert_array_almost_equal(pos[1] - pos[0], [1, 0, 0])


def test_hqiv_potential_shift():
    """V_shift = γ φ δ̇θ′."""
    v = hqiv_potential_shift(phi_avg=1e-10, dot_delta_theta_avg=1e-18, gamma=0.4)
    assert abs(v - 0.4 * 1e-10 * 1e-18) < 1e-35


def test_reciprocal_vectors():
    """Reciprocal vectors b_i·a_j = 2π δ_ij."""
    lat = 5.0 * np.eye(3)
    cry = HQIVCrystal(
        [HQIVAtom([0, 0, 0], charge=0)],
        lattice_vectors=lat,
        supercell_shape=(1, 1, 1),
    )
    rec = cry.reciprocal_vectors()
    assert rec.shape == (3, 3)
    np.testing.assert_array_almost_equal(cry.volume(), 125.0)
    dot = rec @ lat.T
    np.testing.assert_array_almost_equal(dot, 2 * np.pi * np.eye(3))


def test_high_symmetry_k_path():
    """K-path returns Cartesian and fractional k-points and segment labels."""
    lat = 5.0 * np.eye(3)
    k_cart, k_frac, segments = high_symmetry_k_path(lat, "GX", npoints=10)
    assert k_cart.shape == (10, 3)
    assert k_frac.shape == (10, 3)
    assert len(segments) >= 2
    assert segments[0][0] == "G" and segments[1][0] == "X"
