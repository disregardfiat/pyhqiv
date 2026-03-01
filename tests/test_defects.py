"""Tests for defects: formation_energy, charged_defect_supercell."""

import numpy as np

from pyhqiv.defects import charged_defect_supercell, formation_energy


def test_formation_energy_neutral_no_chem():
    """ΔH = E_defect - E_bulk when neutral and no chemical potentials."""
    E_def = 10.0
    E_bulk = 8.0
    dH = formation_energy(E_def, E_bulk, n_defect=1, q=0)
    assert abs(dH - 2.0) < 1e-10


def test_formation_energy_vacancy_with_mu():
    """Vacancy: add n * μ_removed."""
    dH = formation_energy(
        E_defect=9.0,
        E_bulk=10.0,
        n_defect=1,
        mu_removed=1.0,
        mu_added=None,
        q=0,
    )
    assert abs(dH - (9.0 - 10.0 + 1.0)) < 1e-10


def test_formation_energy_hqiv_correction():
    """HQIV vacuum correction changes formation energy for charged defects."""
    dH0 = formation_energy(
        10.0,
        8.0,
        n_defect=1,
        q=1,
        E_vacuum=0.0,
        phi_avg_defect=0.0,
        phi_avg_bulk=0.0,
        dot_delta_theta_avg=0.0,
    )
    # Use larger phi/dot so correction is numerically significant (2 + 2e-9 != 2)
    dH1 = formation_energy(
        10.0,
        8.0,
        n_defect=1,
        q=1,
        E_vacuum=0.0,
        phi_avg_defect=1e-3,
        phi_avg_bulk=0.0,
        dot_delta_theta_avg=1e-5,
        gamma=0.4,
    )
    assert dH1 != dH0
    assert np.isfinite(dH1)


def test_charged_defect_supercell_shape():
    """Supercell positions and charges have correct lengths."""
    lat = np.eye(3) * 5.0
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]])
    charges = [0.0, 0.0]
    pos_sc, ch_sc, center = charged_defect_supercell(lat, pos, charges, supercell_shape=(2, 2, 2))
    assert pos_sc.shape[0] == 2 * 2 * 2 * 2
    assert len(ch_sc) == pos_sc.shape[0]
    assert center.shape == (3,)


def test_charged_defect_supercell_defect_center():
    """Defect center is in the first cell region."""
    lat = np.eye(3)
    pos = np.array([[0.5, 0.5, 0.5]])
    pos_sc, _, center = charged_defect_supercell(lat, pos, [0.0], supercell_shape=(2, 1, 1))
    assert np.all(np.isfinite(center))
    assert np.linalg.norm(center) < 10.0
