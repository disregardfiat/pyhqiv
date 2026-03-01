"""Tests for HQIVPerturbations: stellar modes, fluid stability, phonons, cosmology, linear response."""

import numpy as np
import pytest

from pyhqiv.constants import ALPHA, C_SI, GAMMA
from pyhqiv.perturbations import HQIVPerturbations, PerturbationMode
from pyhqiv.system import HQIVSystem


def test_perturbation_mode():
    """PerturbationMode stores omega, eigenvec, period, growth_time."""
    omega = complex(1e-3, -1e-6)
    vec = np.ones(10)
    m = PerturbationMode(omega, vec, mode_type="l=1 n=0")
    assert m.omega == omega
    assert m.eigenvec.shape == (10,)
    assert m.type == "l=1 n=0"
    assert np.isfinite(m.period) and m.period > 0
    # growth_time = 1/im; negative im (decay) => negative growth_time
    assert np.isfinite(m.growth_time)


def test_hqiv_perturbations_with_system():
    """HQIVPerturbations accepts HQIVSystem as background."""
    sys = HQIVSystem.from_atoms([(0, 0, 0), (1, 0, 0)], charges=[0, 0])
    pert = HQIVPerturbations(sys, gamma=GAMMA, alpha=ALPHA)
    assert pert.background is sys
    assert pert.lattice.gamma == GAMMA
    assert pert.phase_lift.gamma == GAMMA


def test_stellar_oscillations_returns_modes():
    """stellar_oscillations returns list of PerturbationMode with finite periods."""
    sys = HQIVSystem.from_atoms([(0, 0, 0)])
    pert = HQIVPerturbations(sys)
    modes = pert.stellar_oscillations(l=1, n_max=5, r_points=50)
    assert len(modes) <= 5
    assert len(modes) >= 1
    for m in modes:
        assert isinstance(m, PerturbationMode)
        assert np.isfinite(m.period) or m.period == np.inf
        assert m.eigenvec.shape[0] in (50, 100)


def test_stellar_oscillations_with_solar_core():
    """With HQIVSolarCore background, modes use solar radius and density."""
    pytest.importorskip("pyhqiv.solar_core")
    from pyhqiv.solar_core import HQIVSolarCore

    sun = HQIVSolarCore()
    pert = HQIVPerturbations(sun)
    modes = pert.stellar_oscillations(l=1, n_max=3, r_points=30)
    assert len(modes) >= 1
    assert all(hasattr(m, "period") for m in modes)


def test_fluid_instability():
    """fluid_instability returns growth rate array with lapse correction."""
    pert = HQIVPerturbations(HQIVSystem.from_atoms([(0, 0, 0)]))
    k = np.array([0.1, 1.0, 10.0])
    growth = pert.fluid_instability(k)
    assert growth.shape == k.shape
    assert np.all(growth >= 0)
    assert np.all(np.isfinite(growth))


def test_phonon_spectrum():
    """phonon_spectrum returns ω(q) array."""
    pert = HQIVPerturbations(HQIVSystem.from_atoms([(0, 0, 0)]))
    q = np.array([[0.01, 0, 0], [0.1, 0.1, 0]])
    omega = pert.phonon_spectrum(q, omega_scale=1e13)
    assert omega.shape == (2,)
    assert np.all(omega >= 0)
    assert np.all(np.isfinite(omega))


def test_cosmological_perturbation():
    """cosmological_perturbation returns (delta_growth, f)."""
    pert = HQIVPerturbations(HQIVSystem.from_atoms([(0, 0, 0)]))
    delta_growth, f = pert.cosmological_perturbation(k=0.1, z=1.0)
    assert np.isfinite(delta_growth)
    assert np.isfinite(f)
    assert 0 < f <= 1.0


def test_linear_response():
    """linear_response returns complex array with lapse/imprint scaling."""
    pert = HQIVPerturbations(HQIVSystem.from_atoms([(0, 0, 0)]))
    resp = pert.linear_response("density", omega=1e-3, m_shell=10)
    assert resp.shape == (1,)
    assert np.iscomplexobj(resp)
    assert np.all(np.isfinite(resp))


def test_summary():
    """summary returns dict with background_type, gamma, typical_lapse."""
    pert = HQIVPerturbations(HQIVSystem.from_atoms([(0, 0, 0)]))
    s = pert.summary()
    assert s["background_type"] == "HQIVSystem"
    assert s["gamma"] == GAMMA
    assert "typical_lapse" in s
    assert 0 < s["typical_lapse"] <= 1
    assert "modes_computed" in s


def test_phi_and_f_lapse_consistent():
    """_phi and _f_lapse use paper relation f = a/(a+φ/6)."""
    pert = HQIVPerturbations(HQIVSystem.from_atoms([(0, 0, 0)]))
    phi = pert._phi(1.0)
    phi = np.atleast_1d(phi).flat[0]
    f = pert._f_lapse(phi, a_loc=C_SI**2)
    f = np.atleast_1d(f).flat[0]
    expected = (C_SI**2) / ((C_SI**2) + phi / 6.0)
    assert abs(f - expected) < 0.01
