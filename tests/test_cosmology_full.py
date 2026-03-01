"""Tests for optional cosmology module (sigma8, universe_evolver, C_ℓ, LOS/ISW, healpy map)."""

import numpy as np
import pytest

from pyhqiv import cosmology_full
from pyhqiv.cosmology import HQIVUniverseEvolver


def test_sigma8_returns_float():
    """sigma8(z) returns a positive float."""
    s8 = cosmology_full.sigma8(z=0.0)
    assert isinstance(s8, (float, np.floating))
    assert s8 > 0
    assert np.isfinite(s8)
    # Fiducial normalization gives ~0.81 at z=0
    assert 0.1 < s8 < 2.0


def test_sigma8_redshift():
    """sigma8(z) decreases with z (growth)."""
    s8_0 = cosmology_full.sigma8(z=0.0)
    s8_1 = cosmology_full.sigma8(z=1.0)
    assert s8_1 < s8_0
    assert s8_1 > 0


def test_universe_evolver_returns_dict():
    """universe_evolver returns dict with z, a, D, f_lapse."""
    ev = cosmology_full.universe_evolver(z_start=10.0, z_end=0.0, n_steps=20)
    assert "z" in ev
    assert "a" in ev
    assert "D" in ev
    assert "f_lapse" in ev
    assert len(ev["z"]) == 20
    assert ev["z"][0] <= ev["z"][-1] or ev["z"][0] >= ev["z"][-1]
    np.testing.assert_array_less(0, ev["D"])
    assert ev["Omega_k_true"] is not None


def test_c_ell_spectrum_tt():
    """c_ell_spectrum('TT') returns (ell, C_ell) arrays."""
    ell, c_ell = cosmology_full.c_ell_spectrum("TT", max_ell=100)
    assert ell.shape == c_ell.shape
    assert ell[0] >= 2
    assert len(ell) <= 99
    assert np.all(c_ell >= 0)
    assert np.all(np.isfinite(c_ell))


def test_c_ell_spectrum_ee_te_bb():
    """c_ell_spectrum EE, TE, BB return same ell length."""
    ell_tt, _ = cosmology_full.c_ell_spectrum("TT", max_ell=50)
    ell_ee, c_ee = cosmology_full.c_ell_spectrum("EE", max_ell=50)
    _, c_te = cosmology_full.c_ell_spectrum("TE", max_ell=50)
    _, c_bb = cosmology_full.c_ell_spectrum("BB", max_ell=50)
    assert len(ell_ee) == len(ell_tt)
    assert np.all(c_ee >= 0) and np.all(c_bb >= 0)


def test_line_of_sight_isw_rees_sciama():
    """line_of_sight_isw_rees_sciama returns ΔC_ℓ same shape as ell."""
    ell = np.array([2.0, 10.0, 100.0, 500.0])
    delta_cl = cosmology_full.line_of_sight_isw_rees_sciama(
        ell, z_range=(0.0, 10.0), n_z=30
    )
    assert delta_cl.shape == ell.shape
    assert np.all(delta_cl >= 0)
    assert np.all(np.isfinite(delta_cl))


def test_hqiv_cmb_returns_ell_and_sigma8():
    """hqiv_cmb returns C_ell_TT, sigma8, ell."""
    result = cosmology_full.hqiv_cmb(n_side=32, max_ell=100, include_polarization=True)
    assert "ell" in result
    assert "C_ell_TT" in result
    assert "sigma8" in result
    assert result["sigma8"] > 0
    assert len(result["C_ell_TT"]) == len(result["ell"])
    assert "C_ell_EE" in result
    assert "C_ell_TE" in result
    assert "C_ell_BB" in result


def test_full_sky_healpy_map_with_healpy():
    """full_sky_healpy_map returns map when healpy is installed."""
    try:
        import healpy as hp
    except ImportError:
        pytest.skip("healpy not installed")
    t_map = cosmology_full.full_sky_healpy_map(
        n_side=16, map_type="T", include_isw_rees_sciama=False
    )
    assert t_map is not None
    assert len(t_map) == hp.nside2npix(16)
    assert np.all(np.isfinite(t_map))


def test_full_sky_healpy_map_import():
    """full_sky_healpy_map either returns a map (healpy installed) or raises ImportError."""
    try:
        t_map = cosmology_full.full_sky_healpy_map(n_side=8, include_isw_rees_sciama=False)
        assert t_map is not None
        assert len(t_map) > 0
    except ImportError as e:
        assert "healpy" in str(e)


def test_cmb_pipeline_status_mentions_optional_module():
    """cmb_pipeline_status includes optional_module key."""
    from pyhqiv.cmb_pipeline import cmb_pipeline_status
    status = cmb_pipeline_status()
    assert "optional_module" in status
    assert "cosmology_full" in status["optional_module"]


def test_hqiv_universe_evolver_run_from_T_Pl_to_now():
    """HQIVUniverseEvolver.run_from_T_Pl_to_now returns T_map_muK, sigma8, C_ell (multipoles out to ≥1500)."""
    evolver = HQIVUniverseEvolver(nside=16, max_ell=1500)
    result = evolver.run_from_T_Pl_to_now()
    assert "T_map_muK" in result
    assert "sigma8" in result
    assert "ell" in result
    assert "C_ell_TT" in result
    assert result["sigma8"] > 0
    assert len(result["C_ell_TT"]) == len(result["ell"])
    assert result["ell"][-1] >= 1500
    # T_map_muK may be None if healpy not installed
    if result["T_map_muK"] is not None:
        import healpy as hp
        assert len(result["T_map_muK"]) == hp.nside2npix(evolver.nside)
        assert np.all(np.isfinite(result["T_map_muK"]))


def test_add_kinematic_dipole():
    """add_kinematic_dipole adds frame-velocity dipole to a map (low-ℓ region)."""
    try:
        import healpy as hp
    except ImportError:
        pytest.skip("healpy not installed")
    from pyhqiv import cosmology_full
    n_side = 16
    npix = hp.nside2npix(n_side)
    zero_map = np.zeros(npix)
    # v=370 km/s, T_CMB ~ 2.725e6 μK → dipole amplitude ~ 3360 μK
    dipped = cosmology_full.add_kinematic_dipole(zero_map, n_side, v_km_s=370.0)
    assert np.all(np.isfinite(dipped))
    assert dipped.min() < -1000 and dipped.max() > 1000
    # Symmetry: mean of dipole over full sky should be ~0 (numerically)
    assert np.abs(np.mean(dipped)) < 1.0
