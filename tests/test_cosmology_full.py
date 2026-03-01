"""Tests for optional cosmology module stubs (universe_evolver, hqiv_cmb, σ₈, C_ℓ, LOS/ISW)."""

import pytest

from pyhqiv import cosmology_full


def test_universe_evolver_stub():
    """universe_evolver raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc:
        cosmology_full.universe_evolver()
    assert "Full cosmology" in str(exc.value) or "universe" in str(exc.value).lower()


def test_hqiv_cmb_stub():
    """hqiv_cmb raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        cosmology_full.hqiv_cmb(n_side=64)


def test_sigma8_stub():
    """sigma8 raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        cosmology_full.sigma8(z=0.0)


def test_c_ell_spectrum_stub():
    """c_ell_spectrum raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        cosmology_full.c_ell_spectrum(spectrum_type="TT", max_ell=100)


def test_full_sky_healpy_map_stub():
    """full_sky_healpy_map raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        cosmology_full.full_sky_healpy_map(n_side=64)


def test_line_of_sight_isw_rees_sciama_stub():
    """line_of_sight_isw_rees_sciama raises NotImplementedError."""
    import numpy as np
    with pytest.raises(NotImplementedError):
        cosmology_full.line_of_sight_isw_rees_sciama(ell=np.array([2, 10, 100]))


def test_cmb_pipeline_status_mentions_optional_module():
    """cmb_pipeline_status includes optional_module key."""
    from pyhqiv.cmb_pipeline import cmb_pipeline_status
    status = cmb_pipeline_status()
    assert "optional_module" in status
    assert "cosmology_full" in status["optional_module"]
