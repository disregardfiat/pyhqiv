"""Tests for birefringence-based redshift decomposition in pyhqiv.polarization."""

import numpy as np

from pyhqiv import HQIVCosmology, RedshiftDecomposition, decompose_redshift


def test_decompose_redshift_basic_api():
    """decompose_redshift returns a RedshiftDecomposition with sensible fields."""
    cosmo = HQIVCosmology()
    result = decompose_redshift(z_obs=0.5, beta=0.01, source_type="galaxy", cosmology=cosmo)
    assert isinstance(result, RedshiftDecomposition)
    assert np.all(result.z_obs >= 0.0)
    assert np.all(result.z_rec >= 0.0)
    assert np.all(result.z_lapse >= 0.0)
    # Mass factor is 1 + z_lapse
    np.testing.assert_allclose(result.implied_mass_factor, 1.0 + result.z_lapse)
    # Predicted beta should be finite
    assert np.all(np.isfinite(result.beta_predicted_from_z_rec))


def test_decompose_redshift_beta_none_fallback():
    """When beta is None, the decomposition falls back to z_rec ≈ z_obs and z_lapse ≈ 0."""
    cosmo = HQIVCosmology()
    result = decompose_redshift(z_obs=0.3, beta=None, cosmology=cosmo)
    np.testing.assert_allclose(result.z_rec, result.z_obs)
    np.testing.assert_allclose(result.z_lapse, 0.0)


def test_redshift_decomposition_to_dict_and_dataframe():
    """RedshiftDecomposition.to_dict() always works; to_dataframe requires pandas."""
    cosmo = HQIVCosmology()
    result = decompose_redshift(z_obs=0.1, beta=0.005, cosmology=cosmo)
    d = result.to_dict()
    assert "z_obs" in d and "z_rec" in d and "z_lapse" in d
    # to_dataframe may fail if pandas not installed; this is acceptable.
    try:
        df = result.to_dataframe()
        assert df.shape[0] == 1
    except ImportError:
        pass

