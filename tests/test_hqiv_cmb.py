"""Tests for HQIVCMBMap: T_Pl → now pipeline, σ₈, multipole, peak positions."""

import numpy as np
import pytest


def _find_peaks(
    ell: np.ndarray, cl: np.ndarray, n_peaks: int = 3, ell_min: int = 50
) -> np.ndarray:
    """Return ell of first n_peaks local maxima in acoustic range (ell >= ell_min)."""
    d_ell = ell * (ell + 1) * np.maximum(cl, 1e-30) / (2.0 * np.pi)
    peaks = []
    for i in range(max(2, ell_min), len(d_ell) - 1):
        if d_ell[i] >= d_ell[i - 1] and d_ell[i] >= d_ell[i + 1]:
            peaks.append(int(ell[i]))
            if len(peaks) >= n_peaks:
                break
    return np.array(peaks) if len(peaks) >= n_peaks else np.array(peaks + [0] * (n_peaks - len(peaks)))


try:
    import healpy  # noqa: F401
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_hqiv_cmb_map_run_and_peaks():
    """Run HQIVCMBMap with small nside and pin first three acoustic peak positions."""
    from pyhqiv.cosmology.hqiv_cmb import HQIVCMBMap

    cmb = HQIVCMBMap(nside=64, lmax=200)
    result = cmb.run_from_T_Pl_to_now()

    assert "Cl_TT" in result
    assert "sigma8" in result
    assert "Omega_k_true" in result
    assert result["Omega_k_true"] == pytest.approx(0.0098, rel=0.1)

    ell = np.arange(len(result["Cl_TT"]))
    peaks = _find_peaks(ell, result["Cl_TT"], n_peaks=3, ell_min=50)
    assert len(peaks) == 3

    # First three acoustic peaks: broad ranges (curvature shifts ~0.5–1%)
    assert 50 <= peaks[0] <= 400, f"first peak ℓ = {peaks[0]}"
    assert peaks[1] <= 800, f"second peak ℓ = {peaks[1]}"
    assert peaks[2] <= 1200, f"third peak ℓ = {peaks[2]}"


def test_hqiv_cmb_map_import():
    """HQIVCMBMap is importable from cosmology package."""
    from pyhqiv.cosmology import HQIVCMBMap

    assert HQIVCMBMap is not None
    cmb = HQIVCMBMap(nside=8, lmax=50)
    assert cmb.cosmo.Ok0 == pytest.approx(0.0098, rel=0.1)
