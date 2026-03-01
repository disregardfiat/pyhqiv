"""Tests for CMB pipeline stub: status, run() without map generation."""

import pytest

from pyhqiv.cmb_pipeline import HQIVCMBPipeline, cmb_pipeline_status


def test_cmb_pipeline_status():
    """cmb_pipeline_status returns dict with implemented vs not-implemented."""
    status = cmb_pipeline_status()
    assert status["background"] == "implemented"
    assert status["perturbations_class"] == "implemented"
    assert status["boltzmann_hierarchy"] == "not_implemented"
    assert "implemented" in status["line_of_sight_integration"] or status["line_of_sight_integration"] == "not_implemented"
    assert "Omega_k_true" in status
    assert status.get("Omega_k_true") is not None
    assert "design_doc" in status
    assert "optional_module" in status


def test_hqiv_cmb_pipeline_run_no_map():
    """Pipeline.run() without n_side returns background + perturbation info."""
    pipeline = HQIVCMBPipeline()
    result = pipeline.run(z_rec=1100.0, n_side=None)
    assert result["Omega_k_true"] is not None
    assert result["lapse_compression"] is not None
    assert result["age_apparent_Gyr"] is not None
    assert "delta_growth_z0" in result
    assert "lapse_factor_z0" in result
    assert result["z_rec"] == 1100.0
    assert 0 < result["lapse_factor_z0"] <= 1


def test_hqiv_cmb_pipeline_run_map_not_implemented():
    """Pipeline.run(n_side=256) raises NotImplementedError."""
    pipeline = HQIVCMBPipeline()
    with pytest.raises(NotImplementedError) as exc_info:
        pipeline.run(z_rec=1100.0, n_side=256)
    assert "Full CMB map generation" in str(exc_info.value)
    assert "HQIV_CMB_Pipeline" in str(exc_info.value)


def test_hqiv_cmb_pipeline_perturbations_lazy():
    """Perturbations property builds HQIVPerturbations around cosmology."""
    pipeline = HQIVCMBPipeline()
    assert pipeline._pert is None
    p = pipeline.perturbations
    assert p is not None
    assert p.background is pipeline.cosmology
    assert pipeline._pert is p
