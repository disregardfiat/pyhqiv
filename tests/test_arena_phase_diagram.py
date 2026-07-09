"""Arena metrics for generalized HQIV phase diagrams (H₂O LLPT branch)."""

from __future__ import annotations


def test_phase_diagram_golden_loads():
    from pyhqiv.arena.published_benchmarks import phase_diagram_audit

    audit = phase_diagram_audit()
    assert audit["structural_expectations"]["cytosol_phase"] == "liquid"
    assert "anchor_points" in audit
    assert audit["end_members"]["high_density"]["rho_curv"] == 1.0


def test_water_structural_pass_rate():
    from pyhqiv.arena.published_benchmarks import water_phase_diagram_structural_pass_rate

    assert water_phase_diagram_structural_pass_rate() == 1.0


def test_water_metastable_at_llcp():
    from pyhqiv.arena.published_benchmarks import water_metastable_liquid_at_llcp

    assert water_metastable_liquid_at_llcp() == 1.0


def test_water_melt_residual_in_window():
    from pyhqiv.arena.published_benchmarks import water_h2o_melt_temperature_K

    t_melt = water_h2o_melt_temperature_K()
    assert 270.0 <= t_melt <= 276.0


def test_arena_registry_includes_phase_diagram_metrics():
    from pyhqiv.arena.metrics import METRIC_REGISTRY

    names = set(METRIC_REGISTRY())
    for name in (
        "water_h2o_melt_T_residual_K",
        "water_phase_diagram_structural_pass_rate",
        "water_metastable_liquid_at_llcp",
        "water_llcp_observation_distance",
        "water_widom_peak_temperature_residual_K",
        "water_widom_free_energy_peak_residual_K",
        "water_widom_gamma2_window_alignment_K",
        "water_nucleation_defect_ldl_excess",
        "water_hoh_angle_taxonomy_open_gap_deg",
        "water_h2o_bond_angle_residual_deg",
        "protein_hydrophobic_interface_ldl_excess",
        "thermo_allotrope_phase_residual",
    ):
        assert name in names


def test_free_energy_widom_and_nucleation_witnesses():
    from pyhqiv.arena.published_benchmarks import (
        phase_diagram_audit,
        water_nucleation_defect_ldl_excess,
        water_h2o_bond_angle_residual_deg,
        water_widom_gamma2_window_alignment_K,
        water_widom_peak_temperature_residual_K,
    )

    audit = phase_diagram_audit()
    assert "widom_free_energy" in audit
    assert "hoh_angle_witness" in audit
    assert "nucleation_defect_witness" in audit
    assert water_widom_peak_temperature_residual_K() <= 5.0
    assert water_widom_gamma2_window_alignment_K() <= 2.0
    assert water_nucleation_defect_ldl_excess() > 0.0
    assert water_h2o_bond_angle_residual_deg() < 0.1


def test_phase_diagram_metrics_score_finite():
    from pyhqiv.arena import build_default_metrics, compute_score

    res = compute_score(metrics=build_default_metrics())
    by_name = {m.name: m for m in res.metrics}
    for name in (
        "water_phase_diagram_structural_pass_rate",
        "water_metastable_liquid_at_llcp",
        "water_h2o_melt_T_residual_K",
        "water_widom_free_energy_peak_residual_K",
        "water_nucleation_defect_ldl_excess",
    ):
        assert name in by_name
        assert by_name[name].rel_err == by_name[name].rel_err
