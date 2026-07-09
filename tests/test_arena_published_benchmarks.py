"""Arena metrics for published protein folding, HEP decay, and SPARC readouts."""

from __future__ import annotations

import pytest


def test_published_golden_files_load():
    from pyhqiv.arena.published_benchmarks import (
        hep_decay_benchmark,
        miniprotein_fold_audit,
    )

    fold = miniprotein_fold_audit()
    assert fold["summary"]["targets"] == 11
    assert fold["summary"]["passed"] == 11

    hep = hep_decay_benchmark()
    dist = hep["summary"]["diagnostic_branching_n_sigma_distribution"]
    assert dist["open_channel_count"] == 17
    assert dist["within_3sigma_fraction"] == 1.0


def test_miniprotein_metrics_match_published_audit():
    from pyhqiv.arena.published_benchmarks import (
        miniprotein_fold_pass_fraction,
        miniprotein_mean_ca_rmsd,
        miniprotein_trp_cage_ca_rmsd,
    )

    mean_rmsd = miniprotein_mean_ca_rmsd()
    assert 1.5 < mean_rmsd < 2.5
    trp = miniprotein_trp_cage_ca_rmsd()
    assert 3.0 < trp < 5.0
    assert miniprotein_fold_pass_fraction() == 1.0


def test_hep_decay_metrics_match_published_panel():
    from pyhqiv.arena.published_benchmarks import (
        hep_decay_panel_max_z,
        hep_decay_panel_mean_z,
        hep_decay_structural_pass_rate,
    )

    mean_z = hep_decay_panel_mean_z()
    max_z = hep_decay_panel_max_z()
    assert 0.0 < mean_z < 1.0
    assert 0.0 < max_z < 1.5
    assert hep_decay_structural_pass_rate() == 1.0


def test_sparc_residual_ratio():
    from pyhqiv.arena.published_benchmarks import sparc_median_chi2_residual_ratio

    ratio = sparc_median_chi2_residual_ratio()
    assert 0.0 < ratio < 1.0


def test_arena_registry_includes_published_metrics():
    from pyhqiv.arena.metrics import METRIC_REGISTRY

    names = set(METRIC_REGISTRY())
    for name in (
        "miniprotein_mean_ca_rmsd",
        "miniprotein_trp_cage_ca_rmsd",
        "hep_decay_panel_mean_z",
        "hep_decay_panel_max_z",
        "orbital_flyby_sparc_model_residual",
        "water_phase_diagram_structural_pass_rate",
        "water_metastable_liquid_at_llcp",
    ):
        assert name in names


def test_published_metrics_score_finite():
    from pyhqiv.arena import build_default_metrics, compute_score

    res = compute_score(metrics=build_default_metrics())
    by_name = {m.name: m for m in res.metrics}
    for name in (
        "miniprotein_mean_ca_rmsd",
        "hep_decay_panel_mean_z",
        "orbital_flyby_sparc_model_residual",
    ):
        assert name in by_name
        assert by_name[name].rel_err == by_name[name].rel_err  # not NaN
