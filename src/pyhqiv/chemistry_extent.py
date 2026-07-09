"""Compact readouts for the light-cone chemistry extent paper.

The functions here summarize witness payloads. They do not contain laboratory
comparison tables or fitted coefficients; tests/Arena supply quarantined paper
payloads and goldens.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _mean_abs(values: list[Any]) -> float:
    clean = [abs(float(v)) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        raise RuntimeError("chemistry extent summary: empty finite value list")
    return sum(clean) / len(clean)


def summarize_chemistry_panel_accuracy(payload: Mapping[str, Any]) -> dict[str, Any]:
    spectral = payload["spectral"]
    carbon = payload["carbon"]
    carbon_rows = list(carbon.get("rows") or [])
    return {
        "spectral_geometric_mean_error_pct": dict(spectral["geometric_mean_error_pct"]),
        "spectral_reliable_fraction": float(spectral["reliable_count"])
        / float(spectral["count"]),
        "carbon_density_mean_abs_error_pct": _mean_abs(
            [row.get("error_pct") for row in carbon_rows]
        ),
        "carbon_bond_mean_abs_error_pct": _mean_abs(
            [row.get("bond_error_pct") for row in carbon_rows]
        ),
        "carbon_count": int(carbon["count"]),
    }


def summarize_molecule_suite(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    combined = summary["combined_core_plus_expanded"]
    return {
        "core_mean_abs_binding_error_pct": float(summary["core"]["mean_abs_error_pct"]),
        "combined_mean_abs_binding_error_pct": float(combined["mean_abs_error_pct"]),
        "open_shell_mean_abs_binding_error_pct": float(
            summary["open_shell"]["mean_abs_error_pct"]
        ),
        "combined_within_15pct_fraction": float(combined["within_15pct"])
        / float(combined["count"]),
        "total_molecules": int(summary["total_molecules"]),
    }


def summarize_constraint_system(payload: Mapping[str, Any]) -> dict[str, Any]:
    spectroscopy = payload["spectroscopy"]
    rank = float(spectroscopy["rank"])
    null_dim = float(spectroscopy["null_dim"])
    return {
        "n_equations": int(payload["n_equations"]),
        "slot_count": len(payload["slot_catalog"]),
        "condensed_resid_norm": float(payload["condensed"]["resid_norm"]),
        "binding_resid_norm": float(payload["binding"]["resid_norm"]),
        "spectroscopy_rank_fraction": rank / (rank + null_dim),
    }


def summarize_inverse_channel_solve(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gmtkn_resid_norm": float(payload["gmtkn_3slot"]["resid_norm"]),
        "outside_curvature_gas_abs_participation": abs(
            float(payload["outside_curvature_gas_participation_w"])
        ),
        "spectroscopy_resid_norm": float(payload["spectroscopy_x"]["resid_norm"]),
    }


def summarize_nested_wf_geometry(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    count = float(summary["count"])
    return {
        "mean_abs_error_pct": float(summary["mean_abs_error_pct"]),
        "within_15pct_fraction": float(summary["within_15pct"]) / count,
        "within_5pct_fraction": float(summary["within_5pct"]) / count,
        "count": int(summary["count"]),
    }


def summarize_quantum_chem_witnesses(payload: Mapping[str, Any]) -> dict[str, Any]:
    bridge = payload["lean_lih_compton_bridge"]
    theorem_values = list(bridge["imprint_phase_theorems"].values())
    primary = next(
        row
        for row in payload["lih_dynamic_binding"]["binding_readouts"]
        if row["mode"] == "dynamic_compton_participation"
    )
    chart_summary = payload["dynamic_binding_chart"]["summary"]
    return {
        "h2_trace_match_fraction": 1.0
        if payload["h2_trace_referenceM"] == payload["h2_trace_referenceM_expected"]
        else 0.0,
        "lih_imprint_theorem_fraction": sum(1 for v in theorem_values if v)
        / len(theorem_values),
        "lih_dynamic_primary_error_pct": abs(float(primary["error_pct"])),
        "dynamic_chart_mean_abs_error_pct": float(chart_summary["mean_abs_error_pct"]),
        "dynamic_chart_within_15pct_fraction": float(chart_summary["within_15pct"])
        / float(chart_summary["count"]),
    }


def summarize_contact_network_rules(payload: Mapping[str, Any]) -> dict[str, Any]:
    networks = list(payload["networks"])
    rule_counts = [len(n.get("rules", [])) for n in networks]
    contact_counts = [len(n.get("contacts", [])) for n in networks]
    return {
        "contact_kind_count": len(payload["contact_kinds"]),
        "derived_phase_count": len(payload["derived_phases"]),
        "network_count": len(networks),
        "network_with_rules_fraction": sum(1 for c in rule_counts if c > 0)
        / len(networks),
        "network_with_contacts_fraction": sum(1 for c in contact_counts if c > 0)
        / len(networks),
    }


def summarize_allotrope_phase_cooling(payload: Mapping[str, Any]) -> dict[str, Any]:
    species = dict(payload["species"])
    transition_counts = [len(s.get("transitions", [])) for s in species.values()]
    profile_counts = [
        len(s.get("allotrope_spectroscopy_profiles", [])) for s in species.values()
    ]
    return {
        "molecule_count": len(payload["molecules"]),
        "species_count": len(species),
        "transition_coverage_fraction": sum(1 for c in transition_counts if c > 0)
        / len(transition_counts),
        "profile_coverage_fraction": sum(1 for c in profile_counts if c > 0)
        / len(profile_counts),
    }


def _max_available_abs_correlation(section: Mapping[str, Any]) -> float:
    values: list[float] = []
    for correlations_by_feature in section.get("correlations", {}).values():
        for stat in correlations_by_feature.values():
            if stat.get("available"):
                values.append(abs(float(stat["pearson_r"])))
    return max(values) if values else 0.0


def summarize_residual_correlation_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    condensed = payload["condensed_phase"]
    spectroscopy = payload["spectroscopy"]
    coupled_count = len(spectroscopy.get("coupled_relaxation_improvements", [])) + len(
        condensed.get("optical_coupled_relaxation_improvements", [])
    )
    return {
        "condensed_n": int(condensed["n"]),
        "spectroscopy_n": int(spectroscopy["n"]),
        "spectroscopy_reliable_fraction": float(spectroscopy["n_reliable_geometry"])
        / float(spectroscopy["n"]),
        "condensed_max_abs_correlation": _max_available_abs_correlation(condensed),
        "spectroscopy_max_abs_correlation": _max_available_abs_correlation(spectroscopy),
        "in_bracket_flow_target_count": len(
            spectroscopy.get("in_bracket_flow_targets", [])
        ),
        "coupled_relaxation_improvement_count": coupled_count,
    }


def summarize_generator_dependent_coupling(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    rows = list(payload["rows"])
    abelian = summary["abelian"]
    spectral_gap = summary["spectral_gap"]
    return {
        "abelian_mean_abs_error_pct": float(abelian["mean_abs_error_pct"]),
        "spectral_gap_mean_abs_error_pct": float(
            spectral_gap["mean_abs_error_pct"]
        ),
        "spectral_gap_improvement_pct": float(abelian["mean_abs_error_pct"])
        - float(spectral_gap["mean_abs_error_pct"]),
        "spectral_gap_within5_fraction": float(spectral_gap["within_5pct"])
        / len(rows),
        "recommendation_improved": 1.0
        if payload["recommendation"].get("improved_vs_abelian")
        else 0.0,
    }


def summarize_system_matrix_functor(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    return {
        "base_mean_abs_error_pct": float(summary["base"]["mean_abs_error_pct"]),
        "so8_blend_mean_abs_error_pct": float(
            summary["so8_blend_relative"]["mean_abs_error_pct"]
        ),
        "contact_relative_mean_abs_error_pct": float(
            summary["contact_relative"]["mean_abs_error_pct"]
        ),
        "best_is_base_fraction": 1.0
        if payload["recommendation"].get("best_mean_variant") == "base"
        else 0.0,
        "row_count": len(payload["rows"]),
    }


def summarize_second_order_effect(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    rows = list(payload["rows"])
    outside_geff = summary["outside_geff"]
    return {
        "base_mean_abs_error_pct": float(summary["base"]["mean_abs_error_pct"]),
        "outside_geff_mean_abs_error_pct": float(
            outside_geff["mean_abs_error_pct"]
        ),
        "outside_geff_within5_fraction": float(outside_geff["within_5pct"])
        / len(rows),
        "promote_outside_geff_fraction": 1.0
        if payload["recommendation"].get("promote_candidate") == "outside_geff"
        else 0.0,
    }


def summarize_crystal_ethics(payload: Mapping[str, Any]) -> dict[str, Any]:
    lean_rows = list(payload["lean_proof_audit"])
    az_rows = list(payload["a_z_audit"])
    regime_rows = list(payload["regime_audit"])
    return {
        "passes_fraction": 1.0 if payload["passes"] else 0.0,
        "lean_module_pass_fraction": sum(1 for row in lean_rows if row["passes"])
        / len(lean_rows),
        "az_policy_pass_fraction": sum(
            1
            for row in az_rows
            if (not row["uses_sparse_light_override"]) and row["override_allowed"]
        )
        / len(az_rows),
        "regime_match_fraction": sum(
            1 for row in regime_rows if row["expected"] == row["actual"]
        )
        / len(regime_rows),
    }


def build_chemistry_extent_domain_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build compact Arena-facing summaries from paper witness payloads."""
    return {
        "schema_version": 1,
        "source": "lightcone_chemistry_extent paper witness payloads",
        "comparison_policy": "External chemistry data grade readouts only; no DFT/table fit inputs.",
        "chemistry_panel_accuracy": summarize_chemistry_panel_accuracy(
            payloads["chemistry_panel_accuracy"]
        ),
        "molecule_suite": summarize_molecule_suite(payloads["molecule_suite_audit"]),
        "constraint_system": summarize_constraint_system(
            payloads["chemistry_constraint_system"]
        ),
        "inverse_channel_solve": summarize_inverse_channel_solve(
            payloads["chemistry_inverse_channel_solve"]
        ),
        "nested_wf_geometry": summarize_nested_wf_geometry(
            payloads["nested_wf_geometry"]
        ),
        "quantum_chem_witnesses": summarize_quantum_chem_witnesses(
            payloads["quantum_chem_witnesses"]
        ),
        "contact_network_rules": summarize_contact_network_rules(
            payloads["curvature_contact_network_rules"]
        ),
        "allotrope_phase_cooling": summarize_allotrope_phase_cooling(
            payloads["allotrope_phase_cooling_audit"]
        ),
        "residual_correlation_audit": summarize_residual_correlation_audit(
            payloads["chemistry_residual_correlation_audit"]
        ),
        "generator_dependent_coupling": summarize_generator_dependent_coupling(
            payloads["generator_dependent_coupling_audit"]
        ),
        "system_matrix_functor": summarize_system_matrix_functor(
            payloads["system_matrix_functor_audit"]
        ),
        "second_order_effect": summarize_second_order_effect(
            payloads["second_order_effect_audit"]
        ),
        "crystal_ethics": summarize_crystal_ethics(payloads["crystal_ethics_audit"]),
    }
