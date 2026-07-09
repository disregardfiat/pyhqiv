"""Arena metrics for lightcone_chemistry_extent paper witnesses."""

from __future__ import annotations

import unittest

from pyhqiv.chemistry_extent import build_chemistry_extent_domain_summary
from pyhqiv.arena import metrics as arena_metrics
from pyhqiv.arena import published_benchmarks as pb


class ChemistryExtentArenaTests(unittest.TestCase):
    def test_spectroscopy_golden_loads(self) -> None:
        payload = pb.molecular_spectroscopy_audit()
        self.assertEqual(payload["parameter_policy"], "no_fitted_coefficients")
        self.assertEqual(int(payload["summary"]["count"]), 11)
        self.assertGreaterEqual(int(payload["summary"]["count_reliable_geometry"]), 9)

    def test_chemistry_spectroscopy_arena_metrics(self) -> None:
        omega = pb.chemistry_spectroscopy_reliable_omega_e_err_pct()
        re = pb.chemistry_spectroscopy_reliable_r_e_err_pct()
        frac = pb.chemistry_spectroscopy_geometry_reliable_fraction()
        brk = pb.chemistry_spectroscopy_concentration_bracket_hit_rate()
        self.assertGreaterEqual(omega, 0.0)
        self.assertLess(omega, 20.0)
        self.assertGreaterEqual(re, 0.0)
        self.assertLess(re, 20.0)
        self.assertGreaterEqual(frac, 9 / 11)
        self.assertGreaterEqual(brk, 7 / 11)

    def test_condensed_and_crystal_panels(self) -> None:
        n_err = pb.chemistry_condensed_phase_mean_n_err_pct()
        t_err = pb.chemistry_condensed_phase_mean_T_sl_err_pct()
        self.assertGreaterEqual(n_err, 0.0)
        self.assertLess(n_err, 15.0)
        self.assertGreaterEqual(t_err, 0.0)
        self.assertLess(t_err, 15.0)
        self.assertEqual(pb.chemistry_crystal_contact_panel_pass_rate(), 1.0)
        self.assertEqual(pb.chemistry_crystal_fracture_panel_pass_rate(), 1.0)

    def test_domain_summary_golden_loads(self) -> None:
        summary = pb.chemistry_extent_domain_summary()
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(
            summary["comparison_policy"],
            "External chemistry data grade readouts only; no DFT/table fit inputs.",
        )
        self.assertEqual(summary["molecule_suite"]["total_molecules"], 19)
        self.assertEqual(summary["constraint_system"]["n_equations"], 49)
        self.assertEqual(summary["contact_network_rules"]["network_count"], 21)

    def test_package_summary_builder_matches_golden_shape(self) -> None:
        # Minimal payloads exercise the importable pyhqiv.chemistry_extent helpers
        # without depending on the full paper checkout.
        compact = build_chemistry_extent_domain_summary(
            {
                "chemistry_panel_accuracy": {
                    "spectral": {
                        "count": 1,
                        "reliable_count": 1,
                        "geometric_mean_error_pct": {"r_e": 1.0, "D_e": 2.0, "B_e": 3.0},
                    },
                    "carbon": {
                        "count": 1,
                        "rows": [{"error_pct": -4.0, "bond_error_pct": 0.5}],
                    },
                },
                "molecule_suite_audit": {
                    "summary": {
                        "total_molecules": 1,
                        "core": {"mean_abs_error_pct": 2.0},
                        "open_shell": {"mean_abs_error_pct": 3.0},
                        "combined_core_plus_expanded": {
                            "count": 1,
                            "mean_abs_error_pct": 4.0,
                            "within_15pct": 1,
                        },
                    },
                },
                "chemistry_constraint_system": {
                    "n_equations": 1,
                    "slot_catalog": ["bulk_rho"],
                    "condensed": {"resid_norm": 0.1},
                    "binding": {"resid_norm": 0.2},
                    "spectroscopy": {"rank": 1, "null_dim": 0},
                },
                "chemistry_inverse_channel_solve": {
                    "gmtkn_3slot": {"resid_norm": 0.3},
                    "outside_curvature_gas_participation_w": -0.4,
                    "spectroscopy_x": {"resid_norm": 0.5},
                },
                "nested_wf_geometry": {
                    "summary": {
                        "count": 2,
                        "mean_abs_error_pct": 6.0,
                        "within_15pct": 1,
                        "within_5pct": 1,
                    },
                },
                "quantum_chem_witnesses": {
                    "h2_trace_referenceM": 1200,
                    "h2_trace_referenceM_expected": 1200,
                    "lean_lih_compton_bridge": {
                        "imprint_phase_theorems": {"a": True, "b": False},
                    },
                    "lih_dynamic_binding": {
                        "binding_readouts": [
                            {
                                "mode": "dynamic_compton_participation",
                                "error_pct": -0.7,
                            },
                        ],
                    },
                    "dynamic_binding_chart": {
                        "summary": {
                            "count": 2,
                            "mean_abs_error_pct": 8.0,
                            "within_15pct": 1,
                        },
                    },
                },
                "curvature_contact_network_rules": {
                    "contact_kinds": ["a"],
                    "derived_phases": ["gas"],
                    "networks": [{"rules": ["r"], "contacts": ["c"]}],
                },
                "allotrope_phase_cooling_audit": {
                    "molecules": ["H2O"],
                    "species": {
                        "H2O": {
                            "transitions": ["solid_liquid"],
                            "allotrope_spectroscopy_profiles": ["ice"],
                        },
                    },
                },
                "chemistry_residual_correlation_audit": {
                    "condensed_phase": {
                        "n": 1,
                        "correlations": {
                            "density": {
                                "feature": {"available": True, "pearson_r": -0.25},
                            },
                        },
                        "optical_coupled_relaxation_improvements": [{"molecule": "HF"}],
                    },
                    "spectroscopy": {
                        "n": 2,
                        "n_reliable_geometry": 1,
                        "correlations": {
                            "omega": {
                                "feature": {"available": True, "pearson_r": 0.75},
                            },
                        },
                        "in_bracket_flow_targets": [{"name": "HF"}],
                        "coupled_relaxation_improvements": [{"name": "HF"}],
                    },
                },
                "generator_dependent_coupling_audit": {
                    "rows": [{"name": "H2"}, {"name": "HF"}],
                    "summary": {
                        "abelian": {"mean_abs_error_pct": 3.0},
                        "spectral_gap": {"mean_abs_error_pct": 2.0, "within_5pct": 2},
                    },
                    "recommendation": {"improved_vs_abelian": True},
                },
                "system_matrix_functor_audit": {
                    "rows": [{"name": "H2"}],
                    "summary": {
                        "base": {"mean_abs_error_pct": 2.0},
                        "so8_blend_relative": {"mean_abs_error_pct": 2.1},
                        "contact_relative": {"mean_abs_error_pct": 5.0},
                    },
                    "recommendation": {"best_mean_variant": "base"},
                },
                "second_order_effect_audit": {
                    "rows": [{"name": "H2"}, {"name": "HF"}],
                    "summary": {
                        "base": {"mean_abs_error_pct": 2.0},
                        "outside_geff": {"mean_abs_error_pct": 3.0, "within_5pct": 1},
                    },
                    "recommendation": {"promote_candidate": "outside_geff"},
                },
                "crystal_ethics_audit": {
                    "passes": True,
                    "lean_proof_audit": [{"passes": True}],
                    "a_z_audit": [
                        {"uses_sparse_light_override": False, "override_allowed": True},
                    ],
                    "regime_audit": [{"expected": "solid", "actual": "solid"}],
                },
            }
        )
        self.assertEqual(compact["chemistry_panel_accuracy"]["carbon_density_mean_abs_error_pct"], 4.0)
        self.assertEqual(compact["molecule_suite"]["combined_within_15pct_fraction"], 1.0)
        self.assertEqual(compact["quantum_chem_witnesses"]["lih_imprint_theorem_fraction"], 0.5)
        self.assertEqual(compact["residual_correlation_audit"]["spectroscopy_reliable_fraction"], 0.5)
        self.assertEqual(compact["generator_dependent_coupling"]["spectral_gap_improvement_pct"], 1.0)
        self.assertEqual(compact["crystal_ethics"]["passes_fraction"], 1.0)

    def test_additional_dft_replacement_metric_getters(self) -> None:
        self.assertLess(pb.chemistry_public_spectral_r_e_geom_mean_err_pct(), 2.0)
        self.assertLess(pb.chemistry_public_spectral_D_e_geom_mean_err_pct(), 2.0)
        self.assertLess(pb.chemistry_public_spectral_B_e_geom_mean_err_pct(), 2.0)
        self.assertLess(pb.chemistry_carbon_density_mean_err_pct(), 2.0)
        self.assertLess(pb.chemistry_carbon_bond_mean_err_pct(), 0.5)
        self.assertLess(pb.chemistry_molecule_suite_core_binding_err_pct(), 5.0)
        self.assertLess(pb.chemistry_molecule_suite_combined_binding_err_pct(), 5.0)
        self.assertLess(pb.chemistry_molecule_suite_open_shell_binding_err_pct(), 5.0)
        self.assertEqual(pb.chemistry_molecule_suite_within15_fraction(), 1.0)
        self.assertLess(pb.chemistry_constraint_condensed_resid_norm(), 0.25)
        self.assertLess(pb.chemistry_constraint_binding_resid_norm(), 0.05)
        self.assertLess(pb.chemistry_inverse_gmtkn_resid_norm(), 0.05)
        self.assertLess(pb.chemistry_inverse_outside_gas_participation_abs(), 0.02)
        self.assertGreaterEqual(pb.chemistry_nested_wf_within15_fraction(), 0.8)
        self.assertLess(pb.chemistry_quantum_lih_primary_err_pct(), 1.0)
        self.assertEqual(pb.chemistry_quantum_lih_imprint_theorem_fraction(), 1.0)
        self.assertEqual(pb.chemistry_contact_network_rule_coverage_fraction(), 1.0)
        self.assertEqual(pb.chemistry_allotrope_phase_cooling_coverage_fraction(), 1.0)
        self.assertGreaterEqual(pb.chemistry_residual_spectroscopy_reliable_fraction(), 9 / 11)
        self.assertGreater(pb.chemistry_residual_spectroscopy_max_abs_correlation(), 0.9)
        self.assertGreater(pb.chemistry_residual_condensed_max_abs_correlation(), 0.9)
        self.assertEqual(pb.chemistry_residual_flow_target_count(), 8.0)
        self.assertLess(pb.chemistry_generator_spectral_gap_err_pct(), 2.0)
        self.assertGreater(pb.chemistry_generator_spectral_gap_improvement_pct(), 0.9)
        self.assertEqual(pb.chemistry_generator_recommendation_improved(), 1.0)
        self.assertEqual(pb.chemistry_system_matrix_best_is_base_fraction(), 1.0)
        self.assertLess(pb.chemistry_system_matrix_so8_blend_err_pct(), 3.0)
        self.assertLess(pb.chemistry_second_order_outside_geff_err_pct(), 4.0)
        self.assertEqual(pb.chemistry_second_order_promote_outside_geff_fraction(), 1.0)
        self.assertEqual(pb.chemistry_crystal_ethics_pass_fraction(), 1.0)
        self.assertEqual(pb.chemistry_crystal_ethics_lean_pass_fraction(), 1.0)

    def test_metrics_registered(self) -> None:
        names = {
            "chemistry_spectroscopy_reliable_omega_e_err_pct",
            "chemistry_spectroscopy_reliable_r_e_err_pct",
            "chemistry_spectroscopy_geometry_reliable_fraction",
            "chemistry_spectroscopy_concentration_bracket_hit_rate",
            "chemistry_condensed_phase_mean_n_err_pct",
            "chemistry_condensed_phase_mean_T_sl_err_pct",
            "chemistry_crystal_contact_panel_pass_rate",
            "chemistry_crystal_fracture_panel_pass_rate",
            "chemistry_public_spectral_r_e_geom_mean_err_pct",
            "chemistry_public_spectral_D_e_geom_mean_err_pct",
            "chemistry_public_spectral_B_e_geom_mean_err_pct",
            "chemistry_carbon_density_mean_err_pct",
            "chemistry_carbon_bond_mean_err_pct",
            "chemistry_molecule_suite_core_binding_err_pct",
            "chemistry_molecule_suite_combined_binding_err_pct",
            "chemistry_molecule_suite_open_shell_binding_err_pct",
            "chemistry_molecule_suite_within15_fraction",
            "chemistry_constraint_condensed_resid_norm",
            "chemistry_constraint_binding_resid_norm",
            "chemistry_inverse_gmtkn_resid_norm",
            "chemistry_inverse_outside_gas_participation_abs",
            "chemistry_nested_wf_within15_fraction",
            "chemistry_quantum_lih_primary_err_pct",
            "chemistry_quantum_lih_imprint_theorem_fraction",
            "chemistry_contact_network_rule_coverage_fraction",
            "chemistry_allotrope_phase_cooling_coverage_fraction",
            "chemistry_residual_spectroscopy_reliable_fraction",
            "chemistry_residual_spectroscopy_max_abs_correlation",
            "chemistry_residual_condensed_max_abs_correlation",
            "chemistry_residual_flow_target_count",
            "chemistry_generator_spectral_gap_err_pct",
            "chemistry_generator_spectral_gap_improvement_pct",
            "chemistry_generator_recommendation_improved",
            "chemistry_system_matrix_best_is_base_fraction",
            "chemistry_system_matrix_so8_blend_err_pct",
            "chemistry_second_order_outside_geff_err_pct",
            "chemistry_second_order_promote_outside_geff_fraction",
            "chemistry_crystal_ethics_pass_fraction",
            "chemistry_crystal_ethics_lean_pass_fraction",
        }
        registry = arena_metrics.METRIC_REGISTRY()
        for name in names:
            self.assertIn(name, registry)
            val = float(registry[name].compute())
            self.assertEqual(val, val)


if __name__ == "__main__":
    unittest.main()
