"""
Published HQIV benchmark readouts for Arena scoring.

Loads committed golden snapshots from ``tests/data/`` (CI-safe, deterministic).
When ``HQIV_LEAN`` is on ``PYTHONPATH`` (local dev / extended CI), recomputes live
from the hqiv_lab / scripts audit pipelines so improvements move the score.

Comparison policy (both domains): laboratory / PDB witnesses grade readouts only —
never fold or decay inputs.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _tests_data_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tests" / "data"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[3] / "tests" / "data"


def _hqiv_lean_root() -> Path | None:
    env = os.environ.get("HQIV_LEAN_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    for parent in Path(__file__).resolve().parents:
        sibling = parent / "HQIV_LEAN"
        if (sibling / "lakefile.toml").is_file():
            return sibling
    fallback = Path("/home/jr/Repos/HQIV_LEAN")
    if (fallback / "lakefile.toml").is_file():
        return fallback
    return None


def _load_json(name: str) -> dict[str, Any]:
    path = _tests_data_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"missing arena golden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _live_benchmarks_enabled() -> bool:
    return os.environ.get("PYHQIV_LIVE_BENCHMARKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _with_lean_scripts() -> Path | None:
    if not _live_benchmarks_enabled():
        return None
    root = _hqiv_lean_root()
    if root is None:
        return None
    scripts = root / "scripts"
    repo = root
    for p in (str(repo), str(scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _with_lightcone_chemistry_scripts() -> Path | None:
    """Return the lightcone_chemistry_extent scripts bundle, if available."""
    if not _live_benchmarks_enabled():
        return None
    root = _hqiv_lean_root()
    if root is None:
        return None
    scripts = root / "papers" / "lightcone_chemistry_extent" / "scripts"
    if not scripts.is_dir():
        return None
    for p in (str(root), str(scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return scripts


@lru_cache(maxsize=1)
def miniprotein_fold_audit() -> dict[str, Any]:
    """11-target miniprotein ladder audit (mean Cα RMSD, pass fraction)."""
    root = _with_lean_scripts()
    if root is not None:
        try:
            import hqiv_miniprotein_fold_audit as audit  # type: ignore

            witness_path = root / "data" / "miniprotein_witnesses.json"
            if witness_path.is_file():
                return audit.build_payload(witness_path, include_network=False, closure_engine="nerf")
        except Exception:
            pass
    return _load_json("miniprotein_fold_audit_golden.json")


@lru_cache(maxsize=1)
def hep_decay_benchmark() -> dict[str, Any]:
    """Full HEP decay benchmark payload (567 readouts + curated 17-channel σ panel)."""
    root = _with_lean_scripts()
    if root is not None:
        try:
            import hqiv_hep_decay_benchmark as hep_bench  # type: ignore

            return hep_bench.build_payload(
                observations_path=root / "data" / "hep_decay_observations.json",
                published_path=root / "data" / "hadron_published_masses.json",
            )
        except Exception:
            pass
    return _load_json("hep_decay_benchmark_golden.json")


def miniprotein_mean_ca_rmsd() -> float:
    payload = miniprotein_fold_audit()
    summary = payload.get("summary") or {}
    if "mean_ca_rmsd_angstrom" in summary:
        return float(summary["mean_ca_rmsd_angstrom"])
    folds = payload.get("folds") or payload.get("fold_audit", {}).get("folds") or []
    rmsds = [float(f["ca_rmsd_angstrom"]) for f in folds if f.get("ca_rmsd_angstrom") is not None]
    if not rmsds:
        raise RuntimeError("miniprotein audit: no Cα RMSD rows")
    return sum(rmsds) / len(rmsds)


def miniprotein_trp_cage_ca_rmsd() -> float:
    payload = miniprotein_fold_audit()
    folds = payload.get("folds") or payload.get("fold_audit", {}).get("folds") or []
    for row in folds:
        if row.get("name") == "trp_cage":
            val = row.get("ca_rmsd_angstrom")
            if val is not None:
                return float(val)
    raise RuntimeError("miniprotein audit: trp_cage row missing")


def miniprotein_fold_pass_fraction() -> float:
    payload = miniprotein_fold_audit()
    summary = payload.get("summary") or {}
    targets = int(summary.get("targets") or 0)
    passed = int(summary.get("passed") or 0)
    if targets <= 0:
        folds = payload.get("folds") or payload.get("fold_audit", {}).get("folds") or []
        targets = len(folds)
        passed = sum(1 for f in folds if f.get("passed") is True)
    if targets <= 0:
        raise RuntimeError("miniprotein audit: empty fold panel")
    return passed / targets


def hep_decay_panel_mean_z() -> float:
    payload = hep_decay_benchmark()
    summary = payload.get("summary") or {}
    dist = summary.get("diagnostic_branching_n_sigma_distribution") or {}
    if "mean_n_sigma" in dist:
        return float(dist["mean_n_sigma"])
    if "mean_branching_n_sigma" in summary:
        return float(summary["mean_branching_n_sigma"])
    raise RuntimeError("hep decay benchmark: panel mean n_σ missing")


def hep_decay_panel_max_z() -> float:
    payload = hep_decay_benchmark()
    summary = payload.get("summary") or {}
    dist = summary.get("diagnostic_branching_n_sigma_distribution") or {}
    if "max_n_sigma" in dist:
        return float(dist["max_n_sigma"])
    if "max_branching_n_sigma" in summary:
        return float(summary["max_branching_n_sigma"])
    raise RuntimeError("hep decay benchmark: panel max n_σ missing")


def hep_decay_structural_pass_rate() -> float:
    """Fraction of non-readout benchmark cases passing (81/81 structural suite, zero fails)."""
    payload = hep_decay_benchmark()
    rows = payload.get("rows") or payload.get("branching_comparison_rows")
    if rows and isinstance(rows[0], dict) and "status" in rows[0]:
        structural = [r for r in rows if r.get("status") != "readout"]
        if structural:
            passes = sum(1 for r in structural if r.get("status") == "pass")
            fails = sum(1 for r in structural if r.get("status") == "fail")
            denom = passes + fails
            if denom > 0:
                return passes / denom
    summary = payload.get("summary") or {}
    fails = float(summary.get("fail", 0))
    passes = float(summary.get("pass", 0))
    readout = float((summary.get("by_panel") or {}).get("readout", {}).get("readout", 0))
    structural_total = passes + fails
    if structural_total <= 0 and readout > 0:
        structural_total = float(summary.get("total", 0)) - readout
    if structural_total <= 0:
        raise RuntimeError("hep decay benchmark: empty structural pass summary")
    return passes / structural_total if fails == 0 else passes / (passes + fails)


def sparc_median_chi2_residual_ratio() -> float:
    """
    SPARC rotation-curve residual proxy: median χ²_red(HQIV) / median χ²_red(baryonic).
    Lower is better (HQIV explains more of the curve than baryons-only).
    """
    data = _load_json("sparc_hqiv_catalog.json")
    summary = data.get("summary") or {}
    hqiv = float(summary["median_chi2_red_hqiv"])
    bar = float(summary["median_chi2_red_baryonic"])
    if bar <= 0:
        raise RuntimeError("sparc catalog: invalid baryonic chi2")
    return hqiv / bar


@lru_cache(maxsize=1)
def phase_diagram_audit() -> dict[str, Any]:
    """Generalized (T,P) phase diagram audit — H₂O LLPT branch + mixture end members."""
    root = _with_lean_scripts()
    if root is not None:
        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import hqiv_thermodynamic_phase_from_tp as tptp  # type: ignore
            from hqiv_lab.phase_diagram import (  # type: ignore
                WATER_HOH_ANGLE_OBSERVATIONS,
                WATER_LLPT_OBSERVATIONS,
                end_members_for_molecule,
                hoh_angle_witness_row,
                low_density_free_energy_minimum,
                low_density_liquid_fraction,
                material_scales_for_spec,
                phase_diagram_point,
                widom_proxy_peak_at_pressure,
                widom_second_order_window_center_k,
                widom_second_order_window_weight,
            )
            from hqiv_lab.protein_solvent_phase import (  # type: ignore
                PROTEIN_FOLDING_TEMPERATURE_K,
                aqueous_mixture_curvature_at_interface,
                bulk_low_density_fraction,
                local_low_density_fraction_at_interface,
            )
            from hqiv_lab.spec import resolve_spec  # type: ignore

            low, high = end_members_for_molecule("H2O")
            mat = material_scales_for_spec(resolve_spec("H2O"), bulk=True)
            widom_peak = widom_proxy_peak_at_pressure(mat, tptp.STP_PRESSURE_PA)
            t_melt_h2o, _ = tptp.characteristic_temperatures_K(mat)
            gamma2_center = widom_second_order_window_center_k(t_melt_h2o)
            f_at_window = low_density_liquid_fraction(
                gamma2_center, tptp.STP_PRESSURE_PA, mat
            )
            kim_peak = next(
                (
                    o
                    for o in WATER_LLPT_OBSERVATIONS
                    if str(o.get("label", "")).startswith("compressibility maximum")
                ),
                None,
            )
            f_bulk_super = bulk_low_density_fraction(200.0)
            f_hydro = local_low_density_fraction_at_interface(f_bulk_super, "hydrophobic")
            f_nuc_base = low_density_liquid_fraction(220.0, tptp.STP_PRESSURE_PA, mat)
            f_nuc_defect = low_density_liquid_fraction(
                220.0,
                tptp.STP_PRESSURE_PA,
                mat,
                local_coordination_excess=0.25,
            )
            rho_hydro = aqueous_mixture_curvature_at_interface(200.0, "hydrophobic")
            rho_bulk_fold = aqueous_mixture_curvature_at_interface(
                PROTEIN_FOLDING_TEMPERATURE_K, "neutral"
            )
            anchors = {
                "cytosol_310K_1atm": phase_diagram_point("H2O", temperature_k=310.15),
                "melt_273K_1atm": phase_diagram_point("H2O", temperature_k=273.15),
                "llcp_198K_1250atm": phase_diagram_point(
                    "H2O", temperature_k=198.0, pressure_pa=1250.0 * tptp.STP_PRESSURE_PA
                ),
                "ice_271K_1atm": phase_diagram_point("H2O", temperature_k=271.0),
            }

            def _anchor_row(label: str, pt: Any) -> dict[str, Any]:
                return {
                    "label": label,
                    "temperature_K": pt.temperature_k,
                    "pressure_Pa": pt.pressure_pa,
                    "pressure_atm": pt.pressure_pa / tptp.STP_PRESSURE_PA,
                    "derived_phase": pt.derived_phase,
                    "liquid_subphase": pt.liquid_subphase.value if pt.liquid_subphase else None,
                    "f_low_density": pt.f_low_density,
                    "rho_curv": pt.rho_curv,
                    "T_melt_K": pt.T_melt_K,
                    "notes": pt.notes,
                }

            return {
                "source": "hqiv_lab/phase_diagram.py",
                "derivation": "HQIV motif + cohesive ladder (no MD/DFT inputs)",
                "comparison_policy": "external observations grade readouts only",
                "water_llpt_observations": list(WATER_LLPT_OBSERVATIONS),
                "water_hoh_angle_observations": list(WATER_HOH_ANGLE_OBSERVATIONS),
                "end_members": {
                    "low_density": {"label": low.label, "rho_curv": low.rho_curv},
                    "high_density": {"label": high.label, "rho_curv": high.rho_curv},
                },
                "anchor_points": {
                    k: _anchor_row(k, v) for k, v in anchors.items()
                },
                "structural_expectations": {
                    "cytosol_phase": "liquid",
                    "llcp_regime_phase": "metastable_liquid",
                    "ice_below_melt_phase": "solid",
                    "T_melt_K_window": [270.0, 276.0],
                },
                "widom_compressibility_proxy": widom_peak,
                "widom_free_energy": {
                    "minimum_at_gamma2_window": low_density_free_energy_minimum(
                        gamma2_center, tptp.STP_PRESSURE_PA, mat
                    ),
                    "gamma2_window": {
                        "T_melt_K": t_melt_h2o,
                        "center_K": gamma2_center,
                        "peak_minus_center_K": float(widom_peak["temperature_K"]) - gamma2_center,
                        "weight_at_center": widom_second_order_window_weight(gamma2_center, t_melt_h2o),
                        "weight_at_150K": widom_second_order_window_weight(150.0, t_melt_h2o),
                    },
                },
                "kim_compressibility_peak_T_K": kim_peak["T_K"] if kim_peak else None,
                "hoh_angle_witness": {
                    "window_center_1atm": hoh_angle_witness_row(f_at_window),
                    "cytosol_310K_1atm": hoh_angle_witness_row(
                        low_density_liquid_fraction(310.15, tptp.STP_PRESSURE_PA, mat)
                    ),
                },
                "nucleation_defect_witness": {
                    "temperature_K": 220.0,
                    "pressure_atm": 1.0,
                    "local_coordination_excess": 0.25,
                    "f_low_density_baseline": f_nuc_base,
                    "f_low_density_defect": f_nuc_defect,
                    "f_low_density_excess": f_nuc_defect - f_nuc_base,
                },
                "protein_interface": {
                    "supercooled_200K_f_bulk": f_bulk_super,
                    "supercooled_200K_f_hydrophobic": f_hydro,
                    "supercooled_200K_rho_hydrophobic": rho_hydro,
                    "fold_310K_rho_neutral": rho_bulk_fold,
                },
            }
        except Exception:
            pass
    return _load_json("phase_diagram_audit_golden.json")


def water_h2o_melt_temperature_K() -> float:
    audit = phase_diagram_audit()
    return float(audit["anchor_points"]["cytosol_310K_1atm"]["T_melt_K"])


def water_phase_diagram_structural_pass_rate() -> float:
    audit = phase_diagram_audit()
    exp = audit["structural_expectations"]
    anchors = audit["anchor_points"]
    checks = [
        anchors["cytosol_310K_1atm"]["derived_phase"] == exp["cytosol_phase"],
        anchors["llcp_198K_1250atm"]["derived_phase"] == exp["llcp_regime_phase"],
        anchors["ice_271K_1atm"]["derived_phase"] == exp["ice_below_melt_phase"],
    ]
    t_melt = float(anchors["cytosol_310K_1atm"]["T_melt_K"])
    lo, hi = exp["T_melt_K_window"]
    checks.append(lo <= t_melt <= hi)
    return sum(checks) / len(checks)


def water_metastable_liquid_at_llcp() -> float:
    audit = phase_diagram_audit()
    phase = audit["anchor_points"]["llcp_198K_1250atm"]["derived_phase"]
    return 1.0 if phase == "metastable_liquid" else 0.0


def water_llcp_observation_distance() -> float:
    """
    Normalized (T,P) distance from Sciortino LLCP observation to nearest HQIV grid anchor.

    Comparison quarantine only — lower means HQIV places the LLPT regime nearer the
    literature critical point without using MB-pol coordinates as inputs.
    """
    audit = phase_diagram_audit()
    obs = next(
        (o for o in audit.get("water_llpt_observations", []) if o.get("T_K") is not None),
        None,
    )
    if obs is None:
        raise RuntimeError("phase diagram audit: LLCP observation row missing")
    anchor = audit["anchor_points"]["llcp_198K_1250atm"]
    t_err = 20.0  # K scale (supercooled water literature spread)
    p_err = 200.0  # atm scale
    dt = abs(float(anchor["temperature_K"]) - float(obs["T_K"])) / t_err
    dp = abs(float(anchor["pressure_atm"]) - float(obs["P_atm"])) / p_err
    return (dt * dt + dp * dp) ** 0.5


def water_widom_peak_temperature_residual_K() -> float:
    """|T_peak(κ proxy) − Kim et al. compressibility maximum ~229 K| (comparison only)."""
    audit = phase_diagram_audit()
    kim_t = audit.get("kim_compressibility_peak_T_K")
    peak = audit.get("widom_compressibility_proxy")
    if kim_t is None or peak is None:
        raise RuntimeError("phase diagram audit: widom/Kim comparison rows missing")
    return abs(float(peak["temperature_K"]) - float(kim_t))


def water_widom_gamma2_window_alignment_K() -> float:
    """|T_peak(κ proxy) − T_melt·(1−γ²)|, an HQIV-internal structural residual."""
    audit = phase_diagram_audit()
    peak = audit.get("widom_compressibility_proxy")
    window = audit.get("widom_free_energy", {}).get("gamma2_window")
    if peak is None or window is None:
        raise RuntimeError("phase diagram audit: gamma2 Widom window missing")
    return abs(float(peak["temperature_K"]) - float(window["center_K"]))


def water_nucleation_defect_ldl_excess() -> float:
    """Local δB / coordination defect raises f_LDL at a fixed supercooled state."""
    audit = phase_diagram_audit()
    witness = audit.get("nucleation_defect_witness")
    if witness is None:
        raise RuntimeError("phase diagram audit: nucleation_defect_witness missing")
    return float(witness["f_low_density_excess"])


def water_hoh_angle_taxonomy_open_gap_deg() -> float:
    """Current θ_dyn vs gas-phase comparison residual after torque-tree screening."""
    audit = phase_diagram_audit()
    witness = audit.get("hoh_angle_witness", {}).get("cytosol_310K_1atm")
    if witness is None:
        raise RuntimeError("phase diagram audit: hoh_angle_witness missing")
    return abs(float(witness["theta_dyn_minus_ref_deg"]))


def water_h2o_bond_angle_residual_deg() -> float:
    """Alias for the H–O–H dynamic-centre angle residual vs gas-phase comparison row."""
    return water_hoh_angle_taxonomy_open_gap_deg()


def protein_hydrophobic_interface_ldl_excess() -> float:
    """
    Local f_LDL excess at hydrophobic interface vs bulk at supercooled 200 K, 1 atm.

    Positive witness that interface dress biases toward LDL without fitted potentials.
    """
    audit = phase_diagram_audit()
    iface = audit.get("protein_interface")
    if iface is None:
        raise RuntimeError("phase diagram audit: protein_interface block missing")
    return float(iface["supercooled_200K_f_hydrophobic"]) - float(
        iface["supercooled_200K_f_bulk"]
    )


# --- Light-cone chemistry extent (spectroscopy / crystals / condensed phase) ---


@lru_cache(maxsize=1)
def molecular_spectroscopy_audit() -> dict[str, Any]:
    """Diatomic rovibrational witnesses (NIST/CRC/HITRAN comparison quarantine)."""
    root = _with_lightcone_chemistry_scripts()
    if root is not None:
        try:
            import hqiv_molecular_spectroscopy as ms  # type: ignore

            return ms.build_payload()
        except Exception:
            pass
    return _load_json("molecular_spectroscopy_witnesses_golden.json")


@lru_cache(maxsize=1)
def crystal_contact_audit() -> dict[str, Any]:
    """Ionic / metallic / covalent-network crystal contact panel."""
    root = _with_lightcone_chemistry_scripts()
    if root is not None:
        try:
            import hqiv_crystal_contact_geometry as ccg  # type: ignore

            if hasattr(ccg, "build_payload"):
                return ccg.build_payload()
        except Exception:
            pass
    return _load_json("crystal_contact_witnesses_golden.json")


@lru_cache(maxsize=1)
def crystal_fracture_audit() -> dict[str, Any]:
    """Contact-derived Griffith-scale / cleavage / ductile-carrier witnesses."""
    root = _with_lightcone_chemistry_scripts()
    if root is not None:
        try:
            import hqiv_crystal_fracture_witness as cfw  # type: ignore

            if hasattr(cfw, "build_payload"):
                return cfw.build_payload()
        except Exception:
            pass
    return _load_json("crystal_fracture_witnesses_golden.json")


@lru_cache(maxsize=1)
def condensed_phase_audit() -> dict[str, Any]:
    """Condensed-phase density / n / T_sl comparison audit."""
    root = _with_lightcone_chemistry_scripts()
    if root is not None:
        try:
            import hqiv_condensed_phase_audit as cpa  # type: ignore

            if hasattr(cpa, "build_payload"):
                return cpa.build_payload()
        except Exception:
            pass
    return _load_json("condensed_phase_audit_golden.json")


@lru_cache(maxsize=1)
def chemistry_extent_domain_summary() -> dict[str, Any]:
    """Compact DFT-replacement domain summary from lightcone_chemistry_extent."""
    scripts = _with_lightcone_chemistry_scripts()
    if scripts is not None:
        try:
            from pyhqiv.chemistry_extent import build_chemistry_extent_domain_summary

            data = scripts / "data"
            payloads = {
                "chemistry_panel_accuracy": json.loads(
                    (data / "chemistry_panel_accuracy.json").read_text(encoding="utf-8")
                ),
                "molecule_suite_audit": json.loads(
                    (data / "molecule_suite_audit.json").read_text(encoding="utf-8")
                ),
                "chemistry_constraint_system": json.loads(
                    (data / "chemistry_constraint_system.json").read_text(encoding="utf-8")
                ),
                "chemistry_inverse_channel_solve": json.loads(
                    (data / "chemistry_inverse_channel_solve.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "nested_wf_geometry": json.loads(
                    (data / "nested_wf_geometry.json").read_text(encoding="utf-8")
                ),
                "quantum_chem_witnesses": json.loads(
                    (data / "quantum_chem_witnesses.json").read_text(encoding="utf-8")
                ),
                "curvature_contact_network_rules": json.loads(
                    (data / "curvature_contact_network_rules.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "allotrope_phase_cooling_audit": json.loads(
                    (data / "allotrope_phase_cooling_audit.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "chemistry_residual_correlation_audit": json.loads(
                    (data / "chemistry_residual_correlation_audit.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "generator_dependent_coupling_audit": json.loads(
                    (data / "generator_dependent_coupling_audit.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "system_matrix_functor_audit": json.loads(
                    (data / "system_matrix_functor_audit.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "second_order_effect_audit": json.loads(
                    (data / "second_order_effect_audit.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "crystal_ethics_audit": json.loads(
                    (data / "crystal_ethics_audit.json").read_text(encoding="utf-8")
                ),
            }
            return build_chemistry_extent_domain_summary(payloads)
        except Exception:
            pass
    return _load_json("chemistry_extent_domain_summary_golden.json")


def _domain_summary_value(section: str, key: str) -> float:
    summary = chemistry_extent_domain_summary()
    return float(summary[section][key])


def chemistry_public_spectral_r_e_geom_mean_err_pct() -> float:
    return float(
        chemistry_extent_domain_summary()["chemistry_panel_accuracy"][
            "spectral_geometric_mean_error_pct"
        ]["r_e"]
    )


def chemistry_public_spectral_D_e_geom_mean_err_pct() -> float:
    return float(
        chemistry_extent_domain_summary()["chemistry_panel_accuracy"][
            "spectral_geometric_mean_error_pct"
        ]["D_e"]
    )


def chemistry_public_spectral_B_e_geom_mean_err_pct() -> float:
    return float(
        chemistry_extent_domain_summary()["chemistry_panel_accuracy"][
            "spectral_geometric_mean_error_pct"
        ]["B_e"]
    )


def chemistry_carbon_density_mean_err_pct() -> float:
    return _domain_summary_value(
        "chemistry_panel_accuracy", "carbon_density_mean_abs_error_pct"
    )


def chemistry_carbon_bond_mean_err_pct() -> float:
    return _domain_summary_value("chemistry_panel_accuracy", "carbon_bond_mean_abs_error_pct")


def chemistry_molecule_suite_core_binding_err_pct() -> float:
    return _domain_summary_value("molecule_suite", "core_mean_abs_binding_error_pct")


def chemistry_molecule_suite_combined_binding_err_pct() -> float:
    return _domain_summary_value("molecule_suite", "combined_mean_abs_binding_error_pct")


def chemistry_molecule_suite_open_shell_binding_err_pct() -> float:
    return _domain_summary_value("molecule_suite", "open_shell_mean_abs_binding_error_pct")


def chemistry_molecule_suite_within15_fraction() -> float:
    return _domain_summary_value("molecule_suite", "combined_within_15pct_fraction")


def chemistry_constraint_condensed_resid_norm() -> float:
    return _domain_summary_value("constraint_system", "condensed_resid_norm")


def chemistry_constraint_binding_resid_norm() -> float:
    return _domain_summary_value("constraint_system", "binding_resid_norm")


def chemistry_inverse_gmtkn_resid_norm() -> float:
    return _domain_summary_value("inverse_channel_solve", "gmtkn_resid_norm")


def chemistry_inverse_outside_gas_participation_abs() -> float:
    return _domain_summary_value(
        "inverse_channel_solve", "outside_curvature_gas_abs_participation"
    )


def chemistry_nested_wf_within15_fraction() -> float:
    return _domain_summary_value("nested_wf_geometry", "within_15pct_fraction")


def chemistry_quantum_lih_primary_err_pct() -> float:
    return _domain_summary_value("quantum_chem_witnesses", "lih_dynamic_primary_error_pct")


def chemistry_quantum_lih_imprint_theorem_fraction() -> float:
    return _domain_summary_value("quantum_chem_witnesses", "lih_imprint_theorem_fraction")


def chemistry_contact_network_rule_coverage_fraction() -> float:
    summary = chemistry_extent_domain_summary()["contact_network_rules"]
    return (
        float(summary["network_with_rules_fraction"])
        + float(summary["network_with_contacts_fraction"])
    ) / 2.0


def chemistry_allotrope_phase_cooling_coverage_fraction() -> float:
    summary = chemistry_extent_domain_summary()["allotrope_phase_cooling"]
    return (
        float(summary["transition_coverage_fraction"])
        + float(summary["profile_coverage_fraction"])
    ) / 2.0


def chemistry_residual_spectroscopy_reliable_fraction() -> float:
    return _domain_summary_value(
        "residual_correlation_audit", "spectroscopy_reliable_fraction"
    )


def chemistry_residual_spectroscopy_max_abs_correlation() -> float:
    return _domain_summary_value(
        "residual_correlation_audit", "spectroscopy_max_abs_correlation"
    )


def chemistry_residual_condensed_max_abs_correlation() -> float:
    return _domain_summary_value(
        "residual_correlation_audit", "condensed_max_abs_correlation"
    )


def chemistry_residual_flow_target_count() -> float:
    return _domain_summary_value(
        "residual_correlation_audit", "in_bracket_flow_target_count"
    )


def chemistry_generator_spectral_gap_err_pct() -> float:
    return _domain_summary_value(
        "generator_dependent_coupling", "spectral_gap_mean_abs_error_pct"
    )


def chemistry_generator_spectral_gap_improvement_pct() -> float:
    return _domain_summary_value(
        "generator_dependent_coupling", "spectral_gap_improvement_pct"
    )


def chemistry_generator_recommendation_improved() -> float:
    return _domain_summary_value("generator_dependent_coupling", "recommendation_improved")


def chemistry_system_matrix_best_is_base_fraction() -> float:
    return _domain_summary_value("system_matrix_functor", "best_is_base_fraction")


def chemistry_system_matrix_so8_blend_err_pct() -> float:
    return _domain_summary_value("system_matrix_functor", "so8_blend_mean_abs_error_pct")


def chemistry_second_order_outside_geff_err_pct() -> float:
    return _domain_summary_value("second_order_effect", "outside_geff_mean_abs_error_pct")


def chemistry_second_order_promote_outside_geff_fraction() -> float:
    return _domain_summary_value("second_order_effect", "promote_outside_geff_fraction")


def chemistry_crystal_ethics_pass_fraction() -> float:
    return _domain_summary_value("crystal_ethics", "passes_fraction")


def chemistry_crystal_ethics_lean_pass_fraction() -> float:
    return _domain_summary_value("crystal_ethics", "lean_module_pass_fraction")


def chemistry_spectroscopy_reliable_omega_e_err_pct() -> float:
    """Mean |Δω_e|% over geometry-reliable diatomic rows (comparison quarantine)."""
    payload = molecular_spectroscopy_audit()
    summary = payload.get("summary") or {}
    reliable = summary.get("mean_abs_error_pct_reliable") or {}
    if "omega_e" not in reliable:
        raise RuntimeError("spectroscopy audit: reliable ω_e mean missing")
    return float(reliable["omega_e"])


def chemistry_spectroscopy_reliable_r_e_err_pct() -> float:
    """Mean |Δr_e|% over geometry-reliable diatomic rows."""
    payload = molecular_spectroscopy_audit()
    summary = payload.get("summary") or {}
    reliable = summary.get("mean_abs_error_pct_reliable") or {}
    if "r_e" not in reliable:
        raise RuntimeError("spectroscopy audit: reliable r_e mean missing")
    return float(reliable["r_e"])


def chemistry_spectroscopy_geometry_reliable_fraction() -> float:
    """Fraction of spectroscopy panel rows with geometry_reliable=True."""
    payload = molecular_spectroscopy_audit()
    summary = payload.get("summary") or {}
    n = int(summary.get("count") or 0)
    n_rel = int(summary.get("count_reliable_geometry") or 0)
    if n <= 0:
        rows = payload.get("rows") or []
        n = len(rows)
        n_rel = sum(1 for r in rows if r.get("geometry_reliable"))
    if n <= 0:
        raise RuntimeError("spectroscopy audit: empty panel")
    return n_rel / n


def chemistry_spectroscopy_concentration_bracket_hit_rate() -> float:
    """Fraction of concentration brackets that contain the NIST ω_e."""
    payload = molecular_spectroscopy_audit()
    brk = (payload.get("summary") or {}).get("omega_e_concentration_bracket") or {}
    n = int(brk.get("count_with_bracket") or 0)
    hits = int(brk.get("count_nist_within_bracket") or 0)
    if n <= 0:
        raise RuntimeError("spectroscopy audit: concentration bracket summary missing")
    return hits / n


def chemistry_condensed_phase_mean_n_err_pct() -> float:
    """Mean |Δn|% vs NIST/CRC on condensed-phase panel."""
    audit = condensed_phase_audit()
    summary = audit.get("summary") or {}
    if "mean_refractive_index_error_pct_vs_nist" not in summary:
        raise RuntimeError("condensed-phase audit: mean n error missing")
    return float(summary["mean_refractive_index_error_pct_vs_nist"])


def chemistry_condensed_phase_mean_T_sl_err_pct() -> float:
    """Mean |ΔT_sl|% vs NIST melt on condensed-phase panel."""
    audit = condensed_phase_audit()
    summary = audit.get("summary") or {}
    if "mean_T_sl_error_pct_vs_nist" not in summary:
        raise RuntimeError("condensed-phase audit: mean T_sl error missing")
    return float(summary["mean_T_sl_error_pct_vs_nist"])


def chemistry_crystal_contact_panel_pass_rate() -> float:
    """Structural completeness of crystal-contact witness panel (NaCl, Cu, Si, Ge)."""
    audit = crystal_contact_audit()
    witnesses = audit.get("witnesses") or []
    required = {"NaCl", "Cu", "Si", "Ge"}
    present = {w.get("name") for w in witnesses}
    if not required:
        raise RuntimeError("crystal contact audit: empty requirements")
    checks = [
        required.issubset(present),
        all(
            (w.get("nearest_neighbor_angstrom") or 0) > 0
            for w in witnesses
            if w.get("name") in required
        ),
        all(w.get("comparison_regime") == "solid_lattice" for w in witnesses if w.get("name") in required),
    ]
    return sum(1 for c in checks if c) / len(checks)


def chemistry_crystal_fracture_panel_pass_rate() -> float:
    """Structural completeness of fracture-scale witness panel (no handbook inputs)."""
    audit = crystal_fracture_audit()
    ethics = audit.get("ethics") or {}
    witnesses = audit.get("witnesses") or []
    checks = [
        bool(ethics.get("no_tabulated_moduli")),
        bool(ethics.get("no_tabulated_fracture_toughness")),
        bool(ethics.get("not_a_handbook_prediction")),
        len(witnesses) >= 3,
        all((w.get("K_scale_candidate_Pa_sqrt_m") or 0) > 0 for w in witnesses),
    ]
    return sum(1 for c in checks if c) / len(checks)
