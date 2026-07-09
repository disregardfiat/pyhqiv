#!/usr/bin/env python3
"""
Export phase-diagram showcase rows for disregardfiat.tech #arena water tab.

Reads live audit from HQIV_LEAN when available, else pyhqiv golden JSON.

  PYTHONPATH=src HQIV_LEAN_ROOT=/path/to/HQIV_LEAN python scripts/export_phase_diagram_showcase.py
  python scripts/export_phase_diagram_showcase.py --out ../disregardfiat-dot-tech/public/arena/showcase_extras.json --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _phase_diagram_water_rows(audit: dict) -> list[dict]:
    anchors = audit["anchor_points"]
    end = audit["end_members"]
    obs = audit.get("water_llpt_observations") or []
    sciortino = next((o for o in obs if o.get("T_K") is not None and "LLCP" in str(o.get("label", ""))), {})
    hoh = audit.get("hoh_angle_witness", {}).get("cytosol_310K_1atm", {})
    hoh_obs = audit.get("water_hoh_angle_observations") or []
    primary_obs = next((o for o in hoh_obs if o.get("primary_comparison")), {})
    iface = audit.get("protein_interface") or {}
    widom = audit.get("widom_free_energy") or {}
    widom_peak = audit.get("widom_compressibility_proxy") or {}
    kim_peak_t = audit.get("kim_compressibility_peak_T_K")
    rows = [
        {
            "name": "h2o_ldl_rho_curv",
            "label": "LDL end member ρ_curv (tetrahedral melt ratio)",
            "value": end["low_density"]["rho_curv"],
            "reference": end["low_density"]["rho_curv"],
            "unit": "dimless",
            "desc": "Low-density liquid branch from PhaseDiagramMixture — coordination-heavy tetrahedral melt ladder.",
        },
        {
            "name": "h2o_hdl_rho_curv",
            "label": "HDL end member ρ_curv (melt comparison)",
            "value": end["high_density"]["rho_curv"],
            "reference": 1.0,
            "unit": "dimless",
            "desc": "High-density liquid branch — periodic lattice released at melt comparison (ρ = 1).",
        },
        {
            "name": "h2o_T_sl_K",
            "label": "Bulk H₂O T_sl (geometry ladder)",
            "value": anchors["cytosol_310K_1atm"]["T_melt_K"],
            "reference": 273.15,
            "unit": "K",
            "desc": "Solid→liquid scale from ice Ih unit cell + cohesive melt (Arena: water_h2o_melt_T_residual_K).",
        },
        {
            "name": "h2o_cytosol_phase",
            "label": "Phase @ 310 K, 1 atm (cytosol)",
            "value": anchors["cytosol_310K_1atm"]["derived_phase"],
            "reference": "liquid",
            "desc": "Protein-folding box bulk aqueous readout — HDL liquid branch.",
        },
        {
            "name": "h2o_llcp_regime_phase",
            "label": "Phase @ Sciortino LLCP (~198 K, 1250 atm)",
            "value": anchors["llcp_198K_1250atm"]["derived_phase"],
            "reference": "metastable_liquid",
            "desc": (
                f"Comparison anchor for Sciortino et al. ({sciortino.get('T_K', 198)} K, "
                f"{sciortino.get('P_atm', 1250)} atm) — MB-pol coordinates never enter HQIV derivation."
            ),
        },
        {
            "name": "h2o_llcp_rho_curv",
            "label": "ρ_curv @ LLCP-regime anchor",
            "value": anchors["llcp_198K_1250atm"]["rho_curv"],
            "reference": 1.0,
            "unit": "dimless",
            "desc": "Mixture curvature fraction at supercooled high-pressure metastable liquid point.",
        },
        {
            "name": "h2o_theta_tetrahedral_deg",
            "label": "H–O–H tetrahedral network angle θ_tet (LDL end member)",
            "value": hoh.get("theta_tetrahedral_deg"),
            "reference": hoh.get("theta_tetrahedral_deg"),
            "unit": "deg",
            "desc": "VSEPR balance cos θ = −1/3 — ice/LDL network reference, not gas-phase H₂O.",
        },
        {
            "name": "h2o_theta_dynamic_gas_deg",
            "label": "H–O–H dynamic gas angle θ_dyn (HDL slot)",
            "value": hoh.get("theta_dynamic_gas_deg"),
            "reference": primary_obs.get("theta_deg", hoh.get("theta_gas_reference_deg")),
            "unit": "deg",
            "desc": (
                "Torque-tree screened lone-pair dress on O centre (dynamicCentreAngleRad 8 2). "
                f"Comparison median {primary_obs.get('theta_deg', 104.478)}° "
                f"({primary_obs.get('source', 'NIST CCCBDB / Hoy & Bunker 1979')}) — not an input."
            ),
        },
        {
            "name": "h2o_theta_mix_cytosol_deg",
            "label": "Local H–O–H mixture angle @ cytosol f_LDL",
            "value": hoh.get("theta_mixture_deg"),
            "reference": hoh.get("theta_gas_reference_deg"),
            "unit": "deg",
            "desc": "f·θ_tet + (1−f)·θ_dyn at 310 K bulk mixture fraction — protein-folding aqueous box.",
        },
        {
            "name": "h2o_hoh_angle_residual_deg",
            "label": "|θ_dyn − gas comparison|",
            "value": abs(float(hoh.get("theta_dyn_minus_ref_deg", 0.0))),
            "reference": 0.0,
            "unit": "deg",
            "desc": "Arena water_h2o_bond_angle_residual_deg — inside ±0.01° band vs NIST 104.478°.",
        },
        {
            "name": "h2o_widom_peak_T_residual_K",
            "label": "Widom proxy peak − Kim 2017 compressibility max",
            "value": abs(float(widom_peak.get("temperature_K", 0.0)) - float(kim_peak_t))
            if kim_peak_t is not None and widom_peak.get("temperature_K") is not None
            else None,
            "reference": 0.0,
            "unit": "K",
            "desc": "Free-energy susceptibility proxy peak vs Kim et al. Science 2017 (~229 K at 1 atm).",
        },
        {
            "name": "protein_hydrophobic_f_ldl_excess",
            "label": "f_LDL excess @ hydrophobic interface (200 K witness)",
            "value": float(iface.get("supercooled_200K_f_hydrophobic", 0.0))
            - float(iface.get("supercooled_200K_f_bulk", 0.0)),
            "reference": 0.05,
            "unit": "fraction",
            "desc": "ProteinSolventPhaseGeometry interface dress — hydrophobic exposure biases LDL locally.",
        },
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO.parent / "disregardfiat-dot-tech" / "public" / "arena" / "showcase_extras.json",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge phase-diagram rows into existing showcase_extras.json",
    )
    args = parser.parse_args()

    from pyhqiv.arena.published_benchmarks import phase_diagram_audit

    audit = phase_diagram_audit()
    phase_rows = _phase_diagram_water_rows(audit)

    doc: dict
    if args.merge and args.out.is_file():
        doc = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        doc = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "Arena showcase extras",
            "electroweak": [],
            "water": [],
            "hep_decay_channels": [],
        }

    existing = {w["name"] for w in doc.get("water", [])}
    merged_water = [w for w in doc.get("water", []) if not w["name"].startswith("h2o_") and w["name"] != "phase_diagram"]
    merged_water.extend(phase_rows)
    doc["water"] = merged_water
    doc["phase_diagram"] = {
        "source": audit.get("source"),
        "derivation": audit.get("derivation"),
        "comparison_policy": audit.get("comparison_policy"),
        "water_llpt_observations": audit.get("water_llpt_observations"),
        "water_hoh_angle_observations": audit.get("water_hoh_angle_observations"),
        "hoh_angle_witness": audit.get("hoh_angle_witness"),
        "protein_interface": audit.get("protein_interface"),
        "widom_free_energy": audit.get("widom_free_energy"),
        "kim_compressibility_peak_T_K": audit.get("kim_compressibility_peak_T_K"),
        "anchor_points": audit.get("anchor_points"),
        "end_members": audit.get("end_members"),
    }
    doc["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(phase_rows)} phase-diagram water rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
