#!/usr/bin/env python3
"""
Export arena/programme_sigma.json — Wikipedia open-problem map + Arena σ snapshot.

Used by disregardfiat.tech (#mysteries) and documentation. Regenerate after
metric or problem-mapping changes:

  PYTHONPATH=src python scripts/export_programme_sigma.py
  PYTHONPATH=src python scripts/export_programme_sigma.py --arena /path/to/arena_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

WIKIPEDIA_LIST_URL = "https://en.wikipedia.org/wiki/List_of_unsolved_problems_in_physics"

# Curated map: id must be stable for deep links on the website.
PROBLEMS: list[dict] = [
    {
        "id": "theory-of-everything",
        "wikipedia_topic": "General physics",
        "title": "Theory of everything",
        "status": "partial",
        "hqiv": "HQIV is a patch programme: discrete null-lattice combinatorics + horizon thermodynamics + certified so(8) closure aim to link gravity, gauge structure, and mass readouts on one spine — not yet a single accepted TOE Lagrangian.",
        "papers": ["unified-framework", "octonionic-action"],
        "arena_metrics": ["so8_dim", "curvature_norm_combinatorial"],
    },
    {
        "id": "quantum-gravity",
        "wikipedia_topic": "Quantum gravity",
        "title": "Quantum gravity (GR + QM consistency)",
        "status": "partial",
        "hqiv": "Spacetime is modelled as a locally finite causal order with quadratic null-shell growth; gravity enters via modified inertia and HQVM lapse rather than a finished graviton QFT. Discrete carrier + Lean witnesses are the consistency gate.",
        "papers": ["unified-framework", "3d-causal-growth"],
        "arena_metrics": ["omega_k_at_horizon_self", "lapse_factor_ref_point"],
    },
    {
        "id": "black-hole-information",
        "wikipedia_topic": "Quantum gravity",
        "title": "Black hole information paradox",
        "status": "addressed",
        "hqiv": "Soft dynamical firewall: information is released through horizon bookkeeping on the auxiliary channel instead of being destroyed at a sharp Planck wall.",
        "papers": ["unified-framework"],
        "arena_metrics": [],
    },
    {
        "id": "problem-of-time",
        "wikipedia_topic": "Quantum gravity",
        "title": "Problem of time",
        "status": "partial",
        "hqiv": "A shell-indexed rapidity dial and ADM lapse compression separate wall-clock evolution from apparent cosmological time; full reconciliation with every GR clock construction is still being spelled out in the variational layer.",
        "papers": ["unified-framework", "octonionic-action"],
        "arena_metrics": ["lapse_factor_ref_point"],
    },
    {
        "id": "interpretation-of-qm",
        "wikipedia_topic": "Foundations of physics",
        "title": "Interpretation of quantum mechanics",
        "status": "reinterpreted",
        "hqiv": "Born-rule statistics from a finite measurement layer on the auxiliary field; deterministic local accounting — pilot-wave aligned, explicitly not many-worlds branching.",
        "papers": ["auxiliary-fields"],
        "arena_metrics": [],
    },
    {
        "id": "arrow-of-time",
        "wikipedia_topic": "Foundations of physics",
        "title": "Arrow of time",
        "status": "addressed",
        "hqiv": "Macroscopic direction from causal order (no closed loops), nonnegative entropy production on finite patches, cooler outer shells, and a certified Lyapunov relaxation of under-filled early shells.",
        "papers": ["thermodynamics-arrow"],
        "arena_metrics": [],
    },
    {
        "id": "yang-mills-mass-gap",
        "wikipedia_topic": "Quantum physics",
        "title": "Yang–Mills existence and mass gap",
        "status": "out_of_scope",
        "hqiv": "Millennium-class mathematical QFT existence is outside the HQIV patch mandate; colour confinement is approached via discrete networks, not an analytic mass-gap proof.",
        "papers": [],
        "arena_metrics": [],
    },
    {
        "id": "dark-energy",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Dark energy / cosmic acceleration",
        "status": "partial",
        "hqiv": "Acceleration read through G_eff(φ) = φ^(3/5) and observer-centric lapse; coincidence with matter density is tied to selecting the present hypersurface, not a fitted ΛCDM extension.",
        "papers": ["unified-framework"],
        "arena_metrics": ["lapse_factor_ref_point"],
    },
    {
        "id": "dark-matter",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Dark matter",
        "status": "reinterpreted",
        "hqiv": "Galactic dynamics attributed to inertia screening on horizons (SPARC-style readouts) rather than a new stable particle species; Arena tracks flyby/SPARC residuals.",
        "papers": ["unified-framework"],
        "arena_metrics": ["orbital_flyby_sparc_model_residual"],
    },
    {
        "id": "matter-antimatter-asymmetry",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Matter–antimatter asymmetry",
        "status": "addressed",
        "hqiv": "Baryon excess (ETA10) from first-principles dynamic shell integrator (curvature + vev + binding feedback; paper script gives ~6.19782 not 6.1). Not a bolt-on CP parameter. Big mainstream problem: fitted in ΛCDM (Ω_b h² from BBN+CMB, depends on multiple params + ICs); no first-principles from SM.",
        "papers": ["unified-framework", "finite-mode-kirchhoff", "tuft"],
        "arena_metrics": ["bbn_eta10"],
    },
    {
        "id": "hubble-tension",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Hubble tension",
        "status": "partial",
        "hqiv": "Wall-clock vs apparent age split (51.2 Gyr vs 13.8 Gyr appearance) changes distance–redshift readouts; explicit H₀ pipeline comparison is ongoing phenomenology.",
        "papers": ["unified-framework", "finite-mode-kirchhoff"],
        "arena_metrics": ["paper_comparisons_max_abs_z"],
    },
    {
        "id": "horizon-problem",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Horizon / homogeneity problem",
        "status": "partial",
        "hqiv": "Large-scale homogeneity argued from entanglement monogamy on overlapping horizons and discrete shell coupling; not a standard slow-roll inflation field postulate.",
        "papers": ["unified-framework", "finite-mode-kirchhoff"],
        "arena_metrics": [],
    },
    {
        "id": "generations-of-matter",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Three generations of quarks and leptons",
        "status": "addressed",
        "hqiv": "Spin(8) triality on the octonionic carrier organizes three generations; Yukawa hierarchies read from detuned horizon surfaces (mass ladder programme).",
        "papers": ["unified-framework", "so8-closure", "tuft-sm-lagrangian", "hep-decay-readout"],
        "arena_metrics": [
            "derived_proton_mass_MeV",
            "hep_decay_panel_mean_z",
            "hep_decay_structural_pass_rate",
        ],
    },
    {
        "id": "hierarchy-problem",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Hierarchy problem (gravity vs other forces)",
        "status": "addressed",
        "hqiv": "All scales descend from single lock-in shell (m~4) + lattice combinatorics + α=3/5; no quadratic UV divergences or bare-parameter tuning. Hierarchy tuning exponent = 0 (natural).",
        "papers": ["unified-framework", "tuft-sm-lagrangian"],
        "arena_metrics": ["alpha_GUT", "hierarchy_tuning_exponent"],
    },
    {
        "id": "color-confinement",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Color confinement",
        "status": "partial",
        "hqiv": "Confinement built into subatomic network weights and string-tension readouts; analytic proof for arbitrary non-abelian gauge theory is not claimed.",
        "papers": ["unified-framework", "tuft-sm-lagrangian", "hep-decay-readout", "gluon-curvature"],
        "arena_metrics": [
            "derived_proton_mass_MeV",
            "hep_decay_panel_mean_z",
            "hep_decay_panel_max_z",
            "hep_decay_structural_pass_rate",
        ],
    },
    {
        "id": "strong-cp",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Strong CP problem / axions",
        "status": "partial",
        "hqiv": "No independent gluon DOF — CP-odd holonomy from Fano ledger (HEP decay readout); no fitted θ_QCD knob.",
        "papers": ["hep-decay-readout", "gluon-curvature", "tuft-sm-lagrangian"],
        "arena_metrics": ["hep_decay_panel_mean_z", "hep_decay_panel_max_z"],
    },
    {
        "id": "neutrino-mass",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Neutrino masses and hierarchy",
        "status": "open",
        "hqiv": "Lepton ladder work exists in pyhqiv but absolute neutrino masses and Majorana vs Dirac nature are not closed in the Zenodo paper series yet.",
        "papers": ["tuft-sm-lagrangian"],
        "arena_metrics": [],
    },
    {
        "id": "muon-g2",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Muon anomalous magnetic moment (g − 2)",
        "status": "open",
        "hqiv": "Not yet a headline comparison in the Arena paper-comparison suite; high priority for new tests with PDG error bars.",
        "papers": [],
        "arena_metrics": ["paper_comparisons_max_abs_z"],
    },
    {
        "id": "galaxy-rotation",
        "wikipedia_topic": "Astronomy and astrophysics",
        "title": "Galaxy rotation curves",
        "status": "partial",
        "hqiv": "Rotation curves from horizon-modified inertia (live pyhqiv.orbital + SPARC catalog tests); residual metric is an active Arena improvement target.",
        "papers": ["unified-framework"],
        "arena_metrics": ["orbital_flyby_sparc_model_residual"],
    },
    {
        "id": "flyby-anomaly",
        "wikipedia_topic": "Astronomy and astrophysics",
        "title": "Flyby anomaly",
        "status": "partial",
        "hqiv": "Inertia screening formula compared to literature flyby residuals in test data; dynamic corrections compete on Arena σ.",
        "papers": ["unified-framework"],
        "arena_metrics": ["orbital_flyby_sparc_model_residual"],
    },
    {
        "id": "cmb-birefringence",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Cosmic birefringence / large-scale polarization",
        "status": "addressed",
        "hqiv": "Predicted rotation from α = 3/5 and wall-clock age; paper script ~0.379° vs Planck PR4 0.342±0.094 (~0.4σ). Python uses witness 0.3 (Lean) but computes match via surface_wave_self_clock + quantum_mechanics birefringence_redshift.",
        "papers": ["finite-mode-kirchhoff", "auxiliary-fields"],
        "arena_metrics": ["cmb_birefringence_z", "paper_comparisons_max_abs_z"],
    },
    {
        "id": "proton-decay-spin",
        "wikipedia_topic": "High-energy / particle physics",
        "title": "Proton stability and spin crisis",
        "status": "partial",
        "hqiv": "Proton mass anchored at reference shell; spin decomposition from 8×8 composites is programme work — decay lifetime not central claim.",
        "papers": ["tuft-sm-lagrangian"],
        "arena_metrics": ["derived_proton_mass_MeV"],
    },
    {
        "id": "nuclear-binding",
        "wikipedia_topic": "Nuclear physics",
        "title": "Nuclear force and binding energies",
        "status": "partial",
        "hqiv": "HorizonNetwork + 8×8 binding pipeline; light nuclei compared with explicit σ in tests (deuteron gap documented).",
        "papers": ["tuft-sm-lagrangian"],
        "arena_metrics": ["paper_comparisons_max_abs_z"],
    },
    {
        "id": "high-tc-superconductivity",
        "wikipedia_topic": "Condensed matter physics",
        "title": "High-temperature superconductivity",
        "status": "out_of_scope",
        "hqiv": "Condensed-matter mechanism not part of the horizon lattice spine; thermo/allotrope Arena metrics are separate phenomenology hooks.",
        "papers": [],
        "arena_metrics": ["thermo_allotrope_phase_residual", "water_h2o_melt_T_residual_K"],
    },
    {
        "id": "navier-stokes",
        "wikipedia_topic": "Fluid dynamics",
        "title": "Navier–Stokes existence and smoothness",
        "status": "out_of_scope",
        "hqiv": "Millennium problem in mathematics, not targeted by HQIV.",
        "papers": [],
        "arena_metrics": [],
    },
    {
        "id": "abiogenesis",
        "wikipedia_topic": "Biophysics",
        "title": "Abiogenesis",
        "status": "out_of_scope",
        "hqiv": "Origin of life is outside the physics patch; auxiliary-field paper mentions biochemistry only as future simulation regime.",
        "papers": ["auxiliary-fields"],
        "arena_metrics": [],
    },
    {
        "id": "water-anomalies-llpt",
        "wikipedia_topic": "Condensed matter physics",
        "title": "Water anomalies and liquid–liquid critical point",
        "status": "partial",
        "hqiv": "Generalized (T,P) phase engine: motif end members (LDL/HDL ρ_curv), two-branch free-energy minimizer, latent f(1−f) barrier, γ² Widom window, and metastable_liquid branch below T_melt. Sciortino/Kim/Li grade readouts only — never inputs. Lean: PhaseDiagramMixture.",
        "papers": ["thermodynamics-arrow", "unified-framework"],
        "arena_metrics": [
            "water_phase_diagram_structural_pass_rate",
            "water_metastable_liquid_at_llcp",
            "water_h2o_melt_T_residual_K",
            "water_llcp_observation_distance",
            "water_widom_peak_temperature_residual_K",
            "water_widom_free_energy_peak_residual_K",
            "water_widom_gamma2_window_alignment_K",
            "water_nucleation_defect_ldl_excess",
            "water_hoh_angle_taxonomy_open_gap_deg",
            "water_h2o_bond_angle_residual_deg",
            "thermo_allotrope_phase_residual",
        ],
    },
    {
        "id": "protein-folding",
        "wikipedia_topic": "Biophysics",
        "title": "Protein folding (structure prediction)",
        "status": "partial",
        "hqiv": "Miniprotein Cα folds from derived peptide spine + NERF/staged closure — PDB witnesses grade readouts only (11/11 pass, mean ~2.03 Å RMSD, Trp-cage ~3.94 Å).",
        "papers": ["unified-framework", "tuft-sm-lagrangian"],
        "arena_metrics": [
            "miniprotein_mean_ca_rmsd",
            "miniprotein_trp_cage_ca_rmsd",
            "miniprotein_fold_pass_fraction",
            "protein_hydrophobic_interface_ldl_excess",
        ],
    },
    {
        "id": "cosmological-constant",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Cosmological constant / vacuum energy problem",
        "status": "addressed",
        "hqiv": "Finite sum over discrete null-lattice modes 0 to m_now (causal horizon cutoff) for vacuum zero-point ½N(m)ω(m); exact match to paper script. Gives observed small ρ_vac naturally (no tuning).",
        "papers": ["finite_mode_kirchhoff"],
        "arena_metrics": ["vacuum_energy_discrepancy"],
    },
    {
        "id": "flatness-problem",
        "wikipedia_topic": "Cosmology and general relativity",
        "title": "Flatness problem / initial conditions for Ω_k",
        "status": "addressed",
        "hqiv": "Ω_k(now) small positive ~0.0098 derived from lattice shell integral at current age m_now (dynamic with universe age); within obs bounds without initial fine-tuning.",
        "papers": ["lightcone", "now"],
        "arena_metrics": ["flatness_tuning_exponent", "omega_k_present_now"],
    },
]


def _load_arena(path: Path | None) -> dict:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import tempfile

    from pyhqiv.arena import build_default_metrics, compute_score, serialize_score  # type: ignore

    tmp = Path(tempfile.mkdtemp()) / "arena_results.json"
    res = compute_score(metrics=build_default_metrics())
    return serialize_score(res, str(tmp))


def _split_metrics(arena: dict) -> tuple[list[dict], list[dict]]:
    cores: list[dict] = []
    phenom: list[dict] = []
    try:
        from pyhqiv.arena.metrics import METRIC_REGISTRY
        reg = METRIC_REGISTRY()
    except Exception:
        reg = {}
    for m in arena.get("metrics", []):
        reg_m = reg.get(m["name"])
        row = {
            "name": m["name"],
            "value": m["value"],
            "reference": m["reference"],
            "rel_err": m["rel_err"],
            "unit": m.get("unit", ""),
            "protected": m.get("protected", False),
            "desc": m.get("desc", "") or (reg_m.desc if reg_m else ""),
            "mainstream_note": (reg_m.mainstream_note if reg_m else "") or m.get("mainstream_note", ""),
        }
        if m.get("protected"):
            cores.append(row)
        else:
            phenom.append(row)
    return cores, phenom


def build_document(arena: dict) -> dict:
    cores, phenom = _split_metrics(arena)
    try:
        from pyhqiv._version import __version__ as pyv  # type: ignore
    except Exception:
        pyv = "unknown"

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pyhqiv_version": pyv,
        "wikipedia": {
            "title": "List of unsolved problems in physics",
            "url": WIKIPEDIA_LIST_URL,
            "license_note": "Problem titles and grouping follow the English Wikipedia list; HQIV status text is programme commentary, not Wikipedia content.",
        },
        "sigma_snapshot": {
            "overall_score": arena.get("overall_score"),
            "sigma_weighted": arena.get("sigma_weighted"),
            "num_protected_regressions": arena.get("num_regressed_protected", 0),
            "alignment_cores": cores,
            "phenomenology_metrics": phenom,
            "note": (
                "Protected metrics are Lean↔Python exact witnesses (rel_err = 0). "
                "Phenomenology metrics use PDG/Planck/literature error bars; "
                "Arena rewards broad reduction of rel_err across many observables ('sigma everywhere'). "
                "sigma_weighted mixes large-magnitude witnesses — use per-metric rel_err for physics gaps."
            ),
        },
        "status_legend": {
            "addressed": "Explicit mechanism or derivation in the published HQIV spine.",
            "partial": "Direction claimed; quantitative closure or uniqueness still open.",
            "reinterpreted": "Problem reframed (e.g. dark matter as inertia screening).",
            "open": "Acknowledged gap; no paper claim yet.",
            "out_of_scope": "Not a current programme target.",
        },
        "problems": PROBLEMS,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arena", type=str, default=None, help="Existing arena_results.json (else compute fresh)")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "arena" / "programme_sigma.json"))
    args = p.parse_args(argv)

    arena_path = Path(args.arena) if args.arena else None
    arena = _load_arena(arena_path)
    doc = build_document(arena)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(doc['problems'])} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
