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


def _with_lean_scripts() -> Path | None:
    root = _hqiv_lean_root()
    if root is None:
        return None
    scripts = root / "scripts"
    repo = root
    for p in (str(repo), str(scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


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
