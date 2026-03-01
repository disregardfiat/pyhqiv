"""
Seed for the CMB pipeline from HQIV bulk.py (authoritative until baryogenesis complete).

The paper and HQIV repo use horizon_modes/python/bulk.py as the single source of truth
for the early universe: baryogenesis to lock-in, then modified Friedmann to T_cmb.
This module loads that bulk output and returns a normalized seed dict so the
pyhqiv → CMB pipeline (cosmology_full, evolver) uses bulk-derived Ω_k, H₀, η, and
table path instead of the in-package DiscreteNullLattice.

When the HQIV repo is not available or bulk fails, get_bulk_seed() returns None
and the pipeline falls back to the in-package lattice (evolve_to_cmb).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["get_bulk_seed", "BULK_SEED_AVAILABLE"]


def _find_horizon_modes_python() -> Optional[Path]:
    """Resolve horizon_modes/python directory (HQIV repo)."""
    # Explicit env (e.g. HQIV_REPO or path to horizon_modes/python)
    env = os.environ.get("HQIV_HORIZON_MODES_PYTHON") or os.environ.get("HQIV_REPO")
    if env:
        p = Path(env).resolve()
        if (p / "horizon_modes" / "python").is_dir():
            return (p / "horizon_modes" / "python").resolve()
        if (p / "python").is_dir() and (p / "python" / "bulk.py").exists():
            return (p / "python").resolve()
        if p.is_dir() and (p / "bulk.py").exists():
            return p.resolve()
    # Relative to this package: .../Repos/hqvmpy/src/pyhqiv → .../Repos/HQIV/horizon_modes/python
    this_dir = Path(__file__).resolve().parent
    for candidate in [
        this_dir.parent.parent.parent / "HQIV" / "horizon_modes" / "python",
        this_dir.parent.parent.parent / "hqiv" / "horizon_modes" / "python",
        this_dir.parent.parent / "HQIV" / "horizon_modes" / "python",
        this_dir.parent.parent / "hqiv" / "horizon_modes" / "python",
    ]:
        if candidate.is_dir() and (candidate / "bulk.py").exists():
            return candidate.resolve()
    return None


def get_bulk_seed(
    hqiv_path: Optional[str] = None,
    n_steps: int = 6000,
    n_loga: int = 4000,
    T_cmb_k: float = 2.725,
    outpath: Optional[str] = None,
    quiet: bool = True,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Run HQIV bulk.py (baryogenesis → lock-in → Friedmann to T_cmb) and return seed for CMB pipeline.

    When the HQIV repo is available at hqiv_path (or HQIV_REPO / HQIV_HORIZON_MODES_PYTHON),
    runs forward_4d_evolution() and returns a dict with:
      - omega_k_true, eta, a_lock, T_lock_gev, H0_gev, H0_km_s_Mpc
      - table_path (lattice table for CLASS / background)
      - Omega_true_k (alias), lapse_compression (paper value when using bulk)
    Suitable for passing to cosmology_full / evolver as the authoritative seed until baryogenesis
    is complete. Returns None if bulk is not available or run fails.
    """
    hm = _find_horizon_modes_python() if not hqiv_path else Path(hqiv_path).resolve()
    if hqiv_path and not hm.is_dir():
        hm = (
            Path(hqiv_path) / "horizon_modes" / "python"
            if (Path(hqiv_path) / "horizon_modes" / "python").exists()
            else Path(hqiv_path)
        )
    if not hm or not (hm / "bulk.py").exists():
        return None

    saved_path = list(sys.path)
    try:
        sys.path.insert(0, str(hm))
        import bulk as hqiv_bulk  # type: ignore
    except Exception:
        if not quiet:
            import traceback

            traceback.print_exc()
        return None
    finally:
        sys.path[:] = saved_path

    H_GEV_TO_KM_S_MPC = 1.56e38 * 2.998e5  # from bulk.py
    try:
        result = hqiv_bulk.forward_4d_evolution(
            n_steps=n_steps,
            n_loga=n_loga,
            T_cmb_k=T_cmb_k,
            outpath=outpath or (str(hm / "hqiv_lattice_table.dat")),
            **kwargs,
        )
    except Exception:
        if not quiet:
            import traceback

            traceback.print_exc()
        return None

    H0_km_s_Mpc = float(result["H0_gev"]) * H_GEV_TO_KM_S_MPC
    # Paper lapse when using bulk background (bulk does not compute lapse)
    LAPSE_PAPER = 3.96
    return {
        "omega_k_true": result["omega_k_true"],
        "Omega_true_k": result["omega_k_true"],
        "eta": result["eta"],
        "a_lock": result["a_lock"],
        "T_lock_gev": result["T_lock_gev"],
        "H0_gev": result["H0_gev"],
        "H0_km_s_Mpc": H0_km_s_Mpc,
        "table_path": result["outpath"],
        "lapse_compression": LAPSE_PAPER,
        "data": result.get("data"),
    }


def _check_available() -> bool:
    """True if horizon_modes/python/bulk.py is findable (does not run bulk)."""
    return _find_horizon_modes_python() is not None


# True if HQIV horizon_modes/python is on the expected path (does not run bulk)
BULK_SEED_AVAILABLE = _check_available()
