"""
Full run_from_T_Pl_to_now(): multipole spectrum (C_ℓ) and σ₈.

Usage:
  python examples/run_from_T_Pl_to_now_demo.py
  python examples/run_from_T_Pl_to_now_demo.py --nside 128 --no-plot  # no figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pyhqiv import HQIVUniverseEvolver


def main() -> None:
    ap = argparse.ArgumentParser(description="Run HQIV evolver: T_Pl → now, C_ℓ + σ₈")
    ap.add_argument("--nside", type=int, default=64, help="HEALPix nside")
    ap.add_argument("--max-ell", type=int, default=1500, help="Max multipole")
    ap.add_argument("--frame-velocity", type=float, default=370.0, help="Kinematic dipole v (km/s)")
    ap.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = ap.parse_args()

    evolver = HQIVUniverseEvolver(
        nside=args.nside,
        max_ell=max(args.max_ell, 1500),
        frame_velocity_km_s=args.frame_velocity,
    )
    result = evolver.run_from_T_Pl_to_now()

    # Summary
    print("=== run_from_T_Pl_to_now() ===")
    print(f"  σ₈ = {result['sigma8']:.6f}")
    print(f"  ell: [2, {result['ell'][-1]:.0f}], len = {len(result['ell'])}")
    print(f"  C_ell_TT: [{result['C_ell_TT'].min():.4e}, {result['C_ell_TT'].max():.4e}] μK²")
    if result.get("T_map_muK") is not None:
        t = result["T_map_muK"]
        print(f"  T_map_muK: npix={len(t)}, mean={t.mean():.2f} μK, std={t.std():.2f} μK")
    else:
        print("  T_map_muK: None (install pyhqiv[cosmology] for healpy)")
    print(f"  Keys: {list(result.keys())}")

    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not found, skip plot)")
        return

    ell = result["ell"]
    c_tt = result["C_ell_TT"]
    d_tt = ell * (ell + 1) / (2 * np.pi) * c_tt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogy(ell, d_tt, color="C0", label="HQIV TT")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [μK²]")
    ax.set_title(f"run_from_T_Pl_to_now() — σ₈ = {result['sigma8']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, ell[-1])
    fig.tight_layout()
    out = Path(__file__).parent / "run_from_T_Pl_to_now_demo.png"
    fig.savefig(out, dpi=120)
    print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
