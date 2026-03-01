"""
HQIV CMB: mollview map + side-by-side C_ℓ comparison with Planck.

Run full pipeline from Planck epoch to now, plot full-sky map (μK) and
optionally compare C_ℓ to Planck. Requires pyhqiv[cosmology] (healpy) for map.

Usage:
  python examples/cmb_mollview_planck.py              # C_ℓ only (no healpy)
  python examples/cmb_mollview_planck.py --mollview  # map + C_ℓ (needs healpy)
  python examples/cmb_mollview_planck.py --mollview --nside 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pyhqiv.cosmology import HQIVUniverseEvolver


def main() -> None:
    ap = argparse.ArgumentParser(description="HQIV CMB mollview + Planck C_ℓ comparison")
    ap.add_argument("--nside", type=int, default=64, help="HEALPix nside (small for quick run)")
    ap.add_argument("--max-ell", type=int, default=1500, help="Max multipole for C_ℓ")
    ap.add_argument("--mollview", action="store_true", help="Show full-sky mollview (requires healpy)")
    ap.add_argument("--save", type=str, default="", help="Save figure path (e.g. cmb_comparison.png)")
    args = ap.parse_args()

    evolver = HQIVUniverseEvolver(nside=args.nside, max_ell=args.max_ell)
    result = evolver.run_from_T_Pl_to_now()

    print(f"σ₈ = {result['sigma8']:.4f}")
    print(f"C_ℓ TT length: {len(result['C_ell_TT'])}, ell range [2, {result['ell'][-1]:.0f}]")

    # C_ℓ plot: HQIV vs Planck-like reference (phenomenological)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None:
        ell = result["ell"]
        c_tt = result["C_ell_TT"]
        # D_ℓ = ell(ell+1)/(2π) * C_ℓ in μK²
        d_tt = ell * (ell + 1) / (2 * np.pi) * c_tt

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.semilogy(ell, d_tt, label="HQIV (lapse + ISW/Rees–Sciama)", color="C0")
        # Placeholder: Planck-like curve (same shape, different amplitude for comparison)
        # In production, load Planck 2018 C_ℓ from CDN or data file.
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [µK²]")
        ax.set_title("HQIV CMB TT vs Planck (reference)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2, min(args.max_ell, ell[-1]))
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Saved {args.save}")
        plt.show()

    if args.mollview:
        t_map = result.get("T_map_muK")
        if t_map is None:
            print("Mollview skipped: no map (install healpy: pip install pyhqiv[cosmology])")
            return
        try:
            import healpy as hp
            hp.mollview(t_map, title="HQIV CMB from Planck epoch to now (µK)")
            plt = __import__("matplotlib.pyplot", fromlist=["show"])
            plt.show()
        except ImportError:
            print("healpy required for mollview. pip install pyhqiv[cosmology]")


if __name__ == "__main__":
    main()
