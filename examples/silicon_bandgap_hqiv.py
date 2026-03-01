"""
Silicon band-gap example: known ~1.12 eV gap + HQIV shift.

This script demonstrates the semiconductor API (compute_band_gap, hqiv_potential_shift,
high_symmetry_k_path) with a minimal model: a small set of k-points and mock
eigenvalues that approximate Si (indirect gap ~1.12 eV at Gamma–X). The HQIV
potential shift is then applied to show how the gap changes with φ and δ̇θ′.

For a full DFT workflow you would:
  1. Build an ASE/PySCF Si crystal and run a band structure.
  2. Pass the resulting eigenvalues to compute_band_gap(..., phi_avg=..., dot_delta_theta_avg=...).
  3. Use high_symmetry_k_path() to generate the k-path (e.g. G–X–W–G).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pyhqiv import (
    high_symmetry_k_path,
    compute_band_gap,
    hqiv_potential_shift,
    semiconductors,
)


def main() -> None:
    # Cubic Si lattice (5.43 Å)
    a_si = 5.43
    lat = a_si * np.eye(3)
    path = "GXWG"
    k_cart, k_frac, segments = high_symmetry_k_path(lat, path, npoints=40)
    print(f"K-path: {path}, {len(k_cart)} k-points")
    print(f"Segment labels: {segments}")

    # Mock eigenvalues: (n_k, n_bands) — 4 valence, 4 conduction (minimal)
    n_k, n_bands = len(k_cart), 8
    np.random.seed(42)
    ev = np.zeros((n_k, n_bands))
    ev[:, :4] = -2.0 - 0.5 * np.random.rand(n_k, 4)   # valence
    ev[:, 4:] = 1.0 + 0.5 * np.random.rand(n_k, 4)    # conduction
    # Rough indirect gap ~1.12 at Gamma
    ev[0, 3] = -0.5
    ev[0, 4] = 0.62
    gap_nominal = 0.62 - (-0.5)
    print(f"Nominal gap (mock): {gap_nominal:.3f} eV")

    gap, gap_type = compute_band_gap(ev, phi_avg=0.0, dot_delta_theta_avg=0.0)
    print(f"compute_band_gap (no HQIV): {gap:.3f} eV ({gap_type})")

    # HQIV shift: V_shift = gamma * phi_avg * dot_delta_theta_avg
    phi_avg = 1e-10   # small for illustration (eV-scale)
    dot_delta_theta_avg = 1e-18
    v_shift = hqiv_potential_shift(phi_avg, dot_delta_theta_avg)
    print(f"HQIV potential shift: {v_shift:.3e} eV")
    gap_hqiv, _ = compute_band_gap(
        ev, phi_avg=phi_avg, dot_delta_theta_avg=dot_delta_theta_avg
    )
    print(f"compute_band_gap (with HQIV): {gap_hqiv:.3f} eV")

    # DOS on a grid
    energies = np.linspace(-3, 2, 100)
    rho = semiconductors.dos(ev, energies, sigma=0.1)
    print(f"DOS computed on {len(energies)} points, max ρ = {rho.max():.4f}")

    print("Done. Use pyhqiv.semiconductors and high_symmetry_k_path in your DFT pipeline.")


if __name__ == "__main__":
    main()
