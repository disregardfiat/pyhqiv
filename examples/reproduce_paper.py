"""
Reproduce all paper table values and optional figures.

Usage:
  python reproduce_paper.py              # print tables only
  python reproduce_paper.py --plot      # tables + matplotlib figures
  python reproduce_paper.py --plot --pyvista   # also 3D δE surface (if pyvista installed)

Paper: Ettinger, Steven Jr, Horizon-Quantized Informational Vacuum (HQIV), Zenodo 2026.
DOI: 10.5281/zenodo.18794889
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root so pyhqiv is importable when run from examples/
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pyhqiv.constants import (
    GAMMA,
    ALPHA,
    T_LOCK_GEV,
    T_CMB_K,
    T_PL_GEV,
    M_TRANS,
    COMBINATORIAL_INVARIANT,
    OMEGA_TRUE_K_PAPER,
    LAPSE_COMPRESSION_PAPER,
    AGE_WALL_GYR_PAPER,
    AGE_APPARENT_GYR_PAPER,
    HBAR_C_EV_ANG,
    A_LOC_ANG,
)
from pyhqiv.lattice import (
    DiscreteNullLattice,
    discrete_mode_count,
    cumulative_mode_count,
    curvature_imprint_delta_E,
    omega_k_from_shell_integral,
)
from pyhqiv.phase import HQIVPhaseLift
from pyhqiv.algebra import OctonionHQIVAlgebra


def print_main_table() -> None:
    """Print main paper constants and derived table."""
    lattice = DiscreteNullLattice(m_trans=M_TRANS, gamma=GAMMA, alpha=ALPHA)
    result = lattice.evolve_to_cmb(T0_K=T_CMB_K)

    print("=" * 60)
    print("HQIV paper table (reproduced)")
    print("=" * 60)
    print()
    print("Constants (paper values)")
    print("-" * 40)
    print(f"  γ (entanglement monogamy)     {GAMMA}")
    print(f"  α (G_eff exponent)             {ALPHA}")
    print(f"  T_CMB (K)                     {T_CMB_K}")
    print(f"  T_Pl (GeV)                    {T_PL_GEV}")
    print(f"  T_lock (GeV)                  {T_LOCK_GEV}")
    print(f"  m_trans                       {M_TRANS}")
    print(f"  6^7 √3 (combinatorial)       {COMBINATORIAL_INVARIANT:.6e}")
    print(f"  ℏc (eV·Å)                    {HBAR_C_EV_ANG:.6e}")
    print(f"  a_loc (Å)                     {A_LOC_ANG}")
    print()
    print("Lattice → CMB (evolve_to_cmb)")
    print("-" * 40)
    print(f"  Ω_k^true                      {result['Omega_true_k']:.6f}")
    print(f"  Wall-clock age (Gyr)          {result['age_wall_Gyr']}")
    print(f"  Apparent age (Gyr)            {result['age_apparent_Gyr']}")
    print(f"  Lapse compression             {result['lapse_compression']}")
    print()
    print("Reference (paper table)")
    print("-" * 40)
    print(f"  Ω_k^true (paper)              {OMEGA_TRUE_K_PAPER}")
    print(f"  Wall-clock (paper)            {AGE_WALL_GYR_PAPER} Gyr")
    print(f"  Apparent (paper)              {AGE_APPARENT_GYR_PAPER} Gyr")
    print(f"  Lapse (paper)                 {LAPSE_COMPRESSION_PAPER}")
    print("=" * 60)


def print_mode_count_table() -> None:
    """Print mode count sample (dN(m) and cumulative)."""
    print()
    print("Mode counts (sample shells)")
    print("-" * 50)
    print("  m     dN(m)=8·C(m+2,2)   cumulative(0..m)")
    print("-" * 50)
    for m in [0, 1, 2, 10, 100, 500]:
        dN = discrete_mode_count(m)
        cum = cumulative_mode_count(m + 1)  # sum over 0..m
        print(f"  {m:3d}   {dN:12.0f}   {cum:18.0f}")
    print("  ...")
    total_500 = cumulative_mode_count(501)
    print(f"  (m=0..500 total new modes)     {total_500:.0f}")
    print()


def print_delta_E_sample() -> None:
    """Print δE(m) at sample m with T(m)=T_Pl/(m+1)."""
    print("Curvature imprint δE(m) sample (T(m)=E_0/(m+1), E_0=T_Pl)")
    print("-" * 50)
    m_arr = np.array([0, 1, 10, 100, 499])
    T = T_PL_GEV / (m_arr + 1.0)
    delta_E = curvature_imprint_delta_E(m_arr, T)
    print("  m       T (GeV)        δE(m)")
    print("-" * 50)
    for i, m in enumerate(m_arr):
        print(f"  {int(m):3d}   {T[i]:12.6e}   {delta_E[i]:12.6e}")
    print()


def print_omega_k_vs_m_trans() -> None:
    """Print Ω_k^true for a few m_trans values."""
    print("Ω_k^true vs m_trans (calibrated to 0.0098 at m_trans=500)")
    print("-" * 40)
    for m in [100, 300, 500, 501]:
        omega = omega_k_from_shell_integral(m_trans=m)
        print(f"  m_trans = {m:3d}   Ω_k^true = {omega:.6f}")
    print()


def print_algebra_checks() -> None:
    """Print so(8) dimension and hypercharge block check."""
    alg = OctonionHQIVAlgebra(verbose=False)
    dim, history = alg.lie_closure_dimension()
    data = alg.hypercharge_paper_data()
    print("Algebra (so(8) closure, hypercharge)")
    print("-" * 40)
    print(f"  Lie closure dimension            {dim}")
    if data:
        ev = data["eigenvalues_i_block"]
        print(f"  Hypercharge 4×4 block |eigenvalues|  {np.abs(ev)}")
        print(f"  Block entry error                {data['block_entry_error']:.2e}")
        print(f"  max ‖[Y,g₂]‖                     {data['max_commutation_with_g2']:.2e}")
    print()


def run_plots(use_pyvista: bool = False) -> None:
    """Generate optional matplotlib (and optionally pyvista) figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plots. pip install matplotlib")
        return

    lattice = DiscreteNullLattice(m_trans=M_TRANS, gamma=GAMMA, alpha=ALPHA)
    m_grid = np.arange(0, lattice.m_trans, dtype=float)
    delta_E = lattice.get_delta_E_grid()
    cum = lattice.get_cumulative_mode_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(m_grid, delta_E, color="C0")
    axes[0].set_xlabel("Shell index m")
    axes[0].set_ylabel("δE(m)")
    axes[0].set_title("Curvature imprint δE(m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(0, lattice.m_trans + 1), cum, color="C1")
    axes[1].set_xlabel("Shell index m")
    axes[1].set_ylabel("Cumulative mode count")
    axes[1].set_title("Cumulative mode count (hockey-stick)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "reproduce_paper_figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "delta_E_and_cumulative_modes.png", dpi=150)
    plt.close()
    print(f"Saved {out / 'delta_E_and_cumulative_modes.png'}")

    # Phase lift: δθ′(E′)
    phase = HQIVPhaseLift(gamma=GAMMA)
    E_prime = np.linspace(0.01, 2.0, 200)
    dtheta = phase.delta_theta_prime(E_prime)
    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.plot(E_prime, dtheta, color="C2")
    ax.set_xlabel("E′")
    ax.set_ylabel("δθ′(E′)")
    ax.set_title("Phase-horizon angle δθ′(E′) = arctan(E′)×(π/2)")
    ax.grid(True, alpha=0.3)
    fig2.savefig(out / "phase_lift_delta_theta_prime.png", dpi=150)
    plt.close()
    print(f"Saved {out / 'phase_lift_delta_theta_prime.png'}")

    if use_pyvista:
        try:
            import pyvista as pv
        except ImportError:
            print("pyvista not installed; skip 3D. pip install pyvista")
            return
        # Simple 3D: surface m × (small grid) vs δE
        m_1d = np.arange(0, min(100, lattice.m_trans), dtype=float)
        delta_E_1d = lattice.get_delta_E_grid(E_0_factor=1.0)[: len(m_1d)]
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([m_1d, np.zeros_like(m_1d), delta_E_1d])
        grid.dimensions = [len(m_1d), 1, 1]
        grid = grid.cast_to_unstructured_grid()
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars=delta_E_1d, show_edges=True)
        plotter.camera_position = "xy"
        plotter.save_graphic(out / "delta_E_3d.svg")
        plotter.close()
        print(f"Saved {out / 'delta_E_3d.svg'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce HQIV paper tables and optional plots")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib figures")
    parser.add_argument("--pyvista", action="store_true", help="Also generate pyvista 3D (requires --plot)")
    args = parser.parse_args()

    print_main_table()
    print_mode_count_table()
    print_delta_E_sample()
    print_omega_k_vs_m_trans()
    print_algebra_checks()

    if args.plot:
        run_plots(use_pyvista=args.pyvista)


if __name__ == "__main__":
    main()
