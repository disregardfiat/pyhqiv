#!/usr/bin/env python3
"""
Complete HQIV Integrated Space Model: Sun, Earth orbit, high-z galaxy.

Run: python examples/hqiv_space_model_full.py [--plot]

Computes:
- φ(r) in solar core, standard vs HQIV table
- Earth–Sun orbit with lapse (proper time τ vs t)
- Redshift decomposition for a high-z galaxy (e.g. z_app = 11)
- Optional matplotlib plots: φ(r), z decomposition, lapse-corrected orbit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path when run from repo root
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

from pyhqiv.constants import (
    LAPSE_COMPRESSION_PAPER,
    OMEGA_TRUE_K_PAPER,
    AGE_WALL_GYR_PAPER,
    AGE_APPARENT_GYR_PAPER,
)
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.solar_core import (
    HQIVSolarCore,
    phi_solar_radial_profile,
    R_SUN_M,
    RHO_CORE_CGS,
    T_CORE_MK,
)
from pyhqiv.redshift import HQIVRedshift, z_total_apparent
from pyhqiv.orbit import HQIVOrbit, parker_perihelion_lapse


def run_solar_core() -> dict:
    """φ(r) profile and standard vs HQIV table."""
    sun = HQIVSolarCore(
        R_star=R_SUN_M,
        r_core_frac=0.2,
        rho_c_cgs=RHO_CORE_CGS,
        T_c_MK=T_CORE_MK,
    )
    table = sun.standard_vs_hqiv_table(lapse_compression=LAPSE_COMPRESSION_PAPER)
    r_m, phi, f = phi_solar_radial_profile(64, R_sun=R_SUN_M)
    return {
        "table": table,
        "r_m": r_m,
        "phi": phi,
        "f": f,
        "phi_core": float(sun.phi(0.0)),
        "f_core": float(sun.lapse_factor(0.0)),
    }


def run_earth_orbit() -> dict:
    """Earth–Sun orbit with proper time τ from lapse."""
    orb = HQIVOrbit()
    t, r, v, tau = orb.earth_sun_example(n_steps=500, n_orbits=0.25)
    # Lapse at 1 au
    r_au = np.array([1.495978707e11, 0.0, 0.0])
    f_earth = float(np.asarray(orb.lapse_f(r_au.reshape(1, 3))).flat[0])
    return {
        "t_s": t,
        "r_m": r,
        "v_m_s": v,
        "tau_s": tau,
        "f_earth": f_earth,
        "orbit": orb,
    }


def run_redshift(z_app: float = 11.0) -> dict:
    """Redshift decomposition for high-z galaxy (e.g. GN-z11)."""
    lattice = DiscreteNullLattice()
    red = HQIVRedshift(
        lapse_compression=LAPSE_COMPRESSION_PAPER,
        age_wall_Gyr=AGE_WALL_GYR_PAPER,
    ).with_lattice(lattice)
    # Pure expansion: z_exp = z_app, z_grav = 0, z_HQIV = 0
    # Degeneracy: assume 10% of z from HQIV
    decomp = red.decompose_from_apparent(z_app, z_grav=0.0, z_HQIV_fraction=0.1)
    age_wall = red.wall_clock_age_at_emission(np.array([z_app]))
    cosmo = red.cosmology_result()
    return {
        "z_app": z_app,
        "decompose_pure_exp": red.decompose_from_apparent(z_app, z_HQIV_fraction=0.0),
        "decompose_10pct_HQIV": decomp,
        "age_wall_at_emission_Gyr": float(age_wall[0]),
        "cosmology": cosmo,
    }


def run_parker() -> float:
    """Lapse at Parker Solar Probe perihelion."""
    return parker_perihelion_lapse(0.05)


def main(do_plot: bool = False) -> None:
    print("=== HQIV Integrated Space Model (v1.0) ===\n")

    # 1) Solar core
    print("1. Solar core (φ(r), f(r), standard vs HQIV)")
    solar = run_solar_core()
    t = solar["table"]
    print(f"   T_c standard: {t['T_c_MK_standard']} MK")
    print(f"   T_c HQIV effective: {t['T_c_MK_HQIV_effective']:.4f} MK")
    print(f"   ρ_c: {t['rho_c_g_cm3']} g/cm³")
    print(f"   φ_core (m²/s²): {t['phi_core_m2_s2']:.4e}")
    print(f"   f_core: {t['f_core']:.6f}")
    print(f"   age apparent: {t['age_apparent_Gyr']} Gyr, wall-clock: {t['age_wall_clock_Gyr']:.2f} Gyr")
    print(f"   L shift factor (lapse): {t['L_shift_factor']}")

    # 2) Earth orbit
    print("\n2. Earth–Sun orbit (lapse, proper time)")
    orb_out = run_earth_orbit()
    print(f"   f at 1 au: {orb_out['f_earth']:.6f}")
    if orb_out["tau_s"] is not None:
        tau = orb_out["tau_s"]
        t_s = orb_out["t_s"]
        print(f"   Coordinate Δt (0.25 orbit): {t_s[-1] - t_s[0]:.2f} s")
        print(f"   Proper Δτ: {tau[-1] - tau[0]:.2f} s")
        print(f"   Ratio τ/t: {(tau[-1]-tau[0])/(t_s[-1]-t_s[0]):.6f}")

    # 3) Redshift
    print("\n3. Redshift decomposition (high-z galaxy z_app ≈ 11)")
    z_out = run_redshift(11.0)
    print(f"   z_app: {z_out['z_app']}")
    d0 = z_out["decompose_pure_exp"]
    d1 = z_out["decompose_10pct_HQIV"]
    print(f"   Pure expansion: z_exp={d0['z_exp']:.4f}, z_HQIV={d0['z_HQIV']:.4f}")
    print(f"   10% HQIV: z_exp={d1['z_exp']:.4f}, z_HQIV={d1['z_HQIV']:.4f}")
    print(f"   Wall-clock age at emission: {z_out['age_wall_at_emission_Gyr']:.4f} Gyr")
    print(f"   Ω_k^true: {z_out['cosmology']['Omega_true_k']:.4f}")
    print(f"   lapse_compression: {z_out['cosmology']['lapse_compression']}")

    # 4) Parker
    f_parker = run_parker()
    print(f"\n4. Parker Solar Probe perihelion (0.05 au): f = {f_parker:.6f}")

    if do_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not found; skip plots)")
            return
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # φ(r) in solar core
        ax = axes[0, 0]
        r_m = solar["r_m"] / 1e8  # 10^8 m
        ax.plot(r_m, solar["phi"], "b-", label=r"$\varphi(r)$")
        ax.set_xlabel(r"$r$ (10⁸ m)")
        ax.set_ylabel(r"$\varphi$ (m²/s²)")
        ax.set_title(r"$\varphi(r)$ in solar core")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # f(r) in solar core
        ax = axes[0, 1]
        ax.plot(r_m, solar["f"], "g-", label=r"$f(r)$")
        ax.set_xlabel(r"$r$ (10⁸ m)")
        ax.set_ylabel(r"$f$")
        ax.set_title("Lapse f(r) in Sun")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Apparent vs true (wall-clock) redshift
        ax = axes[1, 0]
        z_grid = np.linspace(0.1, 12, 100)
        red = HQIVRedshift(age_wall_Gyr=AGE_WALL_GYR_PAPER)
        age_wall = red.wall_clock_age_at_emission(z_grid)
        age_app_approx = 13.8 * (1 - 1 / np.sqrt(1 + z_grid))  # rough
        ax.plot(z_grid, age_wall, "b-", label="Wall-clock age at emission (Gyr)")
        ax.plot(z_grid, age_app_approx, "k--", alpha=0.7, label="Apparent age (approx)")
        ax.set_xlabel("z (apparent)")
        ax.set_ylabel("Age (Gyr)")
        ax.set_title("Apparent vs wall-clock age at emission")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Orbital τ vs t
        ax = axes[1, 1]
        t_s, tau_s = orb_out["t_s"], orb_out["tau_s"]
        if tau_s is not None:
            ax.plot(t_s / 86400, tau_s / 86400, "b-", label=r"$\tau$ (proper)")
            ax.plot(t_s / 86400, t_s / 86400, "k--", alpha=0.5, label="t (coord)")
            ax.set_xlabel("t (days)")
            ax.set_ylabel("Time (days)")
            ax.set_title("Earth orbit: proper time τ vs coordinate t")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = Path(__file__).resolve().parent / "hqiv_space_model_figures"
        out_dir.mkdir(exist_ok=True)
        fig.savefig(out_dir / "hqiv_space_model.png", dpi=150)
        plt.close()
        print(f"\nPlots saved to {out_dir}/hqiv_space_model.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="Generate matplotlib figures")
    args = ap.parse_args()
    main(do_plot=args.plot)
