"""
1000-neutron mini neutron star: HorizonNetwork + HQIVUniversalSystem.

Runs in <10 s on a laptop. Full 10^57 would use mean-field (see mean_field_mu in horizon_network).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main() -> None:
    from pyhqiv.universal_system import HQIVUniversalSystem
    from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
    from pyhqiv.constants import M_NEUTRON_MEV

    const = get_hqiv_nuclear_constants()
    L = const["LATTICE_BASE_M"]
    rng = np.random.default_rng(42)
    # NS density scale: ~1.4 fm spacing → blob ~ 1.4e-15 m
    n = 1000
    positions = rng.normal(0, 1.4e-15, (n, 3)).astype(float)
    state_8 = np.eye(8)
    particles = [
        {"position": positions[i], "state_matrix": state_8.copy(), "mass_mev": M_NEUTRON_MEV, "type": "neutron"}
        for i in range(n)
    ]
    ns = HQIVUniversalSystem(particles, lattice_base_m=L, expand_to_quarks=False)
    print("Relaxing 1000-neutron mini-NS...")
    ns.relax(steps=80)
    B_per = ns.binding_per_particle()
    print(f"Binding per nucleon: {B_per:.4f} MeV  (expect ~1.8–2.5 at NS density)")
    try:
        giant = set(range(n))
        mu = ns.net._mu_for_indices(giant)
        print(f"μ (giant component): {mu:.4f}")
    except Exception as e:
        print(f"μ not computed: {e}")

    # Mean-field μ for comparison (zero cost for 10^57)
    from pyhqiv.horizon_network import mean_field_mu
    r_n = 1.2e-15  # m
    density_fm3 = n / (4.0 / 3.0 * np.pi * (5e-15) ** 3) * 1e-45  # rough: volume in fm^3
    density_fm3 = 0.5  # typical NS core
    mf_mu = mean_field_mu(density_fm3, r_n)
    print(f"Mean-field μ (density={density_fm3} fm⁻³): {mf_mu:.4f}")


if __name__ == "__main__":
    main()
