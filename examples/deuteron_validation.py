"""
Deuteron calibration: binding target 2.224 MeV, decay-allowed check.

Reuses NuclearConfig + HQIV network path (lattice δE + optional 6-quark for A≤2).
Tune only lattice_base_m in hqiv_scalings.py to hit 2.224 MeV; ³H, ⁴He then predictions.

Usage:
    python examples/deuteron_validation.py

Optional: relax with HQIVCalculator + BFGS (see relax_with_hqiv.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main() -> None:
    from pyhqiv.nuclear import NuclearConfig

    deut = NuclearConfig(1, 1)  # P=1, N=1 → deuteron
    B = deut._binding_energy_mev
    target = 2.224
    print(f"Deuteron binding: {B:.4f} MeV  (target {target} MeV)")
    print(f"Deviation:        {B - target:+.4f} MeV")

    # Decay allowed: free neutron → free proton + e⁻ + ν̄_e (Q = (m_n - m_p - m_e)c² ≈ 0.782 MeV)
    E_n = deut._free_neutron_energy_mev
    E_p = deut._free_proton_energy_mev
    decay_threshold_mev = 0.511  # m_e c²
    decay_allowed = E_n > E_p + decay_threshold_mev
    print(f"Free neutron E:   {E_n:.4f} MeV")
    print(f"Free proton E:    {E_p:.4f} MeV")
    print(f"Decay allowed (n → p + e⁻ + ν̄): {decay_allowed}  (expect True)")

    # Optional: same via HQIVUniversalSystem (deuteron as 2 particles, 6-quark mode)
    from pyhqiv.universal_system import HQIVUniversalSystem
    const = get_hqiv_nuclear_constants()
    L = const["LATTICE_BASE_M"]
    particles = [
        {"position": np.zeros(3), "state_matrix": np.eye(8), "mass_mev": M_PROTON_MEV, "type": "proton"},
        {"position": np.array([2.0e-15, 0.0, 0.0]), "state_matrix": np.eye(8), "mass_mev": M_NEUTRON_MEV, "type": "neutron"},
    ]
    us = HQIVUniversalSystem(particles, lattice_base_m=L, expand_to_quarks=True)
    B_us = us.binding_per_particle() * 2  # total binding for 2 particles
    print(f"UniversalSystem (6-quark): binding ≈ {B_us:.4f} MeV (2× binding_per_particle)")


if __name__ == "__main__":
    main()
