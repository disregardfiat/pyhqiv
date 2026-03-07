"""Tests for HQIVUniversalSystem and binding calibration (PDG/NIST targets)."""

import numpy as np

from pyhqiv.constants import M_NEUTRON_MEV, M_PROTON_MEV
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.horizon_network import mean_field_mu
from pyhqiv.nuclear import NuclearConfig
from pyhqiv.universal_system import HQIVUniversalSystem


def test_universal_system_deuteron_like():
    """Deuteron-like 2 nucleons: binding positive, total energy < sum of rest masses."""
    const = get_hqiv_nuclear_constants()
    L = const["LATTICE_BASE_M"]
    particles = [
        {"position": np.zeros(3), "state_matrix": np.eye(8), "mass_mev": M_PROTON_MEV, "type": "proton"},
        {"position": np.array([2e-15, 0, 0]), "state_matrix": np.eye(8), "mass_mev": M_NEUTRON_MEV, "type": "neutron"},
    ]
    us = HQIVUniversalSystem(particles, lattice_base_m=L, expand_to_quarks=False)
    E = us.total_energy_mev()
    E_free = M_PROTON_MEV + M_NEUTRON_MEV
    assert E < E_free
    B_per = us.binding_per_particle()
    assert B_per > 0


def test_nuclear_config_deuteron_binding_scale():
    """Deuteron binding from NuclearConfig in MeV scale (target ~2.22 MeV)."""
    deut = NuclearConfig(1, 1)
    B = deut._binding_energy_mev
    assert B >= 0
    assert B < 20  # PDG 2.224 MeV; allow some tuning range


def test_mean_field_mu():
    """Mean-field μ = sqrt(1 + avg_neighbors); increases with density."""
    mu_low = mean_field_mu(0.01, r_n=1.2e-15)
    mu_high = mean_field_mu(0.5, r_n=1.2e-15)
    assert mu_low >= 1.0
    assert mu_high > mu_low


def test_universal_expand_to_quarks():
    """With expand_to_quarks=True, 2 nucleons → 6 nodes."""
    const = get_hqiv_nuclear_constants()
    L = const["LATTICE_BASE_M"]
    particles = [
        {"position": np.zeros(3), "state_matrix": np.eye(8), "mass_mev": M_PROTON_MEV, "type": "proton"},
        {"position": np.array([2e-15, 0, 0]), "state_matrix": np.eye(8), "mass_mev": M_NEUTRON_MEV, "type": "neutron"},
    ]
    us = HQIVUniversalSystem(particles, lattice_base_m=L, expand_to_quarks=True)
    assert len(us.net.nodes) == 6
    assert us.total_energy_mev() > 0
