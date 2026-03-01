"""Tests for thermo: HQIVThermoSystem, compute_free_energy, EOS, PhaseDiagram, hqiv_answer_thermo."""

import numpy as np
import pytest

from pyhqiv.thermo import (
    theta_local_from_density,
    phi_from_rho_T,
    shell_fraction_energy_shift,
    lapse_compression_thermo,
    HQIVThermoSystem,
    compute_free_energy,
    HQIVIdealGas,
    HQIVRealGas,
    HQIVHydrogen,
    PhaseDiagramGenerator,
    hqiv_answer_thermo,
    TESTABLE_PREDICTIONS,
    thermo_fluid_lapse,
    thermo_crystal_phi,
    thermo_ase_phase_stability,
)
from pyhqiv.constants import GAMMA, ALPHA


def test_theta_local_from_density():
    """Θ_local = (M/(ρ N_A))^{1/3} in m."""
    rho = 100.0  # kg/m³
    M = 0.002016  # H2
    theta = theta_local_from_density(rho, M)
    assert theta > 0
    assert np.isfinite(theta)
    # Higher rho => smaller theta
    theta_high = theta_local_from_density(1000.0, M)
    assert theta_high < theta


def test_phi_from_rho_T():
    """φ = 2 c² / Θ_local."""
    phi = phi_from_rho_T(100.0, 0.002016, T_K=300.0)
    assert phi > 0
    assert np.isfinite(phi)


def test_shell_fraction_energy_shift():
    """Shell shift dimensionless, positive, finite."""
    sh = shell_fraction_energy_shift(300.0, alpha=ALPHA)
    assert 0 <= sh <= 1
    assert np.isfinite(sh)
    sh_high_T = shell_fraction_energy_shift(5000.0, alpha=ALPHA)
    assert np.isfinite(sh_high_T)


def test_lapse_compression_thermo():
    """f = a/(a+φ/6) in [f_min, 1]."""
    f = lapse_compression_thermo(1.0, 1.0, gamma=GAMMA)
    assert 0 < f <= 1
    assert abs(f - 1.0 / (1.0 + 1.0 / 6.0)) < 0.01


def test_hqiv_thermo_system_rho_ideal():
    """HQIVThermoSystem.rho_from_P_T_ideal = P M / (R T)."""
    sys = HQIVThermoSystem(1e5, 300.0, "H2", gamma=GAMMA)
    rho = sys.rho_from_P_T_ideal()
    assert rho > 0
    assert np.isfinite(rho)


def test_compute_free_energy_returns_tuple():
    """compute_free_energy returns (G_J, info)."""
    G, info = compute_free_energy(1e5, 300.0, "H2", gamma=GAMMA)
    assert np.isfinite(G)
    assert "phi" in info
    assert "shell_shift" in info
    assert "f_lapse" in info


def test_hqiv_ideal_gas_pressure():
    """Ideal gas P = ρ R T / M."""
    eos = HQIVIdealGas(molar_mass_kg=0.002016)
    rho = 1.0
    T = 300.0
    P = eos.pressure(rho, T)
    R = 8.314462618  # J/(mol·K) ≈ K_B * N_A
    expected = rho * R * T / 0.002016
    assert abs(P - expected) < 1e3
    assert eos.fugacity_or_Z(1e5, 300.0) == 1.0


def test_hqiv_real_gas_pressure():
    """Real gas (vdW) P > 0 at moderate rho."""
    eos = HQIVRealGas(a_Pa_m6_mol2=0.25, b_m3_mol=2.66e-5)
    P = eos.pressure(100.0, 300.0)
    assert P > 0
    assert np.isfinite(P)


def test_hqiv_hydrogen_transition_pressure():
    """Metallic H2 transition pressure ~400 GPa (HQIV prediction)."""
    eos = HQIVHydrogen(gamma=GAMMA)
    P0 = eos.transition_pressure_GPa(0.0)
    P300 = eos.transition_pressure_GPa(300.0)
    assert 200 < P0 < 600
    assert 200 < P300 < 600
    assert np.isfinite(P0) and np.isfinite(P300)


def test_phase_diagram_generator_single_phase_G():
    """PhaseDiagramGenerator.gibbs_per_mole_phase returns finite G."""
    eos = HQIVIdealGas(molar_mass_kg=0.002016)
    gen = PhaseDiagramGenerator(eos)
    G = gen.gibbs_per_mole_phase(1e5, 300.0, eos)
    assert np.isfinite(G)


def test_hqiv_answer_thermo_metallic_hydrogen():
    """hqiv_answer_thermo('metallic hydrogen') returns value in GPa."""
    out = hqiv_answer_thermo("metallic hydrogen transition at 300 K")
    assert "answer" in out
    assert out["value"] is not None
    assert out["unit"] == "GPa"
    assert "plot_code" in out


def test_hqiv_answer_thermo_silicon():
    """hqiv_answer_thermo('silicon melting') returns T_m in K."""
    out = hqiv_answer_thermo("silicon melting at 10 GPa")
    assert "answer" in out
    assert out.get("value") is not None
    assert "K" in out.get("unit", "")


def test_testable_predictions_count():
    """Five falsifiable predictions defined."""
    assert len(TESTABLE_PREDICTIONS) >= 5
    for p in TESTABLE_PREDICTIONS:
        assert "id" in p and "statement" in p and "observable" in p


def test_thermo_fluid_lapse():
    """thermo_fluid_lapse returns same shape as f_inertia."""
    f = thermo_fluid_lapse(1.0, 0.5, 1.0)
    assert np.isfinite(f)
    assert 0 < f <= 1


def test_thermo_crystal_phi():
    """thermo_crystal_phi from volume per atom."""
    phi = thermo_crystal_phi(100.0, 8, molar_mass_kg=0.028086)
    assert phi > 0
    assert np.isfinite(phi)


def test_thermo_ase_phase_stability():
    """thermo_ase_phase_stability returns G (joules)."""
    G = thermo_ase_phase_stability(
        potential_energy_J=-100.0,
        volume_m3=1e-28,
        P_Pa=1e5,
        T_K=300.0,
        n_atoms=8,
        gamma=GAMMA,
    )
    assert np.isfinite(G)
