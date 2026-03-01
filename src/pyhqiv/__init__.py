"""
pyhqiv: Python implementation of the Horizon-Quantized Informational Vacuum framework.

If you use this package in research, please cite:
https://doi.org/10.5281/zenodo.18794889
"""

__citation__ = """
@misc{ettinger2026hqiv,
    title        = {Horizon-Quantized Informational Vacuum (HQIV) Framework},
    author       = {Steven Ettinger},
    year         = 2026,
    doi          = {10.5281/zenodo.18794889},
    url          = {https://doi.org/10.5281/zenodo.18794889}
}
"""

__doc__ = """
pyhqiv: Python implementation of the Horizon-Quantized Informational Vacuum framework.

If you use this package in research, please cite:
https://doi.org/10.5281/zenodo.18794889
"""

from pyhqiv.constants import (
    A_LOC_ANG,
    GAMMA,
    ALPHA,
    HBAR_C_EV_ANG,
    T_PL_GEV,
    T_LOCK_GEV,
    T_CMB_K,
    M_TRANS,
    COMBINATORIAL_INVARIANT,
    OMEGA_TRUE_K_PAPER,
    LAPSE_COMPRESSION_PAPER,
    AGE_WALL_GYR_PAPER,
    AGE_APPARENT_GYR_PAPER,
)
from pyhqiv.algebra import OctonionHQIVAlgebra
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.cosmology import HQIVCosmology
from pyhqiv.phase import HQIVPhaseLift
from pyhqiv.atom import HQIVAtom
from pyhqiv.system import HQIVSystem
from pyhqiv.fields import PhaseHorizonFDTD
from pyhqiv.fluid import eddy_viscosity, f_inertia, g_vac_vector, modified_momentum_rhs
from pyhqiv import molecular
from pyhqiv import waveguide
from pyhqiv.crystal import HQIVCrystal, hqiv_potential_shift, high_symmetry_k_path
from pyhqiv import semiconductors
from pyhqiv.semiconductors import (
    compute_band_gap,
    dos,
    effective_mass,
    compute_conductivity_tensor,
    dielectric_function_epsilon,
)
from pyhqiv import defects
from pyhqiv.defects import formation_energy, charged_defect_supercell
from pyhqiv.export import (
    export_charge_density_vesta,
    export_charge_density_ovito,
    pyscf_hqiv_shift,
)
from pyhqiv.response import compute_conductivity, response_tensor_diagonal
from pyhqiv.solar_core import HQIVSolarCore, phi_solar_radial_profile
from pyhqiv.redshift import HQIVRedshift, z_total_apparent, z_expansion_from_scale_factor
from pyhqiv.orbit import HQIVOrbit, parker_perihelion_lapse
from pyhqiv.thermo import (
    HQIVThermoSystem,
    HQIVEquationOfState,
    HQIVIdealGas,
    HQIVRealGas,
    HQIVHydrogen,
    PhaseDiagramGenerator,
    compute_free_energy,
    hqiv_answer_thermo,
    phi_from_rho_T,
    theta_local_from_density,
    shell_fraction_energy_shift,
    lapse_compression_thermo,
    thermo_fluid_lapse,
    thermo_crystal_phi,
    thermo_ase_phase_stability,
    TESTABLE_PREDICTIONS,
    plot_phase_diagram_standard_vs_hqiv,
)
from pyhqiv.ase_interface import (
    HQIVCalculator,
    hqiv_energy_at_positions,
    hqiv_forces_analytic,
    hqiv_stress_virial,
)
from pyhqiv.protocols import (
    NullLatticeProtocol,
    NullLatticeBase,
    PhaseLiftProtocol,
    PhaseLiftBase,
)

__all__ = [
    "A_LOC_ANG",
    "ALPHA",
    "GAMMA",
    "HBAR_C_EV_ANG",
    "molecular",
    "T_PL_GEV",
    "T_LOCK_GEV",
    "T_CMB_K",
    "M_TRANS",
    "COMBINATORIAL_INVARIANT",
    "OMEGA_TRUE_K_PAPER",
    "LAPSE_COMPRESSION_PAPER",
    "AGE_WALL_GYR_PAPER",
    "AGE_APPARENT_GYR_PAPER",
    "OctonionHQIVAlgebra",
    "DiscreteNullLattice",
    "HQIVCosmology",
    "HQIVPhaseLift",
    "HQIVAtom",
    "HQIVSystem",
    "PhaseHorizonFDTD",
    "f_inertia",
    "g_vac_vector",
    "eddy_viscosity",
    "modified_momentum_rhs",
    "waveguide",
    "HQIVCrystal",
    "hqiv_potential_shift",
    "high_symmetry_k_path",
    "semiconductors",
    "compute_band_gap",
    "dos",
    "effective_mass",
    "compute_conductivity_tensor",
    "dielectric_function_epsilon",
    "defects",
    "formation_energy",
    "charged_defect_supercell",
    "export_charge_density_vesta",
    "export_charge_density_ovito",
    "pyscf_hqiv_shift",
    "compute_conductivity",
    "response_tensor_diagonal",
    "HQIVCalculator",
    "hqiv_energy_at_positions",
    "hqiv_forces_analytic",
    "hqiv_stress_virial",
    "NullLatticeProtocol",
    "NullLatticeBase",
    "PhaseLiftProtocol",
    "PhaseLiftBase",
    "HQIVSolarCore",
    "phi_solar_radial_profile",
    "HQIVRedshift",
    "z_total_apparent",
    "z_expansion_from_scale_factor",
    "HQIVOrbit",
    "parker_perihelion_lapse",
    "HQIVThermoSystem",
    "HQIVEquationOfState",
    "HQIVIdealGas",
    "HQIVRealGas",
    "HQIVHydrogen",
    "PhaseDiagramGenerator",
    "compute_free_energy",
    "hqiv_answer_thermo",
    "phi_from_rho_T",
    "theta_local_from_density",
    "shell_fraction_energy_shift",
    "lapse_compression_thermo",
    "thermo_fluid_lapse",
    "thermo_crystal_phi",
    "thermo_ase_phase_stability",
    "TESTABLE_PREDICTIONS",
    "plot_phase_diagram_standard_vs_hqiv",
]

try:
    from pyhqiv._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
