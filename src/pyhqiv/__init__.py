"""
pyhqiv: Python implementation of the Horizon-Quantized Informational Vacuum framework.

.. warning::
   Experimental status. All features are experimental. APIs and numerical results may change.
   Public contribution and feedback are greatly appreciated.

   The CMB pipeline in particular has known issues (analytic transfer vs full Boltzmann
   hierarchy, phenomenological map vs first-principles projection, peak positions/shape).
   See docs/HQIV_CMB_Pipeline.md for details.

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

from pyhqiv import defects, molecular, semiconductors, waveguide
from pyhqiv.algebra import OctonionHQIVAlgebra
from pyhqiv.energy_field import (
    HQIVEnergyField,
    effective_horizon_from_energy_mev,
    merge_constituents,
)
from pyhqiv.ase_interface import (
    HQIVCalculator,
    hqiv_energy_at_positions,
    hqiv_forces_analytic,
    hqiv_stress_virial,
)
from pyhqiv.atom import HQIVAtom
from pyhqiv.bulk_seed import BULK_SEED_AVAILABLE, get_bulk_seed
from pyhqiv.cmb_pipeline import HQIVCMBPipeline, cmb_pipeline_status
from pyhqiv.constants import (
    A_LOC_ANG,
    AGE_APPARENT_GYR_PAPER,
    AGE_WALL_GYR_PAPER,
    ALPHA,
    COMBINATORIAL_INVARIANT,
    GAMMA,
    HBAR_C_EV_ANG,
    LAPSE_COMPRESSION_PAPER,
    M_TRANS,
    OMEGA_TRUE_K_PAPER,
    T_CMB_K,
    T_LOCK_GEV,
    T_PL_GEV,
)
from pyhqiv.cosmology import HQIVCosmology, HQIVUniverseEvolver
from pyhqiv.crystal import HQIVCrystal, high_symmetry_k_path, hqiv_potential_shift
from pyhqiv.defects import charged_defect_supercell, formation_energy
from pyhqiv.export import (
    export_charge_density_ovito,
    export_charge_density_vesta,
    pyscf_hqiv_shift,
)
from pyhqiv.fields import PhaseHorizonFDTD
from pyhqiv.fluid import eddy_viscosity, f_inertia, g_vac_vector, modified_momentum_rhs
from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.orbit import HQIVOrbit, parker_perihelion_lapse
from pyhqiv.perturbations import HQIVPerturbations, PerturbationMode
from pyhqiv.phase import HQIVPhaseLift
from pyhqiv.protocols import (
    NullLatticeBase,
    NullLatticeProtocol,
    PhaseLiftBase,
    PhaseLiftProtocol,
)
from pyhqiv.redshift import HQIVRedshift, z_expansion_from_scale_factor, z_total_apparent
from pyhqiv.polarization import RedshiftDecomposition, decompose_redshift
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.horizon_network import HorizonNetwork, relax_nucleon_positions, relax_quark_positions
from pyhqiv.subatomic import (
    color_singlet_projector,
    make_proton_from_quark_states,
    nucleon_effective_theta_m,
    nucleon_energies_mev,
    neutron_effective_theta_m,
    neutron_energy_mev,
    proton_effective_theta_m,
    proton_energy_mev,
    quark_binding_angles,
    quark_state_matrix,
)
from pyhqiv.nuclear import (
    Nuclide,
    NuclearConfig,
    nuclide_from_symbol,
    half_life_nuclide_hqiv,
    decay_chain_nuclide_hqiv,
)
from pyhqiv.response import compute_conductivity, response_tensor_diagonal
from pyhqiv.semiconductors import (
    compute_band_gap,
    compute_conductivity_tensor,
    dielectric_function_epsilon,
    dos,
    effective_mass,
)
from pyhqiv.solar_core import HQIVSolarCore, phi_solar_radial_profile
from pyhqiv.system import HQIVSystem
from pyhqiv.thermo import (
    TESTABLE_PREDICTIONS,
    HQIVEquationOfState,
    HQIVHydrogen,
    HQIVIdealGas,
    HQIVRealGas,
    HQIVThermoSystem,
    PhaseDiagramGenerator,
    compute_free_energy,
    hqiv_answer_thermo,
    lapse_compression_thermo,
    phi_from_rho_T,
    plot_phase_diagram_standard_vs_hqiv,
    shell_fraction_energy_shift,
    thermo_ase_phase_stability,
    thermo_crystal_phi,
    thermo_fluid_lapse,
    theta_local_from_density,
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
    "HQIVUniverseEvolver",
    "get_bulk_seed",
    "BULK_SEED_AVAILABLE",
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
    "RedshiftDecomposition",
    "decompose_redshift",
    "get_hqiv_nuclear_constants",
    "HorizonNetwork",
    "relax_nucleon_positions",
    "relax_quark_positions",
    "HQIVEnergyField",
    "effective_horizon_from_energy_mev",
    "merge_constituents",
    "color_singlet_projector",
    "make_proton_from_quark_states",
    "proton_energy_mev",
    "neutron_energy_mev",
    "proton_effective_theta_m",
    "neutron_effective_theta_m",
    "nucleon_energies_mev",
    "nucleon_effective_theta_m",
    "quark_binding_angles",
    "quark_state_matrix",
    "Nuclide",
    "NuclearConfig",
    "nuclide_from_symbol",
    "half_life_nuclide_hqiv",
    "decay_chain_nuclide_hqiv",
    "HQIVOrbit",
    "parker_perihelion_lapse",
    "HQIVPerturbations",
    "PerturbationMode",
    "HQIVCMBPipeline",
    "cmb_pipeline_status",
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
