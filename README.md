# pyhqiv — Horizon-Quantized Informational Vacuum (HQIV)

[![PyPI version](https://badge.fury.io/py/pyhqiv.svg)](https://badge.fury.io/py/pyhqiv)
[![CI](https://github.com/disregardfiat/pyhqiv/actions/workflows/ci.yml/badge.svg)](https://github.com/disregardfiat/pyhqiv/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18794889.svg)](https://doi.org/10.5281/zenodo.18794889)

**Why HQIV?** HQIV unifies causal-horizon monogamy with discrete null-lattice combinatorics to predict curvature (Ω_k), CMB-consistent ages, and phase-horizon corrections to Maxwell/fluids/molecules. See the [paper](https://doi.org/10.5281/zenodo.18794889) for the full framework.

Production-ready, pip-installable Python package implementing the **Horizon-Quantized Informational Vacuum (HQIV)** framework exactly as defined in the paper:

> **Ettinger, Steven Jr**, *Horizon-Quantized Informational Vacuum (HQIV): A Unified Framework from Causal Horizon Monogamy and Discrete Null-Lattice Combinatorics*. Zenodo, 2026. [https://doi.org/10.5281/zenodo.18794889](https://doi.org/10.5281/zenodo.18794889)

## Citation

If you use this package in research, please cite the paper. On GitHub you can use the **Cite this repository** button (from `CITATION.cff` in the repo root):

```bibtex
@misc{ettinger2026hqiv,
  author       = {Ettinger, Steven Jr},
  title        = {Horizon-Quantized Informational Vacuum (HQIV): A Unified Framework from Causal Horizon Monogamy and Discrete Null-Lattice Combinatorics},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18794889},
  url          = {https://doi.org/10.5281/zenodo.18794889}
}
```

## Installation

```bash
pip install pyhqiv
```

From source:

```bash
git clone https://github.com/disregardfiat/pyhqiv.git && cd pyhqiv
pip install -e .
```

Also on [TestPyPI](https://test.pypi.org/project/pyhqiv/) for pre-release testing:  
`pip install --index-url https://test.pypi.org/simple/ pyhqiv`

| Extra | Purpose |
|-------|---------|
| `ase` | ASE for structure/relaxation (protein/crystal) |
| `mda` | MDAnalysis for trajectories (protein demo) |
| `qutip` | QuTiP for quantum optics |
| `jax` | JAX for JIT-accelerated lattice integrals / evolve_to_cmb |
| `pyvista` | PyVista for 3D visualization |
| `all` | All of the above |

```bash
pip install pyhqiv[ase,mda,qutip,jax,pyvista]
# or
pip install pyhqiv[all]
```

## Quick start

**Cosmology (one call):** Evolve to the CMB and get Ω_k^true, wall-clock/apparent ages, and lapse:  
`result = HQIVCosmology().evolve_to_cmb(T0_K=2.725)` → Ω_k ≈ 0.0098, age 51.2 / 13.8 Gyr, lapse ≈ 3.96.

```python
from pyhqiv import DiscreteNullLattice, HQIVSystem, HQIVCosmology
import numpy as np

# Cosmology
result = HQIVCosmology().evolve_to_cmb(T0_K=2.725)  # Omega_true_k ≈ 0.0098

# Fields
lattice = DiscreteNullLattice(m_trans=500, gamma=0.40)
sys = HQIVSystem.from_atoms([(0, 0, 0), (1.5, 0, 0)], charges=[1, -1], gamma=0.40)
grid = np.mgrid[-2:2:11j, -2:2:11j, -2:2:11j].reshape(3, -1).T
E, B = sys.compute_fields(grid, t=0.0)
```

## API examples (every module)

```python
import numpy as np
from pyhqiv import (
    # Constants (paper values)
    GAMMA, ALPHA, T_PL_GEV, T_LOCK_GEV, T_CMB_K, M_TRANS,
    COMBINATORIAL_INVARIANT, OMEGA_TRUE_K_PAPER, LAPSE_COMPRESSION_PAPER,
    HBAR_C_EV_ANG, A_LOC_ANG,
    # Lattice & cosmology
    DiscreteNullLattice,
    HQIVCosmology,
    # Phase lift
    HQIVPhaseLift,
    # Algebra (so(8), hypercharge)
    OctonionHQIVAlgebra,
    # Atom & system
    HQIVAtom, HQIVSystem,
    # Fluid (modified NS)
    f_inertia, g_vac_vector, eddy_viscosity, modified_momentum_rhs,
    # Molecular (PROtien)
    molecular,
    # Waveguide
    waveguide,
    # Fields
    PhaseHorizonFDTD,
    # Crystal & response
    HQIVCrystal, hqiv_potential_shift, compute_conductivity, response_tensor_diagonal,
    # Thermodynamics (phase diagrams, EOS, no reference data)
    HQIVThermoSystem, compute_free_energy, HQIVHydrogen, PhaseDiagramGenerator,
    hqiv_answer_thermo, plot_phase_diagram_standard_vs_hqiv, TESTABLE_PREDICTIONS,
)

# --- constants ---
print(GAMMA, COMBINATORIAL_INVARIANT, HBAR_C_EV_ANG)

# --- algebra: so(8) closure, hypercharge 4×4 block ---
alg = OctonionHQIVAlgebra(verbose=False)
dim, _ = alg.lie_closure_dimension()  # 28
data = alg.hypercharge_paper_data()

# --- lattice & cosmology: δE(m), evolve_to_cmb, Ω_k, ages, lapse ---
lattice = DiscreteNullLattice(m_trans=500, gamma=0.40)
result = lattice.evolve_to_cmb(T0_K=2.725)
delta_E = lattice.get_delta_E_grid()
cosmo = HQIVCosmology()
cosmo_result = cosmo.evolve_to_cmb(T0_K=2.725)  # Omega_true_k, 51.2/13.8 Gyr, lapse ≈ 3.96

# --- phase: δθ′(E′), ˙δθ′, lapse ---
phase = HQIVPhaseLift(gamma=0.40)
dtheta = phase.delta_theta_prime(0.5)
dot_dtheta = phase.delta_theta_prime_dot(H_homogeneous=1e-18)

# --- atom: local Θ, φ = 2c²/Θ ---
atom = HQIVAtom(position=(0, 0, 0), charge=1)
phi = atom.phi_local(np.array([[1.0, 0, 0]]))

# --- system: multi-atom, E/B on grid ---
sys = HQIVSystem.from_atoms([(0, 0, 0), (1, 0, 0)], charges=[1, -1])
E, B = sys.compute_fields(np.array([[0.5, 0, 0]]), t=0.0)

# --- fluid: f(a,φ), g_vac, ν_eddy ---
f = f_inertia(0.1, 1.0)
g_vac = g_vac_vector(1.0, 0.5, np.ones(3), np.zeros(3))
nu = eddy_viscosity(Theta_local=1.0, dot_delta_theta=1e-18, l_coh=1e-3, coherence_factor=0.5)

# --- molecular: Θ(Z,coord), bond length, damping ---
theta_C = molecular.theta_local(6, 2)  # ≈ 1.53 Å
r_eq = molecular.bond_length_from_theta(1.53, 1.33)
mag = molecular.damping_force_magnitude(1.0, 0.5, a_loc=1.0)

# --- waveguide: k_c², radius, taper, mode solver ---
from pyhqiv.waveguide import kc_squared_hqiv, waveguide_radius_constant_phi, hqiv_waveguide_mode_solver
kc2 = kc_squared_hqiv(omega=2*np.pi*1e9, beta=10.0, m_phase=1, dot_delta_theta=1e-18)
a = waveguide_radius_constant_phi(phi_target=1e10)
gx, gy = np.mgrid[0:1:5j, 0:1:5j]
evals, evecs, mask = hqiv_waveguide_mode_solver(gx, gy, 2*np.pi*1e9, 0.0, n_modes=2)

# --- fields: FDTD ---
fdtd = PhaseHorizonFDTD(shape=(10, 10, 10), dx=0.1, dt=0.05)
fdtd.step()

# --- crystal: PBC, Bloch sum ---
from pyhqiv.atom import HQIVAtom
atoms = [HQIVAtom([0, 0, 0], 0), HQIVAtom([0.5, 0, 0], 0)]
crystal = HQIVCrystal(atoms, lattice_vectors=np.eye(3), supercell_shape=(2, 1, 1))
bloch = crystal.bloch_sum(k_point=[0, 0, 0])
pos_sc = crystal.supercell_positions()

# --- response: conductivity ---
sigma = compute_conductivity(omega=1e10, sigma_0=1.0, phi_avg=1e5)
tensor = response_tensor_diagonal(omega=1e10, dim=3, sigma_0=1.0)

# --- band-gap: potential shift for PySCF ---
V_shift = hqiv_potential_shift(phi_avg=1e-10, dot_delta_theta_avg=1e-18)
# Use V_shift in pyscf.pbc as effective potential shift
```

## Package layout

| Path | Description |
|------|-------------|
| `src/pyhqiv/algebra.py` | Octonion HQIV algebra (so(8) closure, hypercharge 4×4 block) |
| `src/pyhqiv/lattice.py` | Discrete null lattice, δE(m), T(m), evolve_to_cmb |
| `src/pyhqiv/phase.py` | HQIVPhaseLift: δθ′(E′), ˙δθ′, ADM lapse compression |
| `src/pyhqiv/atom.py` | HQIVAtom (position, charge, species, local Θ, φ) |
| `src/pyhqiv/system.py` | HQIVSystem (multi-atom, monogamy γ, E/B on grid) |
| `src/pyhqiv/fields.py` | Phase-horizon FDTD / spectral Maxwell (γ(φ/c²)(˙δθ′/c) terms) |
| `src/pyhqiv/fluid.py` | Modified Navier–Stokes: f_inertia, g_vac, ν_eddy (laminar → standard NS) |
| `src/pyhqiv/thermo.py` | First-principles thermodynamics: phase diagrams, EOS, critical points (no DAC/reference data) |
| `src/pyhqiv/perturbations.py` | Unified linear perturbations with lapse/φ: stellar oscillations, fluid stability, phonons, cosmology |
| `src/pyhqiv/waveguide.py` | HQIV waveguide: k_c²(ω,β,m), constant-φ circle, taper, hyperbolic, mode solver |
| `src/pyhqiv/molecular.py` | PROtien: Θ(Z, coord), bond_length_from_theta, damping_force_magnitude |
| `src/pyhqiv/crystal.py` | HQIVCrystal: PBC, supercell, bloch_sum, reciprocal_vectors; high_symmetry_k_path; hqiv_potential_shift |
| `src/pyhqiv/response.py` | compute_conductivity, response_tensor_diagonal (phase-horizon corrected) |
| `src/pyhqiv/ase_interface.py` | HQIVCalculator (ASE: energy, forces, stress); hqiv_energy_at_positions, hqiv_forces_analytic, hqiv_stress_virial |
| `src/pyhqiv/semiconductors.py` | compute_band_gap, dos, effective_mass, compute_conductivity_tensor, dielectric_function_epsilon |
| `src/pyhqiv/defects.py` | formation_energy (HQIV vacuum correction), charged_defect_supercell |
| `src/pyhqiv/export.py` | export_charge_density_vesta, export_charge_density_ovito; pyscf_hqiv_shift |
| `src/pyhqiv/constants.py` | Paper constants (γ, α, T_Pl, 6^7√3, HBAR_C_EV_ANG, A_LOC_ANG, etc.) |

## Paper numbers (reproduced)

| Quantity | Value | Source |
|----------|--------|--------|
| Ω_k^true | +0.0098 | Shell integral m = 0 … 500 |
| m_trans | 500 | Discrete–continuous transition |
| γ | 0.40 | Entanglement monogamy |
| α | 0.60 | G_eff exponent |
| T_lock | 1.8 GeV | QCD lock-in |
| 6^7√3 | ≈ 4.849×10^5 | Combinatorial invariant |
| Wall-clock age | 51.2 Gyr | Lattice → CMB |
| Apparent age | 13.8 Gyr | ADM lapse compression ≈ 3.96× |

## Tests

```bash
pip install -e ".[all]"
pytest tests/ -v
```

With coverage (optional):

```bash
pip install pytest-cov
pytest tests/ -v --cov=pyhqiv --cov-report=term-missing --cov-report=html
# open htmlcov/index.html
```

CI runs pytest with `--cov=pyhqiv --cov-report=term-missing --cov-report=html` and uploads the HTML report as an artifact (7-day retention). Config: `.coveragerc`. To add a coverage badge, integrate [Codecov](https://codecov.io) or [Coveralls](https://coveralls.io) and add their badge to this README.

The test `tests/test_paper_numbers.py` checks Ω_true_k, γ, combinatorial invariant, lapse factor, and lattice δE(m) / mode counts to 6 decimal places. Additional tests cover ASE calculator (energy/forces/stress), crystal (PBC, k-path), fluid, semiconductors (band_gap, DOS, effective_mass, dielectric), defects, export, and thermo (EOS, phase diagram, hqiv_answer_thermo).

## Reproducibility

From the repo root (or with `pyhqiv` on PYTHONPATH):

```bash
python examples/reproduce_paper.py           # all paper table values
python examples/reproduce_paper.py --plot   # tables + figures (requires matplotlib)
python examples/reproduce_paper.py --plot --pyvista  # add 3D figure (requires pyvista)
```

Figures are written to `examples/reproduce_paper_figures/`. For a thin HQIV→folding minimizer shim (ASE energy/forces from `HQIVSystem`), see `examples/folding_shim_example.py`. Thermodynamics examples (phase diagrams, no reference data): `examples/thermo_metallic_hydrogen_phase_diagram.py`, `examples/thermo_silicon_melting.py`, `examples/thermo_argon_critical.py`, `examples/thermo_answer_any_question.py`, `examples/thermo_ase_phase_stability.py`.

## Thermodynamics (first principles, no DAC/reference data)

From the single axiom **E_tot = m c² + ħ c/Δx** with **Δx ≤ Θ_local(ρ, T)** the package derives phase diagrams, equations of state, and critical points without diamond-anvil or empirical databases:

- **HQIVThermoSystem**, **compute_free_energy(P, T, composition, gamma)** — Gibbs free energy with full φ and lapse correction.
- **HQIVEquationOfState**, **HQIVIdealGas**, **HQIVRealGas**, **HQIVHydrogen** — EOS with lapse; metallic H2 transition at ρ ≈ 0.6–1.0 g/cm³ from φ only.
- **PhaseDiagramGenerator** — P–T coexistence via Gibbs minimization (G1 = G2).
- **hqiv_answer_thermo(question)** — One-function pipeline: parse question → build system from axiom → return answer + plot code.
- **thermo_fluid_lapse**, **thermo_crystal_phi**, **thermo_ase_phase_stability** — Hooks with `fluid.py`, `crystal.py`, `ase_interface.py`.
- **TESTABLE_PREDICTIONS**, **plot_phase_diagram_standard_vs_hqiv** — Falsifiable predictions and side-by-side standard vs HQIV plots.

Enables the full "space model": solar core, rocket propellants, high-z stellar evolution from one equation → entire phase diagram.

## Advanced modeling (perturbations)

Unified linear perturbations with full HQIV lapse/φ corrections — stellar oscillations (Kepler/TESS), fluid instabilities, phonon spectra, cosmological density perturbations:

```python
from pyhqiv import HQIVPerturbations, HQIVSolarCore

background = HQIVSolarCore()  # or HQIVSystem, future HQIVStar/HQIVNeutronStar
pert = HQIVPerturbations(background=background)
modes = pert.stellar_oscillations(l=1, n_max=5)
print([m.period for m in modes])  # periods with lapse-compressed frequencies
print(pert.summary())
```

**CMB pipeline (roadmap):** Full universe evolution from recombination (z ≈ 1100) to now and synthetic CMB map (lapse-compressed perturbations, no Boltzmann hierarchy) is designed in `docs/HQIV_CMB_Pipeline.md`. Entry point: `HQIVCMBPipeline`, status: `cmb_pipeline_status()`. Includes peculiar velocities, ISW/Rees–Sciama, and φ-corrected lensing for Planck-comparable power spectra and testable low-ℓ deviations.

## Materials / semiconductors

For theorists in materials and semiconductors, the package provides:

- **Full ASE Calculator** — geometry relaxation with HQIV potential:
  ```python
  from ase.optimize import BFGS
  from pyhqiv import HQIVCalculator
  calc = HQIVCalculator(gamma=0.40)
  atoms.calc = calc
  BFGS(atoms).run()   # get_potential_energy(), get_forces(), get_stress()
  ```
  See `examples/relax_with_hqiv.py`.

- **HQIVCrystal** — PBC, supercell, Bloch sum; **high_symmetry_k_path()** for k-path generation (e.g. `"GXWG"`).

- **Semiconductor API** — `compute_band_gap()`, `dos()`, `effective_mass()`, `compute_conductivity_tensor()`, `dielectric_function_epsilon()` with HQIV corrections. See `examples/silicon_bandgap_hqiv.py`.

- **Defect utilities** — `formation_energy()` with HQIV vacuum correction; `charged_defect_supercell()` for charged-defect supercells.

- **Hybrid interfaces** — `hqiv_potential_shift()` (and `pyscf_hqiv_shift()`) for PySCF periodic band structure; `export_charge_density_vesta()` / `export_charge_density_ovito()` for charge-density export with HQIV modulation.

## Extensibility

Custom lattices and phase lifts can implement the public protocols/base classes:

- **NullLatticeProtocol** / **NullLatticeBase** — implement `shell_temperature`, `delta_E`, `mode_count_per_shell`, `omega_k_true`, `evolve_to_cmb`, `get_delta_E_grid`, `get_cumulative_mode_counts`.
- **PhaseLiftProtocol** / **PhaseLiftBase** — implement `delta_theta_prime`, `delta_theta_prime_dot`, `lapse_compression`, `maxwell_lift_coefficient`.

Subclass the base classes for a new lattice or phase model; the built-in `DiscreteNullLattice` and `HQIVPhaseLift` satisfy the protocols by default.

## Pre-commit

```bash
pip install pre-commit && pre-commit install
```

Runs ruff (lint + format), mypy, and generic hooks on commit. Config: `.pre-commit-config.yaml`.

For releases, the CI build runs `scripts/update_citation_cff.py` to set `CITATION.cff` version and `date-released` from the current tag/date. You can run it manually with `--version X.Y.Z --date YYYY-MM-DD` to sync before a release.

## License

MIT.
