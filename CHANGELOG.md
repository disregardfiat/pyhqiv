# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.0] - 2026-03-01

### Added

- **HQIV Integrated Space Model**
  - **HQIVCosmology** (`cosmology.py`): Single-entry API for lattice evolution to CMB — Ω_k^true, wall-clock/apparent ages (51.2 / 13.8 Gyr), lapse compression (≈3.96).
  - **HQIVRedshift** (`redshift.py`): Full redshift decomposition — z_expansion, z_gravitational, z_HQIV_mass_lapse, z_total; wall-clock age at emission; lapse degeneracy via `decompose_from_apparent()`.
  - **HQIVSolarCore** (`solar_core.py`): φ(r) profile from center to surface, lapse f(r), standard vs HQIV table (T_c, ρ_c, luminosity shift, age); `phi_solar_radial_profile()` for plots.
  - **HQIVOrbit** (`orbit.py`): Lapse-corrected orbital mechanics — φ(r), f(r), proper-time rate dτ/dt = f, clock desync; Earth–Sun and Parker Solar Probe examples; `parker_perihelion_lapse()`.
- **Documentation**
  - `docs/HQIV_Integrated_Space_Model.md`: Full seven-section Space Model (architecture, solar core, redshift decomposition, orbits, code, visualizations, testable predictions).
  - README Quick start: cosmology one-liner with `HQIVCosmology().evolve_to_cmb()` for balance with materials examples.
- **Examples**
  - `examples/hqiv_space_model_full.py`: Sun + Earth orbit + high-z galaxy; optional `--plot` for φ(r), f(r), z vs age, τ vs t.

### Changed

- Package now serves both **materials** (CIF → HQIV relaxation → band gap / mobility in &lt;20 lines) and **cosmology/astrophysics** (orbits with lapse-corrected clocks, full redshift decomposition, solar-core φ(r), JWST/rocket-science ready).
- `fluid.py` and `waveguide.py`: Module docstrings expanded with one paragraph linking back to the single-source axiom (E_tot = m c² + ħ c/Δx, φ = 2c²/Θ_local, f = a_loc/(a_loc + φ/6)).

### Fixed

- (None in this release.)

[0.4.0]: https://github.com/disregardfiat/pyhqiv/releases/tag/v0.4.0
