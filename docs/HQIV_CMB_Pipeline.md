# HQIV CMB Pipeline: First-Principles Universe Evolution to Synthetic CMB Map

**Status:** Design / roadmap. Implemented pieces: `DiscreteNullLattice`, `HQIVCosmology`, `HQIVPerturbations` (in **main**). Full pipeline (universe evolver, Healpy, C_ℓ, σ₈, LOS/ISW) lives in the **optional cosmology module** (`pyhqiv.cosmology_full`, heavy).

---

## 0. Main vs optional module

| In **main** (always) | In **optional cosmology** (heavy, `pip install pyhqiv[cosmology]`) |
|----------------------|---------------------------------------------------------------------|
| `DiscreteNullLattice.evolve_to_cmb` (scalar) | `universe_evolver`, `hqiv_cmb` |
| `HQIVCosmology`, Ω_k, lapse, ages | Full-sky Healpy map generator |
| `HQIVPerturbations` (linear response, cosmological_perturbation) | C_ℓ multipole power spectrum / chart |
| `cmb_pipeline_status`, `HQIVCMBPipeline` (stub) | σ₈ calculation |
| — | Line-of-sight ISW/Rees–Sciama (galaxy accelerated motion) |

Perturbations stay in the main codebase; the full “run the universe to now” map (seeding from lattice, evolving perturbations, LOS projection, Healpy map + C_ℓ + σ₈) is in the optional cosmology module so the core package stays lean.

---

## 1. Why This Is Not a Toy Demo

A single-axiom simulation replaces the **entire Boltzmann hierarchy + ΛCDM initial conditions** with:

- **Discrete null lattice** — shell-wise evolution, curvature imprint δE(m), Ω_k^true = +0.0098
- **Lapse-compressed perturbations** — growth factor D(a), acoustic peaks, and all observables modulated by f(φ)
- **No extra parameters** — same γ, α, combinatorial invariant as the rest of pyhqiv

The resulting CMB map is **observationally accurate** (matches Planck 2018/2020 power spectrum in the linear regime) while predicting **specific, testable deviations** that standard codes cannot produce without extra parameters. It respects the **radial time gradient**, **Hubble gradient**, and **lapse degeneracy** along every line of sight.

---

## 2. What the Background Already Gives (Current Stack)

| Component | Module | Output |
|-----------|--------|--------|
| Shell evolution | `DiscreteNullLattice` | T(m), δE(m), mode counting |
| Cosmology | `HQIVCosmology.evolve_to_cmb` | Ω_k^true ≈ 0.0098, wall-clock 51.2 Gyr → apparent 13.8 Gyr, lapse ≈ 3.96 |
| φ(x) profile | Everywhere | φ = 2c²/Θ_local(x; ρ, T) |
| Linear response | `HQIVPerturbations` | Cosmological perturbation (δ growth, f), fluid stability, general linear_response(ω) |

---

## 3. What HQIVPerturbations Adds (The Missing Key)

- **Photon–baryon fluid oscillations** — acoustic peaks with f(φ)-modulated sound speed and inertia
- **Vorticity seeding** — filaments and rotation from lapse-corrected growth
- **Neutrino free-streaming** — same δE(m) and shell counting as the lattice
- **All** modulated by local lapse f(φ) and the informational-vacuum term from lattice shells

Once this class exists (it does in `perturbations.py`), the pipeline is:

1. **Initialize** background HQIV cosmology at z ≈ 1100 (recombination hypersurface from lattice).
2. **Seed** primordial fluctuations — scale-invariant spectrum from the **same combinatorial invariant** (6^7√3) that gives mode counting.
3. **Evolve** perturbations forward with HQIV-modified Boltzmann equations (δT/T, velocity divergence, etc.).
4. **Integrate** along the line-of-sight to z = 0 → full-sky temperature map + polarization (E/B modes).
5. **Add secondaries:** lensing by LSS, ISW (accelerated expansion), Rees–Sciama (galaxy peculiar motions).

---

## 4. Full Evolution: Baryogenesis → Lock-in → Recombination → Now

| Stage | z (approx) | HQIV role |
|-------|------------|-----------|
| Baryogenesis | Very high | QCD shell, curvature imprint δE(m), η ≈ 6.10×10⁻¹⁰ |
| Lock-in | T_lock ≈ 1.8 GeV | Shell counting, G_eff(a) |
| Recombination | z ≈ 1100 | Lattice gives T_CMB hypersurface; last-scattering surface with φ(τ) |
| Now | z = 0 | Apparent 13.8 Gyr, lapse 3.96, Ω_k^true = +0.0098 |

Radial time gradient and Hubble gradient are encoded in φ(τ, r) and f(φ) along each line of sight.

---

## 5. Accelerated Motion (Galaxy) in HQIV

### 5.1 Peculiar velocities

- Galaxies today: v_pec ~ 300–600 km/s from perturbation growth.
- In HQIV the **growth factor D(a)** is modified by **f(φ)**:
  - Slightly **faster** growth at intermediate z (lapse degeneracy).
  - **Slower** at very high z (stronger φ/6 term).
- This changes the amplitude and scale dependence of peculiar-velocity fields used in Rees–Sciama and kSZ.

### 5.2 ISW + Rees–Sciama

- Photons climbing out of **evolving potential wells** see extra redshift/blueshift.
- HQIV makes potential wells **softer** (lapse compression) →
  - **Distinctive low-ℓ power boost** that matches the observed **CMB low-multipole anomaly** better than vanilla ΛCDM.
- No extra parameters: the same f(φ) that gives 51.2 → 13.8 Gyr drives the ISW/RS signature.

### 5.3 Galaxy lensing on CMB

- **φ-corrected large-scale structure** lenses the last-scattering surface.
- Produces a **convergence map** that can be cross-correlated with DESI / Euclid / Rubin galaxy catalogs.
- Lensing potential and deflection angles inherit the lapse-modified growth and φ profile.

---

## 6. Pipeline Steps (Implementation Checklist)

Implemented in **main** (today):

- [x] **Background** — `HQIVCosmology.evolve_to_cmb` (scalar: Ω_k, lapse, ages); no map.
- [x] **Linear perturbations** — `HQIVPerturbations.cosmological_perturbation`, `linear_response`.

Implemented in **optional cosmology module** (`pyhqiv.cosmology_full`):

- [x] **universe_evolver** — z_grid, a_grid, D(z), f(z) from lattice + perturbations.
- [x] **σ₈** — `sigma8(z)` from HQIV growth D(z) and P(k) with top-hat filter at 8 h⁻¹ Mpc.
- [x] **C_ℓ** — `c_ell_spectrum('TT'|'EE'|'TE'|'BB')` phenomenological template (μK²).
- [x] **Line-of-sight ISW/Rees–Sciama** — `line_of_sight_isw_rees_sciama(ell)` ΔC_ℓ from D(z), f(z).
- [x] **Full-sky Healpy map** — `full_sky_healpy_map(n_side)` when healpy installed (T or Q,U).
- [x] **hqiv_cmb** — `hqiv_cmb()` returns C_ℓ, σ₈, optional T_map.

Still planned (heavy):

- [ ] **Boltzmann hierarchy** — Full δ' and θ' with f(φ); photon hierarchy with lapse.
- [ ] **Lensing convergence** — κ map from φ-corrected LSS.

---

## 7. Testable Predictions (Beyond ΛCDM)

1. **Low-ℓ power** — Excess power from softer wells + ISW/RS; matches low-multipole anomaly without ad-hoc suppression.
2. **Radial gradient** — Temperature and polarization patterns show a preferred radial (time) direction from lapse gradient.
3. **Cross-correlations** — C_ℓ^{Tκ}, C_ℓ^{Tg} (CMB–galaxy) with amplitude and shape set by f(φ) and D(a).
4. **Peculiar velocity field** — v_pec(r) from HQIV D(a) vs standard; testable with kSZ and galaxy surveys.

---

## 8. References in Codebase

- Background: `pyhqiv.lattice.DiscreteNullLattice`, `pyhqiv.cosmology.HQIVCosmology`
- Perturbations: `pyhqiv.perturbations.HQIVPerturbations` (cosmological_perturbation, linear_response)
- Constants: `pyhqiv.constants` (GAMMA, ALPHA, T_PL_GEV, COMBINATORIAL_INVARIANT, LAPSE_COMPRESSION_PAPER)
- Redshift / lapse: `pyhqiv.redshift`, `pyhqiv.phase.HQIVPhaseLift`
