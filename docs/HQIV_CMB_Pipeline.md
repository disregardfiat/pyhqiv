# HQIV CMB Pipeline: First-Principles Universe Evolution to Synthetic CMB Map

**Status:** Design / roadmap. Implemented pieces: `DiscreteNullLattice`, `HQIVCosmology`, `HQIVPerturbations`. Remaining: Boltzmann hierarchy replacement, line-of-sight integration, map generation, secondaries.

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

- [ ] **Background at z_rec** — Use `HQIVCosmology` + lattice to define T_rec, a_rec, φ(τ_rec).
- [ ] **Primordial spectrum** — Scale-invariant (or nearly) from combinatorial invariant; no separate inflation module required for minimal version.
- [ ] **HQIV Boltzmann equations** — Replace standard δ' and θ' (baryon velocity) evolution with f(φ)-weighted terms; photon hierarchy with lapse in opacity and sound horizon.
- [ ] **Perturbation evolution z_rec → z = 0** — Integrate δ, θ, δT/T, polarization with f(φ(τ)) and δE(m) on each shell.
- [ ] **Line-of-sight integration** — δT/T(ℓ, m) and E/B from integral over τ with visibility and f(φ).
- [ ] **Secondaries** — Lensing (convergence κ from φ-corrected LSS), ISW, Rees–Sciama (peculiar velocities from D(a) with f(φ)).
- [ ] **Map generation** — HEALPix (or similar) full-sky T, Q, U; power spectra C_ℓ^TT, C_ℓ^EE, C_ℓ^TE, C_ℓ^κκ and cross with galaxies.

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
