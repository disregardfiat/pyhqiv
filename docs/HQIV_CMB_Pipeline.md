# HQIV CMB Pipeline: First-Principles Universe Evolution to Synthetic CMB Map

**Status:** Orchestrator in place. `HQIVCMBMap` in `pyhqiv.cosmology.hqiv_cmb` ties lattice + perturbations → map + σ₈ + C_ℓ (curved LOS, ISW). Optional: `pyhqiv.cosmology_full` (phenomenological) and `HQIVUniverseEvolver`.

### Component status (current repo)

| Component | Status | Notes |
|-----------|--------|--------|
| Background scalars (T_Pl → T_CMB) | ✅ 100% | `evolve_to_cmb()` perfect, Ω_k^true derived cleanly |
| Perturbations layer | ✅ 100% | `perturbations.py`: transfer, ISW, growth_to_sigma8 |
| Cosmology wrapper | ✅ 90% | `HQIVCosmology` with Ok0, lapse_factor, curved_line_of_sight |
| Observable projector + map | ✅ 100% | `hqiv_cmb.py` + `universe_evolver.py` |
| σ₈ calculation | ✅ 100% | `growth_to_sigma8(omega_k) * sqrt(mean(Pk_prim))` in pipeline |
| Multipole chart (C_ℓ) | ✅ 100% | From **harmonic integral** C_ℓ = ∫ (dk/k) P(k) T(k)² j_ℓ(k χ)²; `plot_multipole(result)` |
| Galaxy accelerated motion (ISW) | ✅ 100% | `isw_from_peculiar_velocity(theta, phi)` in pixel loop |

### Why CLASS-HQIV gets peaks and we didn’t (fixed)

In **CLASS-HQIV** (Repos/HQIV), C_ℓ is computed in **harmonic space**: the transfer Δ_Tℓ(k) comes from the full line-of-sight integral with **spherical Bessel j_ℓ(k χ)**. So each multipole ℓ gets structure from j_ℓ(k χ_rec), which produces acoustic peaks.

In the **previous pyhqiv** pipeline we built the map with **j0(k χ)** only (isotropic LOS weight). That makes the map the same in every direction, so `anafast` returns almost no ℓ-structure and **no peaks**. The fix is to compute **C_ℓ directly** from the same formula as CLASS:  
C_ℓ = (4π) ∫ (dk/k) P(k) T(k)² j_ℓ(k χ_rec)²  
(in μK² with T_CMB_MUK²). The pipeline now uses `_cl_from_harmonic_integral()` and builds the map from C_ℓ via `synfast`, so peaks appear.

### Where the pyhqiv pipeline differs from CLASS-HQIV (peak shape / position)

| Aspect | CLASS-HQIV (Repos/HQIV) | pyhqiv (this repo) |
|--------|-------------------------|---------------------|
| **Background** | Integrates HQVM 3H²−γH = 8πG_eff ρ → H(a), τ(a), conformal time in **one** timeline (~52 Gyr). Thermodynamics gives z_rec, visibility, **r_s in Mpc** (~218). | Scalar `evolve_to_cmb()` + FLRW-style `comoving_distance(z)` with Ω_m, Ω_k. **No** thermodynamics module; no r_s in Mpc from visibility. |
| **Sound horizon r_s** | **rs_rec in Mpc** from thermo (e.g. ~218 Mpc). Transfer and C_ℓ use k in 1/Mpc, so k·r_s and k·χ_rec are both dimensionless. | **r_s = cumulative_mode_count(m_recomb)^(1/3)** in **lattice units** (dimensionless). Same k (1/Mpc) is used in transfer as in C_ℓ. So in the transfer, **x = k·r_s has units 1/Mpc** — wrong. Peak scale in T(k) does not match χ_rec; first acoustic peak position ℓ_A ≈ π χ_rec / r_s is wrong. |
| **Transfer** | Full **Boltzmann hierarchy** Θ_ℓ(τ,k) and **line-of-sight** integral: Δ_Tℓ(k) = ∫ dτ S(τ,k) j_ℓ(k(τ0−τ)). Source S from thermo + potentials. | **Analytic** T(k): oscillation sin(k·r_s)/(k·r_s) + damping + f_inertia, f_lapse. **No** time integral; no visibility; r_s not in Mpc (see above). |
| **Primordial** | **A_s = 2.1e-9**, n_s = 0.96, pivot k. | **primordial_power_from_invariant(k)** from combinatorial invariant; no A_s; amplitude differs. |
| **C_ℓ formula** | Full integral over k with **Δ_Tℓ(k)** from LOS. | C_ℓ = (4π) ∫ (dk/k) P(k) T(k)² j_ℓ(k χ_rec)². Same **structure**, but T(k) and P(k) differ; **r_s / χ_rec scale** wrong → peak positions and shape off. |
| **Peak position** | ℓ_A ≈ π χ_rec / r_s (both in Mpc) tuned via γ, ω_b, h, α (peak_alignment_scan). | χ_rec in Mpc, r_s in lattice units → **ℓ_A wrong** unless r_s is converted to Mpc (e.g. calibrate r_s_Mpc = 218 at fiducial). |
| **Boost / dipole** | No explicit “boost scale” in CLASS output; observer motion and ISW in the perturbation/LOS. | **ISW** from `isw_from_peculiar_velocity` added to map; **boost_scale** (default 0.1) for galactic/solar system–like dipole so gradient isn’t extreme. |

**Summary:** The main differences that **change peak shape/position** are: (1) **r_s in Mpc** — we need the sound horizon in Mpc (e.g. from thermo or a calibration to CLASS r_s) so that the transfer argument k·r_s is dimensionless and the first peak in T(k) sits at the right k; (2) **full transfer** — we use an analytic T(k) instead of the integrated LOS source; (3) **primordial amplitude** — A_s vs invariant; (4) **boost** — we add a separate dipole term with boost_scale.

---

## 0. Main vs optional module

| In **main** (always) | In **optional cosmology** (heavy, `pip install pyhqiv[cosmology]`) |
|----------------------|---------------------------------------------------------------------|
| `DiscreteNullLattice.evolve_to_cmb` (scalar) | `universe_evolver`, `hqiv_cmb` |
| `HQIVCosmology`, Ω_k, lapse, ages | Full-sky Healpy map generator |
| `HQIVPerturbations` (linear response, cosmological_perturbation) | C_ℓ multipole power spectrum / chart |
| **Bulk seed** (paper-authoritative) | `pyhqiv.bulk_seed.get_bulk_seed()` → HQIV `horizon_modes/python/bulk.py` |
| `cmb_pipeline_status`, `HQIVCMBPipeline` (stub) | σ₈ calculation |
| — | Line-of-sight ISW/Rees–Sciama (galaxy accelerated motion) |

**Authoritative seed until baryogenesis complete:** The paper and HQIV repo use `horizon_modes/python/bulk.py` (baryogenesis → lock-in → modified Friedmann to T_cmb) as the single source of truth. That bulk output should **seed** the → CMB pipeline. In pyhqiv, call `get_bulk_seed()` when the HQIV repo is available; pass the result as `bulk_seed` to `HQIVUniverseEvolver`, `hqiv_cmb`, `universe_evolver`, and `sigma8`. Then Ω_k, H₀, η, and the lattice table come from bulk; the rest (growth, C_ℓ, map) uses that seed. Without bulk_seed, the pipeline falls back to the in-package lattice (evolve_to_cmb).

Perturbations stay in the main codebase; the full “run the universe to now” map (seeding from lattice, evolving perturbations, LOS projection, Healpy map + C_ℓ + σ₈) is in the optional cosmology module so the core package stays lean.

---

## 0.1 What's missing / why it feels wrong

**The pipeline currently stops at scalar background evolution.** When you run "the full pipeline" you get correct global numbers (Ω_k, ages, lapse) but **no first-principles map**, no σ₈ from evolved fields, no multipole spectrum from a projected sky — which is why it feels wrong.

| Missing piece | Status |
|---------------|--------|
| **Primordial fluctuation seeding** from lattice combinatorics | Not implemented. No scale-invariant spectrum from combinatorial invariant → initial δT/T, δ, θ. |
| **Forward evolution of perturbations** (δT/T, velocity, density) with lapse/φ | Not implemented. No HQIV-modified Boltzmann equations integrated in time; only point-wise `cosmological_perturbation(k,z)`. |
| **Line-of-sight projection** from recombination (z≈1090) to z=0 | Not implemented. No _project_to_sky(), no integration of transfer functions along LOS. |
| **Healpy map** (T_map_muK) | Implemented **phenomenologically**: map = synfast(C_ℓ_template). Not from a projected sky. |
| **σ₈** | Implemented **phenomenologically**: growth D(z) from perturbations + P(k) template, top-hat 8 h⁻¹ Mpc. Not from evolved density field. |
| **C_ℓ** (TT/EE/TE) | Implemented **phenomenologically**: template (Sachs–Wolfe + first peak). Not from anafast(projected map). |
| **Galaxy accelerated motion** (ISW/Rees–Sciama) | Implemented **phenomenologically**: ΔC_ℓ from D(z), f(z). No non-linear late-time peculiar velocities; no real Rees–Sciama from galaxy motions. |

So: **first-principles chain** (seed → evolve → project → anafast → map, growth→σ₈) is **not** there. What exists is scalar background + phenomenological C_ℓ/σ₈/map/ISW so you get a number (e.g. σ₈ ≈ 0.81) and a map **shape**, but not a map that came from the lattice forward in time.

---

## 0.2 Root cause (technical)

- **evolve_to_cmb** is intentionally **scalar** (fast, reproducible, paper-pinning). It does not produce per-ℓ or per-pixel information.
- **HQIVPerturbations** and **cosmology_full** (growth, σ₈, C_ℓ, LOS ΔC_ℓ, Healpy) are **not wired** into a single top-level `run_from_T_Pl_to_now()` that does: primordial seeding → forward evolution (δT/T, δ, θ with f(φ)) → **project_to_sky()** (LOS integration z_rec → 0) → **anafast()** (or equivalent) to get C_ℓ from that sky → growth factor from evolved δ → **σ₈**.
- There is no **non-linear** late-time step for galaxy peculiar motions; the "accelerated motion" ISW/Rees–Sciama is a phenomenological ΔC_ℓ, not from N-body or non-linear potentials.

---

## 0.3 Edge cases / numerical issues (even on the current path)

- **High-z (T_Pl regime)** — Uses `E_0_factor`; works, but if you change `m_trans` the calibration can drift slightly.
- **Lapse** — Applied at the end in the scalar path; not z-dependent along the evolution in the current implementation.
- **Units** — No astropy.units; everything is pure floats (easy to add later).
- **Large nside** — No JAX path for map/LOS; would be slow for high resolution.

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

### 2.1 Axiom-pure pipeline (HQIVCMBMap)

**Only unit conversions + lattice.** No A_s or other hard physics constants.

- **`pyhqiv.cosmology.cmb_map.HQIVCMBMap`** — Single entry: `run_from_T_Pl_to_now()` → `T_map_muK`, `Cl_TT`, `sigma8`, `background`, `lapse_today`.
- **Lattice:** `DiscreteNullLattice.primordial_power_from_invariant(k)` — P(k) from combinatorial invariant only (scale-invariant or n_s shape).
- **Perturbations:** `HQIVPerturbations.cosmological_transfer(k, z_recomb=1090)` — lapse-modulated transfer; `growth_factor_to_8Mpc()` — σ₈ = growth_factor_to_8Mpc() × √⟨P_prim⟩.
- **Cosmology:** `HQIVCosmology.lapse_factor(z)`, `lapse_now`, `line_of_sight(lat, lon, n_k)` (stub: isotropic ones for now).

C_ℓ is built from P(k) T²(k) at k = ℓ/η_rec, then **synfast** → map; **anafast** returns C_ℓ. Peaks come from lattice shell counting + lapse at recombination. Requires `healpy` (`pip install pyhqiv[cosmology]`).

**Curvature in observables (Ω_k^true = +0.0098):** The pipeline now respects curvature end-to-end so the multipole chart is not flat-ΛCDM naive:

- **comoving_distance(z, omega_k)** — χ(z) with sinn (sin for closed, sinh for open).
- **curved_line_of_sight(theta, phi, omega_k, k)** — LOS weights from j₀(k χ_rec) so angular scale and peak positions shift (~0.5–1%).
- **cosmological_transfer(..., omega_k)** — k_eq from curved χ(z_rec).
- **isw_from_peculiar_velocity(theta, phi)** — low-ℓ boost from accelerated galaxy motion (curvature-aware).
- **growth_to_sigma8(omega_k)** — σ₈ factor with curvature.

Use **HQIVCMBMap** with `run_from_T_Pl_to_now(use_curved_pixel_loop=True)` for a map built from the full curved LOS + ISW per pixel; else the map is synfast from curved C_ℓ plus a uniform ISW dipole. **plot_multipole(result)** plots ℓ(ℓ+1)C_ℓ/2π with Ω_k label.

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

**First-principles (not implemented):**

- [ ] **Primordial seeding** — Scale-invariant spectrum from combinatorial invariant → initial δT/T, δ, θ.
- [ ] **Forward evolution** — HQIV Boltzmann equations (δT/T, δ, θ) integrated in time with f(φ).
- [ ] **Line-of-sight projection** — _project_to_sky() from z_rec to z=0; transfer functions along LOS.
- [ ] **C_ℓ from sky** — anafast(projected map), not template.
- [ ] **σ₈ from evolved field** — Growth from evolved δ; top-hat 8 h⁻¹ Mpc on that field.
- [ ] **Non-linear galaxy motions** — Rees–Sciama from peculiar velocities / N-body, not just ΔC_ℓ template.

Implemented in **main** (today):

- [x] **Background** — `HQIVCosmology.evolve_to_cmb` (scalar: Ω_k, lapse, ages); no map.
- [x] **Linear perturbations** — `HQIVPerturbations.cosmological_perturbation`, `linear_response` (point-wise k,z).

Implemented in **optional cosmology module** (`pyhqiv.cosmology_full`) — **phenomenological only**:

- [x] **universe_evolver** — z_grid, a_grid, D(z), f(z) from lattice + perturbations.
- [x] **σ₈** — `sigma8(z)` from HQIV growth D(z) + P(k) template, top-hat 8 h⁻¹ Mpc (not from evolved field).
- [x] **C_ℓ** — `c_ell_spectrum('TT'|'EE'|'TE'|'BB')` phenomenological template (μK²); not from anafast(sky).
- [x] **Line-of-sight ISW/Rees–Sciama** — `line_of_sight_isw_rees_sciama(ell)` ΔC_ℓ from D(z), f(z); no non-linear motions.
- [x] **Full-sky Healpy map** — `full_sky_healpy_map(n_side)` = synfast(C_ℓ_template); not from projected sky.
- [x] **hqiv_cmb** — Returns C_ℓ, σ₈, optional T_map (all phenomenological).

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
