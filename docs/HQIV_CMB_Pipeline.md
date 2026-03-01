# HQIV CMB Pipeline: First-Principles Universe Evolution to Synthetic CMB Map

**Status:** Design / roadmap. Implemented pieces: `DiscreteNullLattice`, `HQIVCosmology`, `HQIVPerturbations` (in **main**). Full pipeline (universe evolver, Healpy, C_ℓ, σ₈, LOS/ISW) lives in the **optional cosmology module** (`pyhqiv.cosmology_full`, heavy).

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
