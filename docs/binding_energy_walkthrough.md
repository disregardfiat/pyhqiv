# Binding energy calculation walkthrough

## 0. Hierarchical (bottom-up) picture

Binding energy must be computed **hierarchically, bottom to top**:

1. **Sub-nucleon layer**  
   Protons and neutrons are made of sub-atomic constituents (quarks / partons; in HQIV, horizon modes or lattice dof at that scale). Each constituent has a horizon Θ and contributes ħc/Θ to tension energy. **Bottom:** compute energy of these constituents (free vs bound into a single nucleon).

2. **Nucleon layer**  
   **Proton** = bound system of its constituents → E_proton (or effective Θ_proton).  
   **Neutron** = bound system of its constituents → E_neutron (or Θ_neutron).  
   So “free proton” and “free neutron” are already bound states at this layer; their energies come from the layer below.

3. **Nuclear layer**  
   **Nucleus (e.g. He-4)** = bound system of P protons + N neutrons.  
   - E_free = P × E_proton + N × E_neutron (each nucleon with its *nucleon-level* energy).  
   - E_nucleus = energy of the P+N nucleons when bound *in the nucleus* (shared horizons / mode counting at nuclear scale).  
   **Nuclear binding energy**  
   B_nuclear = E_free − E_nucleus  
   (energy released when forming the nucleus from free nucleons).

So:

- **Layer 0 (sub-nucleon):** constituents → bound nucleon (proton/neutron). Implemented in **`pyhqiv.subatomic`**.
- **Layer 1 (nucleon):** E_proton, E_neutron from layer 0 via `subatomic.nucleon_effective_theta_m()`; or from scalings when not using layer 0.
- **Layer 2 (nucleus):** E_nucleus from Θ_i of nucleons in the nucleus; B = E_free − E_nucleus.

Nuclear binding and E_info now use **HorizonNetwork** (overlap graph + composite invariant): see §4.

---

## 1. What the code currently computes (no hierarchy)

**Definition used in code:**  
`B = E_free - E_bound` (MeV)

- **E_free** = energy of P free protons + N free neutrons (each with horizon Θ_free).
- **E_bound** = energy of the bound nucleus (each nucleon with horizon Θ_i from mode counting).

So **B > 0** means the bound state has *lower* energy than free nucleons (fusion releases energy). That is the standard nuclear binding energy convention.

---

## 2. Energy from horizon (HQIV)

From the module docstring: `E_tot = Σ m c² + Σ ħc/Θ_i`. The “tension” part is **ħc/Θ** per degree of freedom.

- **Larger Θ** → smaller 1/Θ → **lower** tension energy.
- **Smaller Θ** → larger 1/Θ → **higher** tension energy.

So for **B = E_free − E_bound** to be positive (bound state lower in energy), we need **E_bound < E_free**, i.e. bound nucleons must have **larger Θ** on average than free nucleons (less confined, lower tension).

---

## 3. Free-nucleon and bound horizons (first principles only)

No new fitted constants. Layer 0 is now **charge-driven**: the hypercharge block in the octonion algebra assigns fractional charges \(Q_u = +2/3\), \(Q_d = -1/3\). Three quark spheres (radii \(r_q \propto \hbar c/(m_q c^2)\)) must touch (monogamy) while minimising electrostatic energy
\[
E_{\rm Coul} = \frac{1}{4\pi\epsilon_0} \sum_{i<j} \frac{Q_i Q_j}{d_{ij}}, \qquad d_{ij} = r_i + r_j.
\]
The same three-force algorithm as the nuclear layer is used: hard-sphere repulsion, soft attraction to touching, and Coulomb between all pairs \(\propto Q_i Q_j/d^2\). Equilibrium gives **binding angles** (VSEPR-like): proton (uud) has u-u repulsion → larger u-u angle (~109° tetrahedral preference); neutron (udd) has d-d attraction → more acute d-d angle.

- **Θ_free** from the composite \(\mu = \sum r_i / \sqrt{\sum r_i^2}\) (radii only): \(\Theta_{\rm free} = L\times 8\times \mu\). So **proton has larger free horizon** than neutron (\(\mu_{uud} > \mu_{udd}\) when \(r_u > r_d\)).
- **E_free** = tension + Coulomb: \(E = \hbar c/\Theta + E_{\rm Coul}\). Neutron ends up **heavier** (\(E_n > E_p\)) because \(E_{\rm Coul}\) differs (u-u repulsion vs d-d attraction).

API: `quark_binding_angles("uud")` / `("udd")` return the three bond angles (rad); `relax_quark_positions(radii, charges)` in `horizon_network` does the relaxation. The 8×8 merge (color singlet) still encodes the same valence; the geometry alone sets Θ and E at layer 0.

At the nuclear layer, **free** proton/neutron Θ and E come from `proton_effective_theta_m()`, `neutron_effective_theta_m()`, `proton_energy_mev()`, `neutron_energy_mev()`. **Bound** Θ and **B** come from `HorizonNetwork`: `E_free = sum(single-nucleon network energies)`, `E_bound = one network of P+N`, `B = E_free − E_bound`.

---

## 4. Bound-state horizon: HorizonNetwork (sphere-touching geometry only)

Binding and bound Θ use **HorizonNetwork** (`pyhqiv.horizon_network`), **no Δ in the algebra**:

- **Radius per node** \(r_i = \hbar c / (m_i c^2)\) from mass (inverse-frequency sphere).
- **Edges:** connect when distance < \(r_{\rm eq}(i,j)\) (min-energy state). \(r_{\rm eq} = \texttt{R\_EQ\_SCALE}\times (r_i + r_j)\) so the graph models the balanced well (~1.4 fm for nucleons), not strict touch.
- **Component coherence** on the full connected component: \(\mu_{\rm comp} = \sum r_i / \sqrt{\sum r_i^2}\), \(\Theta_{\rm comp} = L \times 8 \times \mu_{\rm comp}\). Single node → μ = 1 → Θ = L×8; cluster (e.g. 4 nucleons) → μ > 1 → binding.
- **Node-local valence** from the same overlap graph: for nucleon \(i\), take the subgraph made of node \(i\) and its touching neighbours and compute \(\mu_i\) with the same formula. The per-node bound horizon is the geometric mean
  \[
  \Theta_i = L \times 8 \times \sqrt{\mu_{\rm comp}\mu_i}.
  \]
  This preserves one graph at all scales while allowing distinct local tensions inside asymmetric nuclei.
- **E_free** = sum of single-nucleon network `total_energy()` (each μ = 1); **E_bound** = one network of P+N → μ > 1 → **B = E_free − E_bound** > 0.
- **Geometric nucleon packing:** bound positions minimize total network energy (balanced well). `minimize_nucleon_configuration(radii_m, is_proton, lattice_base_m)` uses `scipy.optimize` with `HorizonNetwork.total_energy()` as objective; initial guess is tetrahedral for A=4 (edges at contact) or `relax_nucleon_positions` for other A. No per-nucleus if-statements: He-4 → tetrahedron, ⁸Be → two alphas, ¹²C → alpha-triangle, etc., from the same potential. Final positions → overlap graph → μ → Θ_eff → B.
- Quark 8×8 matrices are **color (g₂) + flavor scale only**; binding comes purely from the Pythagorean mode multiplier in the network.

No eps_delta, no algebraic Tr(M@Δ) for Θ; only masses → radii → μ. Same construction scales to 238 spheres (U) or residues (proteins).

### 4.1 Effective potential for the balanced well (pure algebra, no new constants)

The total interaction between two horizons is an effective potential already implicit in the network; it can be written explicitly for **any two horizon radii** (universally applicable):

\[
V_{\rm eff}(r) = V_{\rm rep}(r) + V_{\rm attr}(r), \quad
V_{\rm rep}(r) = \frac{A}{r^{12}}, \quad
V_{\rm attr}(r) = -B\,\frac{\phi(r)}{r^6} - C\cdot\operatorname{tr}(M_1 M_2 \Delta)\,e^{-r/\lambda_{\rm coh}}.
\]

- **V_rep:** hard-core Pauli/horizon exclusion; \(A = \hbar c\,(r_1+r_2)^{11}\) from the algebra.
- **V_attr:** van-der-Waals-like horizon-overlap term with \(\phi(r) = 2c^2/\Theta(r)\), \(\Theta(r) = L\times 8\times \mu(r)\); plus composite invariant \(\operatorname{tr}(M_1 M_2 \Delta)\) with coherence length \(\lambda_{\rm coh}\).
- The minimum \(dV_{\rm eff}/dr = 0\) gives equilibrium separation **r_eq ≈ 1.4 fm** for nucleon pairs (alpha-particle rms radius scale). No tuning — A, B, C and \(\lambda_{\rm coh}\) are fixed by ħc, lattice_base, and algebra.

**API (universal for any 2 horizons):**

- **`effective_potential_pair(r_m, r1_m, r2_m, lattice_base_m, ...)`** — returns \(V_{\rm eff}(r)\) in MeV; optional trace term and \(\lambda_{\rm coh}\).
- **`equilibrium_separation_two_horizons(r1_m, r2_m, lattice_base_m, ...)`** — returns r_eq in metres (e.g. ~1.4 fm for two nucleons). Use this to get the appropriate distance for any two horizon radii.
- **`minimize_nucleon_configuration(radii_m, is_proton, lattice_base_m, ...)`** (in `nuclear`) — finds equilibrium positions by minimizing `HorizonNetwork.total_energy()`; reuses the same algebra and φ-coupling.

Layer-0 quark touching can use the same minimizer with quark radii and charges to close the full hierarchical ladder.

---

## 5. Physical binding (B > 0)

- **Physical binding** means the nucleus has *lower* energy than free nucleons → **E_bound < E_free** → **B > 0**.
- That requires **larger** effective Θ in the bound cluster: guaranteed by μ > 1 when the component has N ≥ 2 touching spheres.
- Magnitude of B set by geometry (radii from masses); nucleon masses set node radii in the network, while the free proton/neutron ordering is fixed by the layer-0 8×8 merge.

---

## 6. Summary

| Step | What |
|------|------|
| Node radius | \(r_i = \hbar c / (m_i c^2)\); from mass only. |
| Bound Θ | One graph, two levels: \(\mu_{\rm comp}\) for total binding and \(\Theta_i = L\times 8\times\sqrt{\mu_{\rm comp}\mu_i}\) for local decay tension. |
| E_free, E_bound, B | First principles only; no constants in the math engine. |
| Tests | Geometric path gives B > 0 (~25 MeV He-4); local valence split now distinguishes free neutron, tritium, and He-4. |

---

## 6.1 Full matrix and color vs Coulomb (not yet used)

**Current code does *not* use:**

- A **full matrix of all energy states** (e.g. the 8×8 / so(8) structure from `pyhqiv.algebra.OctonionHQIVAlgebra`).
- **Color force *against* Coulomb force** as two competing terms. At the sub-nucleon scale the strong (color) force binds constituents while the EM (Coulomb) force repels; the bound state is the balance of both.

**What exists:**

- **Subatomic (layer 0):** Charge-driven geometry: `relax_quark_positions(radii, charges)` with \(Q_u = +2/3\), \(Q_d = -1/3\); binding angles from equilibrium; \(\Theta = L\times 8\times \mu\), \(E = \hbar c/\Theta + E_{\rm Coul}\). Color (g₂) in the 8×8 merge; Coulomb sets the angles. No matrix invariants needed for Θ or E.
- **algebra.py:** 8×8 matrices, SU(3)_c, U(1)_Y (hypercharge) — fractional charges are the geometric shadow of the hypercharge block; the same relaxation machinery applies at nucleon and nuclear layers.

**Intended direction:** Sub-nucleon (and optionally nuclear) binding should use the **full state matrix** (algebra / HQVM) with **color (confinement) and Coulomb (repulsion)** both entering, so the eigenvalues or effective Θ come from diagonalizing that competition, not from a single 8 − Coulomb_reduction.

### 6.2 What the ladder now shows for neutron, tritium, and He-4

**Free nucleons:** Layer 0 is **charge-driven**: \(Q_u = +2/3\), \(Q_d = -1/3\), same sphere-touching + Coulomb relaxation as nuclei. From \(\mu = \sum r_i/\sqrt{\sum r_i^2}\) with \(r_u > r_d\) we get **proton Θ_free > neutron Θ_free**. Neutron is **heavier** (\(E_n > E_p\)) from the Coulomb term (u-u repulsion vs d-d attraction). So free-neutron β⁻ and tritium β⁻ have positive Q-value without any decay constant.

**Tritium (H-3):** The nuclear overlap graph has one neutron on a weaker local subgraph → that neutron is the β⁻ source. The “pressure” is the angular mismatch in the quark triangles (acute d-d in the converting neutron).

**Light nuclei (¹H, ²H, ³H):** For A ≤ 3 the code uses a **quark-level** HorizonNetwork (3×A quarks). Per-nucleon bound Θ come from the geometric mean of that nucleon's three quarks' Θ, so **quark interactions drive nucleon attraction and decay**; the single-nucleon picture is only macroscopically accurate. Decay chains (e.g. ³H → ³He) use this quark-driven logic; β⁻ is allowed when N > 0 and ΔE > 0.

**Tritium (H-3):** Quark-level network gives per-nucleon thetas; the neutron(s) allow β⁻ to ³He.

**He-4:** A > 3 so nucleon-level network only; valence-saturated graph → no β± channel; binding remains large and positive.

### 6.2.1 Binding angles and ladder implications

- **Strange quark (s = −1/3):** Same charge as d but heavier → larger radius → wider angles in hyperons (Λ, Σ) from the same relaxation.
- **Mesons (q̄q):** Two spheres → 180° linear “angle” → correct for π⁰, etc.
- **Heavy nuclei:** Angle relaxation inside each nucleon feeds into nuclear packing; no double-counting.
- **Protein folding:** Amino-acid partial charges → same angle rule gives secondary-structure preferences (e.g. α-helix ~100°, β-sheet ~120°) from geometry.
- **Isolated quark:** No partner → no angle → confinement only in triplets.
- **Tetraquarks / pentaquarks:** Four- or five-sphere relaxation → exotic angles from the same machinery.

One rule, one axiom, zero extra constants from Planck scale to biology.

### 6.2.2 Paper dynamics: modified inertia in nuclear decay

The paper’s **modified inertia** \(f(a_{\mathrm{loc}},\phi) = a_{\mathrm{loc}}/(a_{\mathrm{loc}} + \phi/6)\) (particle action \(S = -m c \int f \, ds\)) is applied to nuclear β-snap probability and decay rate:

- **Snap probability**  
  \(P_{\mathrm{snap}} = \exp(-\Delta E / kT_{\mathrm{eff}}) \times \varphi/(\varphi + \varphi_{\mathrm{crit}})\), with  
  \(kT_{\mathrm{eff}} = (\hbar c/\Theta) \times f\).  
  At the nucleus, \(a_{\mathrm{loc}} = c^2/\Theta_{\mathrm{avg}}\), \(\phi = 2c^2/\Theta_{\mathrm{avg}}\), so \(f = 3/4\) for a typical nuclear Θ. The effective thermal energy for barrier crossing is reduced (\(f < 1\)), so the Boltzmann factor is steeper — barrier crossing is harder in observer time.

- **Decay rate**  
  \(\lambda_{\mathrm{obs}} = (P_{\mathrm{snap}}/\tau_{\mathrm{tick}}) \times f / \mathrm{scale}\).  
  Observer-time rate is scaled by \(f\) (same lapse as in \(d\tau = f \, dt\)), so half-lives are longer when \(f < 1\).

**Implementation:** `NuclearConfig._lapse_f()` computes \(f\) from \(\Theta_{\mathrm{avg}}\) of unstable and stable configurations; `snap_probability` uses \(kT_{\mathrm{eff}} = kT_{\mathrm{horizon}} \times f\); `decay_rate_per_s` multiplies the raw rate by \(f\). See `pyhqiv.fluid.f_inertia`.

### 6.2.3 Pure algebraic entanglement (phase-lifted fanoplane fusion)

A **position-free** binding path uses only 8×8 matrices, Δ, and the phase-lifted commutator \([M_1,M_2]_\Delta = M_1\Delta M_2 - M_2\Delta M_1\):

- **Entangled composite:** \(M_{12} = M_1 + M_2 + [M_1,M_2]_\Delta\) (non-separable; algebraic analogue of \(\nabla\phi\times\mathbf{E}\)).
- **Holding distance:** \(\sigma = \|[M_1,M_2]_\Delta\|_F\) (dimensionless; no radii or fm).
- **Phase-lift:** \(M = M_{\mathrm{base}} + \theta\Delta\) with \(\theta = (\pi/2)\arctan(E/E_{\mathrm{Pl,eff}})\) from the axiom; \(E_{\mathrm{Pl,eff}} = E_{\mathrm{Pl}}\times(L_{\mathrm{Pl}}/L)\) so the lattice shell sets the effective Planck scale. Ensures \(\operatorname{tr}(M@\Delta)\neq 0\).
- **Binding:** \(B = E_{\mathrm{free}} - E_{\mathrm{bound}} = -\operatorname{tr}([M_1,M_2]_\Delta \Delta)\). By cyclic trace, \(\operatorname{tr}([M_1,M_2]_\Delta\Delta)=0\) for any \(M_1,M_2\), so this formula yields \(B=0\); a different invariant may be needed for non-zero algebraic binding.

**API:** `build_nucleon_matrix_with_phase(is_proton, lattice_base_m)` — nucleon matrix with axiom-derived θΔ. `pyhqiv.entanglement` — `entangle_particles`, `holding_distance`, `binding_energy_pair`, `iterated_fusion`, `binding_energy_algebraic`. `binding_energy_mev_algebraic(P, N)` returns B from this path. The default binding used by `NuclearConfig` remains the geometry-based HorizonNetwork path.

### 6.3 Universal integral and composition rule (paper-exact)

The paper axiom is local; for any extended object the total energy is the volume integral of the local density:

$$E_{\\rm total} = \\int \\left( \\rho(x) c^2 + \\frac{\\hbar c}{\\Delta x(x)} \\right) d^3x$$

with Δx(x) ≤ Θ_local(x). Θ_local is determined by the local octonion state (8×8 matrix). When systems bind, their 8×8 matrices combine via the algebra (left action, then projection onto invariant subspaces — Spin(8) triality, g₂ color singlets). The effective Θ_local of the composite is an invariant of the projected state (e.g. trace or effective_modes = 8 + trace(state @ Δ)). Horizon monogamy gives mode sharing in overlapping causal diamonds, so **bound states get larger effective Θ_local** (smaller tension term) than free constituents → positive binding energy from the same axiom.

**Implemented:**
- **`HQIVEnergyField.from_atoms(atoms, positions)`** — compose species 8×8 matrices, project to singlet; **`effective_theta_local(lattice_base_m, local_density)`** — Θ from algebraic invariant (8 + trace(M@Δ)).
- **`hqiv_energy_for_angles(phi, psi, atoms=..., positions=...)`** — integrates the axiom over a small volume (matrix path); scalar fallback when atoms is None.
- **Nuclear matrix path (optional):** `NuclearConfig(..., use_matrix_bound_theta=True)` uses **`_bound_theta_from_matrix_composition`**: compose P proton + N neutron 8×8 states (left action), then **effective_modes = 8 + trace(M @ Δ)** and Θ_bound = lattice_base × effective_modes. **No tuning:** if B is wrong, the fix is in the composition (nucleon matrices, projection, or which invariant to use), not a numerical fudge.

### 6.4 Single merge component (subatomic → solar and beyond)

One process, all scales: **`merge_constituents(constituents, project_singlet=..., algebra)`** composes 8×8 states via left action (octonion multiplication) and optionally projects to the invariant (singlet). The **total energy of the system defines its horizon**: E_tot = ∫ (ρ c² + ħc/Δx) d³x over the system, then **Θ_system = ħc / E_tot** via **`effective_horizon_from_energy_mev(E_tot)`**. So we are not assigning horizons to subatomic particles independently; they come from the merged composite and from the integrals.

- **Subatomic:** 3 quarks → `merge_constituents([M1,M2,M3], project_singlet=True)` → nucleon. Same merge.
- **Nuclear:** P protons + N neutrons → `merge_constituents([proton_field, ...], project_singlet=False)` for invariant; bound Θ from effective_modes = 8 + trace(M @ Δ). Same merge.
- **Molecular:** N atoms → `from_atoms(atoms)` → `merge_constituents(species_matrices, project_singlet=True)`. Same merge.
- **Solar and beyond:** Merge regions/cells the same way; total energy of the aggregate defines its effective horizon. No tuning at any scale.

**API:** `merge_constituents(list_of_8x8_or_EnergyField, project_singlet=True)`, `effective_horizon_from_energy_mev(energy_mev)` (Θ in m), `HQIVEnergyField.energy_mev_from_theta_m(theta_m)` (E = ħc/Θ). If horizons are wrong at any scale, the fix is in the inputs to merge (constituent matrices, integrals) or the invariant used to read Θ from the composite, not scale-specific fudges.

### 6.5 Universal 8×8 energy field (implemented)

- **`pyhqiv.energy_field.HQIVEnergyField`** carries an 8×8 matrix state. Same equation applies from quark scale to macro liquid He:
  - `energy_density(mass_density, delta_x)` = scalar (ρc² + ħc/Δx) + algebraic part (trace(state @ Δ)).
  - `project_scalar_phi()` = 2c²/trace(M) for backward compatibility with scalar φ code.
- **Ladder:** All layers use **`merge_constituents`**; subatomic (quarks→nucleon), nuclear (nucleons→nucleus), molecular (atoms→molecule) and beyond share one merge. Effective Θ from composite invariant or from **effective_horizon_from_energy_mev(E_tot)** when E_tot comes from integrals.
- **Integration:** `HQIVSystem(..., energy_field=field)` uses `field.project_scalar_phi()` in the constitutive relation; lattice/thermo/fluid can replace scalar δE with `field.energy_density(...)` and use `field.coherence()` for superfluid.

---

## 7. How to implement the hierarchy (sketch)

- **Layer 0 (sub-nucleon)**  
  Valence content \(uud\) / \(udd\); fractional charges \(Q_u = +2/3\), \(Q_d = -1/3\) from hypercharge.  
  **Relax** three quark spheres with hard-sphere + soft attraction + Coulomb (\(Q_i Q_j/d^2\)).  
  **Binding angles** from equilibrium positions; \(\Theta_{\rm free} = L\times 8\times \mu\), \(E = \hbar c/\Theta + E_{\rm Coul}\).  
  Proton has larger Θ; neutron heavier from \(E_{\rm Coul}\). Color (g₂) provides the glue; fractional charge sets the angles.

- **Layer 1 (nucleon)**  
  Use E_proton, E_neutron from layer 0 (or, until layer 0 exists, keep Θ_free_p, Θ_free_n from constants as the effective nucleon horizons).  
  E_free_nucleons = P × E_proton + N × E_neutron.

- **Layer 2 (nucleus)**  
  Compute Θ_i for each nucleon *inside the nucleus* from the overlap graph: one global \(\mu_{\rm comp}\) plus one local \(\mu_i\) per node.  
  E_nucleus = ħc Σ(1/Θ_i).  
  For correct sign, nuclear binding must give **larger** per-nucleon Θ in the nucleus than at layer 1 (so E_nucleus < E_free_nucleons).  
  B_nuclear = E_free_nucleons − E_nucleus.

With this change the same ladder already distinguishes

- free neutron `β-`
- tritium `β-`
- He-4 stability

without any isotope-specific constants or if-branches.
