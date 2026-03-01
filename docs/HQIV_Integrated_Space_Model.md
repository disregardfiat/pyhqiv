# HQIV Integrated Space Model (v1.0)

**Horizon-Quantized Informational Vacuum (HQIV) Integrated Space Model** — definitive generator linking solar/stellar core physics, orbital mechanics at all scales, and redshift in every regime via the single axiom.

**Core axiom (first step in every calculation):**
$$E_{\mathrm{tot}} = m c^2 + \hbar c / \Delta x \quad \text{with} \quad \Delta x \leq \Theta_{\mathrm{local}}(x)$$
$$\Rightarrow \quad \varphi(x) = 2 c^2 / \Theta_{\mathrm{local}}(x)$$
$$\Rightarrow \quad \text{local lapse compression} \quad f(a_{\mathrm{loc}}, \varphi) = a_{\mathrm{loc}} / (a_{\mathrm{loc}} + \varphi/6)$$

---

## 1. HQIV Space Model Architecture Overview

The HQIV Integrated Space Model unifies solar core physics, orbital mechanics (planetary, spacecraft, galactic, black-hole), and redshift (expansion, gravitational, and HQIV mass-lapse) under the **single axiom** \(E_{\mathrm{tot}} = m c^2 + \hbar c/\Delta x\) with \(\Delta x \leq \Theta_{\mathrm{local}}(x)\). The auxiliary field \(\varphi(x) = 2c^2/\Theta_{\mathrm{local}}(x)\) sets the local horizon scale; the lapse compression \(f = a_{\mathrm{loc}}/(a_{\mathrm{loc}} + \varphi/6)\) modifies effective inertia and proper-time rate. At cosmological scales, the discrete null lattice yields **Ω_k^true ≈ +0.0098**, wall-clock age **51.2 Gyr**, and apparent age **13.8 Gyr** via **lapse_today ≈ 3.96**. A key consequence is **lapse degeneracy**: the same apparent redshift can arise from different combinations of expansion, gravitational potential, and HQIV mass-lapse along the line of sight, so that "true" (wall-clock) vs "apparent" (local chronometer) quantities must be carefully decomposed for tests.

---

## 2. Solar Core Insights (Local φ Regime)

### 2.1 φ(r) profile from center to surface

- **Input:** \(\rho(r)\) from standard solar model (e.g. Bahcall–Ulrich / AGSS09). Enclosed mass \(M(r)\), local scale \(\Theta_{\mathrm{local}}(r)\) from causal-horizon monogamy at radius \(r\).
- **Model:** \(\Theta_{\mathrm{local}}(r)\) = characteristic length at \(r\); e.g. \(\Theta(r) \propto r + r_c\) with \(r_c \sim 0.05\,R_\odot\) to avoid singularity, or \(\Theta(r) \propto (G M(r)/c^2)^{1/2}\) (Schwarzschild-like). Then \(\varphi(r) = 2 c^2 / \Theta_{\mathrm{local}}(r)\).
- **Numerical (fiducial):** \(R_\odot \approx 6.96\times10^{10}\) cm, \(r_{\mathrm{core}} \approx 0.2\,R_\odot\), \(\rho_c \approx 150\) g/cm³, \(T_c \approx 15.7\) MK.

### 2.2 Modified core quantities (HQIV vs standard)

| Quantity | Standard SSM | HQIV-corrected |
|----------|--------------|----------------|
| \(T_c\) (core temp) | 15.7 MK | Slightly reduced if \(\varphi\) raises effective opacity / reduces mean free path |
| \(\rho_c\) (core density) | 150 g/cm³ | Same to first order; equation of state with \(f\) can shift equilibrium |
| Luminosity | \(L_\odot\) | \(L_{\mathrm{app}} \sim L_{\mathrm{true}}/f\) (lapse compresses emitted power per unit time) |
| pp-chain / CNO rate | Standard \(\propto T^{n}\) | \(\propto T^n \cdot f(\varphi)\) (clock rate in core) |
| Neutrino flux | \(\Phi_\nu\) (BPS) | \(\Phi_\nu\) scaled by lapse at production vs detection |
| Sound speed \(c_s\) | \(\sqrt{\Gamma P/\rho}\) | \(c_s^2\) corrected by \(f\) in momentum equation |
| Opacity | \(\kappa\) (OP/OPAL) | Effective \(\kappa_{\mathrm{eff}}\) if \(\varphi\) alters photon mean free path |

### 2.3 Numerical comparison

- **Standard:** \(T_c \approx 15.7\times10^6\) K, \(\rho_c \approx 150\) g/cm³, \(L_\odot \approx 3.828\times10^{26}\) W, age \(\approx 4.57\) Gyr (apparent).
- **HQIV:** Wall-clock solar age \(\approx 4.57 \times 3.96 \approx 18.1\) Gyr; core \(\varphi_c\) large ⇒ \(f_c < 1\) ⇒ slightly lower effective \(T\) for same fusion yield; luminosity shift \(\sim 1/f\) at emission (observer frame).

### 2.4 Edge case: early-universe solar-mass stars

- Higher background \(\varphi\) (smaller \(\Theta_{\mathrm{local}}\) at high \(z\)) ⇒ larger \(\varphi/6\) in denominator of \(f\) ⇒ **smaller \(f\)** ⇒ stronger lapse, slower local clocks. Early "solar-mass" stars would have **higher apparent central temperature** for same wall-clock fusion history, or **older wall-clock age** for same apparent luminosity.

---

## 3. Redshift Decomposition (All Components)

For any object (e.g. GN-z11, supernova at \(z=1\), pulsar):

| Component | Formula | Role |
|-----------|---------|------|
| **z_expansion** | \((1+z_{\mathrm{exp}}) = a_0/a_{\mathrm{emit}}\) (FLRW) | Scale factor; standard Hubble law |
| **z_gravitational** | \((1+z_{\mathrm{grav}}) \approx 1 + \Delta\Phi/c^2\) | Local potential (e.g. galaxy, cluster, BH) |
| **z_HQIV_mass_lapse** | \((1+z_{\mathrm{HQIV}}) = \prod_{\mathrm{path}} (1 + \varphi/(6 a_{\mathrm{loc}}))^{-1}\) or integrated lapse | φ along line-of-sight + observer-centric compression |
| **z_total (apparent)** | \((1+z_{\mathrm{app}}) = (1+z_{\mathrm{exp}})(1+z_{\mathrm{grav}})(1+z_{\mathrm{HQIV}})\) | What a local chronometer / photon ratio sees |
| **z_total (true/wall-clock)** | Same photons; wall-clock time at emitter = apparent time \(\times\) lapse_compression | 51.2 Gyr vs 13.8 Gyr cosmology |

**Explicit degeneracy:** A high-\(z\) galaxy with \(z_{\mathrm{app}} = 11\) can be interpreted as (i) pure FLRW \(z_{\mathrm{exp}}=11\), or (ii) \(z_{\mathrm{exp}} = 8\) plus \(z_{\mathrm{HQIV}}\) accounting for integrated \(\varphi\) along the path (lapse degeneracy). Fitting both requires independent probes (e.g. BAO, supernovae, chronometers).

**Numerical example (GN-z11, \(z_{\mathrm{app}} \approx 11\)):**

- Standard: \(z_{\mathrm{exp}} = 11\), age at emission \(\approx 0.4\) Gyr after Big Bang.
- HQIV: Wall-clock age at emission \(\approx 0.4 \times 3.96 \approx 1.58\) Gyr; \(z_{\mathrm{exp}}\) (scale factor) slightly less if part of \(z_{\mathrm{app}}\) is \(z_{\mathrm{HQIV}}\).

---

## 4. Orbital Mechanics in HQIV

### 4.1 Newtonian / GR baseline

- Newton: \(\ddot{\mathbf{r}} = -GM/r^2\,\hat{\mathbf{r}}\); Keplerian ellipses.
- GR: Periapsis precession, Shapiro delay, time dilation \(d\tau/dt = \sqrt{1 - 2GM/(c^2 r)}\).

### 4.2 HQIV corrections

- **Lapse on proper time:** \(d\tau/dt = 1/\sqrt{1 + \gamma(\varphi/c^2)(\dot{\delta\theta}'/c)}\) (homogeneous limit \(\approx 1/3.96\) today).
- **Effective inertia:** \(f(a_{\mathrm{loc}}, \varphi) = a_{\mathrm{loc}}/(a_{\mathrm{loc}} + \varphi/6)\); in equations of motion, effective mass or force scaling.
- **Clock desynchronization:** Spacecraft clock vs Earth: \(\Delta t_{\mathrm{spacecraft}} = \Delta t_{\mathrm{Earth}} \cdot f_{\mathrm{sc}}/f_{\mathrm{Earth}}\) (plus GR and velocity terms).

### 4.3 Examples

| System | Baseline | HQIV effect |
|--------|----------|-------------|
| Earth–Sun | Kepler \(a \approx 1\) au, \(T \approx 1\) yr | Lapse shift on GPS-like clocks: \(\sim (1/f - 1) \approx 0.25\) today |
| Parker Solar Probe | Perihelion \(\sim 0.05\) au | High \(\varphi\) near Sun ⇒ \(f\) smaller ⇒ clock runs slower; orbit integration with \(f(r)\) |
| Galactic rotation | Flat curve (MOND/dark matter) | Flatness from local \(\varphi\) gradients? \(v^2/r \sim \nabla\varphi\) contribution |
| Sgr A* orbit | GR precession | Lapse-corrected proper time; possible extra precession from \(f(r)\) |

### 4.4 Ready-to-run: REBOUND / Gala + pyhqiv

- Use REBOUND or Gala for Newton/GR; at each step compute \(\varphi(r)\), \(f(r)\), and advance proper time with \(d\tau = dt/f\) or scale forces by \(1/f\). See `HQIVOrbit` class below.

---

## 5. Ready-to-Run pyhqiv Extensions (v0.4.0)

### 5.1 HQIVSolarCore

```python
from pyhqiv import HQIVSolarCore, phi_solar_radial_profile

sun = HQIVSolarCore(R_star=6.9634e8, r_core_frac=0.2, rho_c_cgs=150.0, T_c_MK=15.7)
table = sun.standard_vs_hqiv_table(lapse_compression=3.96)  # T_c, ρ_c, f_core, age
r_m, phi, f = phi_solar_radial_profile(128)  # φ(r), f(r) for plotting
```

### 5.2 HQIVRedshift

```python
from pyhqiv import HQIVRedshift, DiscreteNullLattice

lattice = DiscreteNullLattice()
red = HQIVRedshift(lapse_compression=3.96, age_wall_Gyr=51.2).with_lattice(lattice)
decomp = red.decompose_from_apparent(z_app=11.0, z_HQIV_fraction=0.1)
age_wall = red.wall_clock_age_at_emission(11.0)
```

### 5.3 HQIVOrbit

```python
from pyhqiv import HQIVOrbit, parker_perihelion_lapse

orb = HQIVOrbit()  # M_central = M_sun
t, r, v, tau = orb.earth_sun_example(n_steps=500, n_orbits=0.25)
f_parker = parker_perihelion_lapse(0.05)  # at 0.05 au
```

### 5.4 Full example

Run: `python examples/hqiv_space_model_full.py --plot` for Sun + Earth orbit + high-z galaxy and figures.

---

## 6. Visualizations

- φ(r) in solar core: radial profile from center to \(R_\odot\).
- Apparent vs true redshift vs \(z\): plot \(z_{\mathrm{app}}\) and \(z_{\mathrm{true}}\) (or wall-clock age) vs scale factor.
- Lapse-corrected orbital precession or time dilation: e.g. Earth–Sun \(\tau(t)\) or Parker probe \(f(r)\).

See `examples/hqiv_space_model_full.py` (matplotlib) and optional pyvista for 3D orbits.

---

## 7. Testable Predictions & Next Experiments

| Probe | Observable | HQIV prediction |
|-------|------------|-----------------|
| Solar neutrinos (Borexino, SNO+) | \(\Phi_\nu\), spectrum | Slight shift from core lapse; rate \(\propto f_{\mathrm{core}}\) |
| Gaia / pulsar timing | Proper time, distances | Clock desync \(\sim (1/f - 1)\); secular drift |
| JWST / Roman high-\(z\) | \(z_{\mathrm{app}}\), ages | Lapse degeneracy: older wall-clock ages; possible \(z_{\mathrm{HQIV}}\) component |
| Spacecraft (lunar, Mars) | Clock sync | \(f\) differs along orbit; compare with GR + HQIV |
| Lab (semiconductor band-gap) | \(\Delta E_g(\varphi)\) | Micro-analog of \(\varphi\); pyhqiv `compute_band_gap`, `hqiv_potential_shift` |

**Precise numbers:** Ω_k^true = 0.0098, wall-clock age 51.2 Gyr, lapse_today ≈ 3.96, solar \(\rho_c \approx 150\) g/cm³, \(T_c \approx 15.7\) MK. Always compare standard result side-by-side with HQIV; quantitative to 4–6 digits.
