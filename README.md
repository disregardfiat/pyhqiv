# pyhqiv — Horizon-Quantized Informational Vacuum (HQIV)

[![PyPI version](https://badge.fury.io/py/pyhqiv.svg)](https://badge.fury.io/py/pyhqiv)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18794889.svg)](https://doi.org/10.5281/zenodo.18794889)

Production-ready, pip-installable Python package implementing the **Horizon-Quantized Informational Vacuum (HQIV)** framework exactly as defined in the paper:

> **Ettinger, Steven Jr**, *Horizon-Quantized Informational Vacuum (HQIV): A Unified Framework from Causal Horizon Monogamy and Discrete Null-Lattice Combinatorics*. Zenodo, 2026. [https://doi.org/10.5281/zenodo.18794889](https://doi.org/10.5281/zenodo.18794889)

## Citation

If you use this package in research, please cite the paper:

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
git clone https://github.com/disregardfiat/hqvmpy.git && cd hqvmpy
pip install -e .
```

Optional extras (PDB loading, JAX, QuTiP, visualization):

```bash
pip install pyhqiv[ase,mda,qutip,jax,pyvista]
# or
pip install pyhqiv[all]
```

## Quick start

```python
from pyhqiv import DiscreteNullLattice, HQIVSystem

# Paper numbers: Ω_k^true ≈ +0.0098, m_trans = 500, γ ≈ 0.40
lattice = DiscreteNullLattice(m_trans=500, gamma=0.40)
result = lattice.evolve_to_cmb(T0_K=2.725)
print(result["Omega_true_k"])   # ≈ 0.0098
print(result["age_wall_Gyr"])   # ≈ 51.2
print(result["lapse_compression"])  # ≈ 3.96

# Multi-atom system with phase-corrected fields (optional: from PDB)
sys = HQIVSystem.from_atoms([(0, 0, 0), (1.5, 0, 0)], charges=[1, -1], gamma=0.40)
import numpy as np
grid = np.mgrid[-2:2:11j, -2:2:11j, -2:2:11j].reshape(3, -1).T
E, B = sys.compute_fields(grid, t=0.0)
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
| `src/pyhqiv/constants.py` | Paper constants (γ, α, T_Pl, 6^7√3, etc.) |

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

The test `tests/test_paper_numbers.py` checks Ω_true_k, γ, combinatorial invariant, lapse factor, and lattice δE(m) / mode counts to 6 decimal places.

## License

MIT.
