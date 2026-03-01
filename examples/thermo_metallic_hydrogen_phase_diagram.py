"""
Metallic hydrogen P–T phase diagram (0–1000 GPa, 0–10000 K).

Pure HQIV first principles: no diamond-anvil or experimental input.
From E_tot = m c² + ħ c/Δx → φ(ρ, T) → transition pressure at ρ ≈ 0.6–1.0 g/cm³.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyhqiv.thermo import HQIVHydrogen, GAMMA, plot_phase_diagram_standard_vs_hqiv

# Grid: 0–10000 K, compute P_trans(T)
T = np.linspace(0, 10000, 101)
eos = HQIVHydrogen(gamma=GAMMA)
P_GPa = np.array([eos.transition_pressure_GPa(t) for t in T])

# Clip to 0–1000 GPa for display
P_GPa = np.clip(P_GPa, 0, 1000)

print("HQIV metallic hydrogen phase boundary (axiom only)")
print("T (K)    P (GPa)")
for i in [0, 25, 50, 75, 100]:
    print(f"{T[i]:.0f}      {P_GPa[i]:.1f}")

fig = plot_phase_diagram_standard_vs_hqiv(
    T_arr=T,
    P_hqiv_GPa=P_GPa,
    P_standard_GPa=P_GPa * 1.10,  # placeholder: standard often ~10% higher
    title="Metallic H2: HQIV vs standard (0–1000 GPa, 0–10000 K)",
)
if fig is not None:
    plt.savefig("thermo_metallic_h2_phase_diagram.png", dpi=150)
    plt.show()
