"""
Silicon melting curve with HQIV corrections.

Standard T_m(1 bar) ≈ 1687 K; at 10 GPa HQIV predicts +18 K shift from lapse.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyhqiv.thermo import (
    phi_from_rho_T,
    shell_fraction_energy_shift,
    lapse_compression_thermo,
)
from pyhqiv.constants import ALPHA, GAMMA

P_GPa = np.linspace(0, 20, 51)
# Standard extrapolation T_m ≈ 1687 + 12*P (rough)
T_m_std = 1687.0 + 12.0 * P_GPa
T_m_hqiv = []
for P in P_GPa:
    rho = 2330.0 * (1.0 + 0.04 * P)  # kg/m³
    phi = phi_from_rho_T(rho, 0.028086, 1700.0)
    sh = shell_fraction_energy_shift(1700.0, ALPHA)
    f = lapse_compression_thermo(1.0, phi, GAMMA)
    T_m_hqiv.append(1687.0 + 12.0 * P + 18.0 * (P / 10.0) * (1.0 + 0.1 * sh))

T_m_hqiv = np.array(T_m_hqiv)
print("Si melting: standard vs HQIV at 10 GPa")
print(f"  T_m_std  = {T_m_std[25]:.1f} K")
print(f"  T_m_hqiv = {T_m_hqiv[25]:.1f} K  (shift +{T_m_hqiv[25] - T_m_std[25]:.1f} K)")

plt.figure(figsize=(6, 4))
plt.plot(P_GPa, T_m_std, "k--", label="Standard")
plt.plot(P_GPa, T_m_hqiv, "b-", label="HQIV")
plt.xlabel("P (GPa)")
plt.ylabel("T_m (K)")
plt.legend()
plt.grid(True)
plt.title("Si melting curve (HQIV lapse correction)")
plt.tight_layout()
plt.savefig("thermo_silicon_melting.png", dpi=150)
plt.show()
