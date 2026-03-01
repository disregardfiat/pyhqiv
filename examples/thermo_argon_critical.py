"""
Argon critical-point shift in semiconductor vacuum chamber.

Standard: T_c = 150.87 K, P_c = 4.898 MPa.
HQIV lapse at ρ_c shifts critical point (falsifiable prediction).
"""

import numpy as np
import matplotlib.pyplot as plt
from pyhqiv.thermo import phi_from_rho_T, lapse_compression_thermo

T_c_std = 150.87  # K
P_c_std = 4.898e6  # Pa
rho_c = 535.0  # kg/m³
M_Ar = 0.03995  # kg/mol

phi = phi_from_rho_T(rho_c, M_Ar, T_K=T_c_std)
f = lapse_compression_thermo(1.0, phi)
T_c_hqiv = T_c_std * (1.0 + 0.02 * (1.0 - f))
P_c_hqiv = P_c_std * (1.0 + 0.02 * (1.0 - f))

print("Ar critical point: standard vs HQIV (vacuum chamber)")
print(f"  Standard: T_c = {T_c_std:.2f} K, P_c = {P_c_std/1e6:.3f} MPa")
print(f"  HQIV:     T_c = {T_c_hqiv:.2f} K, P_c = {P_c_hqiv/1e6:.3f} MPa")
print(f"  Delta T_c = {T_c_hqiv - T_c_std:.3f} K, Delta P_c = {(P_c_hqiv - P_c_std)/1e6:.4f} MPa")

plt.figure(figsize=(5, 4))
plt.plot(T_c_std, P_c_std / 1e6, "ko", label="Standard Ar critical", markersize=10)
plt.plot(T_c_hqiv, P_c_hqiv / 1e6, "b^", label="HQIV shift", markersize=10)
plt.xlabel("T (K)")
plt.ylabel("P (MPa)")
plt.legend()
plt.grid(True)
plt.title("Ar critical point shift (HQIV)")
plt.tight_layout()
plt.savefig("thermo_argon_critical.png", dpi=150)
plt.show()
