"""
Molecular / protein HQIV: Θ from Z and coordination, bond length from Θ, damping,
and lightweight HQIV energy primitives for torsion / coupling DOFs.

Single source of truth for PROtien and other molecular HQIV uses. Re-exports from
utils and constants so callers can do:

    from pyhqiv.molecular import theta_local, bond_length_from_theta, ...

and exposes temperature-aware torsion energy helpers such as
`hqiv_energy_for_angles` and `coupling_angle_energy_profile`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from pyhqiv.constants import (
    A_LOC_ANG,
    ALPHA,
    GAMMA,
    HBAR_C_EV_ANG,
    K_B_SI,
    T_PL_K,
)
from pyhqiv.utils import (
    bond_length_from_theta,
    damping_force_magnitude,
    phi_from_theta_local,
    theta_for_atom,
    theta_local,
)

__all__ = [
    "theta_local",
    "theta_for_atom",
    "bond_length_from_theta",
    "damping_force_magnitude",
    "phi_from_theta_local",
    "GAMMA",
    "HBAR_C_EV_ANG",
    "A_LOC_ANG",
    "hqiv_energy_for_angles",
    "hqiv_energy_for_angles_batch",
    "coupling_angle_energy_profile",
    "coupling_angle_energy_profile_batch",
]


def _shell_fraction_energy_shift(
    T_K: float,
    alpha: float = ALPHA,
    T_Pl_K: float = T_PL_K,
) -> float:
    """
    Dimensionless HQIV shell-fraction × ln(1 + α ln(T_Pl/T)) factor (thermo-style).

    This mirrors `thermo.shell_fraction_energy_shift` but is kept local here to
    avoid heavier imports for molecular/protein use. For protein-like temperatures
    (≪ T_Pl) this is a very small correction but still provides a well-defined
    HQIV temperature dependence.
    """
    T = max(float(T_K), 1e-30)
    x = min(T / T_Pl_K, 1.0 - 1e-10)
    ln_term = np.log(max(T_Pl_K / T, 1.0))
    inner = max(1.0 + alpha * ln_term, 1.0 + 1e-10)
    return float(x * np.log(inner))


def _torsion_shape(dof_type: str, angles_rad: np.ndarray) -> np.ndarray:
    """
    Dimensionless periodic shape V(θ) in [0, ~1] for a given torsion/coupling DOF.

    The exact form is phenomenological but HQIV-consistent: multi-well, periodic,
    and easily vectorizable. It is intended to provide a small discrete state
    graph per DOF (φ, ψ, ω, χ, etc.) rather than a high-resolution Ramachandran.
    """
    a = np.asarray(angles_rad, dtype=float)
    # Basic multi-well torsion potential built from a small set of cosines.
    dof = dof_type.lower()
    if dof in ("phi", "psi"):
        # Threefold periodicity with a bias towards helices/sheets.
        v = 0.5 * (1.0 - np.cos(3.0 * a)) + 0.2 * (1.0 - np.cos(a))
    elif dof == "omega":
        # Peptide bond: mostly trans, occasional cis.
        v = 0.7 * (1.0 - np.cos(a)) + 0.3 * (1.0 - np.cos(2.0 * a))
    elif dof.startswith("chi"):
        # Side-chain χ: typically threefold with mild asymmetry.
        v = 0.5 * (1.0 - np.cos(3.0 * a)) + 0.1 * (1.0 - np.cos(2.0 * a))
    else:
        # Generic coupling/group DOF (helix tilt, loop hinge, domain twist, ...).
        v = 0.5 * (1.0 - np.cos(2.0 * a))
    # Normalize so the maximum is O(1) but avoid division by zero for flat shapes.
    vmax = float(np.max(v)) if np.size(v) > 0 else 1.0
    if vmax <= 0.0:
        return np.zeros_like(v)
    return v / vmax


def _torsion_energy_ev(
    dof_type: str,
    angles_rad: np.ndarray,
    theta_local_ang: float,
    temperature_K: Optional[float] = None,
    *,
    a_loc: float = A_LOC_ANG,
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    HQIV torsion / coupling energy for one DOF in eV, vectorized over angles.

    Parameters
    ----------
    dof_type : str
        "phi", "psi", "omega", "chi", "helix_tilt", "loop_hinge", etc.
    angles_rad : array_like
        Torsion or group angle(s) in radians.
    theta_local_ang : float
        Local diamond size Θ in Å for this residue/segment.
    temperature_K : float, optional
        Temperature in K. If None, returns the T-independent HQIV baseline.
    a_loc : float
        Local acceleration scale in lapse/damping; A_LOC_ANG by default.
    gamma : float
        HQIV thermodynamic coefficient (defaults to pyhqiv.constants.GAMMA).

    Returns
    -------
    energies_ev : np.ndarray
        HQIV energy in eV for each angle in `angles_rad`.
    """
    theta_eff = max(float(theta_local_ang), 1e-8)
    # Informational energy scale ~ ħc / Θ (eV·Å / Å → eV).
    E_scale_ev = HBAR_C_EV_ANG / theta_eff
    # φ and a simple |∇φ| proxy tied to angular displacement.
    phi = float(phi_from_theta_local(theta_eff, c=1.0))
    angles = np.asarray(angles_rad, dtype=float)
    grad_phi = (phi / theta_eff) * np.ones_like(angles)
    damping = damping_force_magnitude(phi, grad_phi, a_loc=a_loc, gamma=gamma)
    # Periodic shape in [0, 1] for this DOF.
    V_shape = _torsion_shape(dof_type, angles)
    # Baseline HQIV energy landscape (T-independent).
    base = E_scale_ev * V_shape * (1.0 + damping)

    if temperature_K is None:
        return base

    # HQIV-motivated effective kT scaling: small shell-fraction × log correction.
    shell_shift = _shell_fraction_energy_shift(temperature_K)
    # Convert k_B T (J) to eV with the HQIV shell fraction; for protein T this is
    # tiny but provides a consistent ΔE/kT scale for Metropolis.
    kT_eff_ev = (K_B_SI * temperature_K * shell_shift) / 1.602176634e-19
    # Treat temperature as an overall softening of barriers: higher T flattens V.
    # We keep a simple linear interpolation between base and a softened landscape.
    soften_factor = 1.0 / (1.0 + (kT_eff_ev / max(E_scale_ev, 1e-12)))
    return base * soften_factor


def hqiv_energy_for_angles(
    phi: float,
    psi: float,
    theta_local_ang: float,
    temperature: Optional[float] = None,
    *,
    a_loc: float = A_LOC_ANG,
    gamma: float = GAMMA,
) -> float:
    """
    HQIV backbone energy for a single residue from (φ, ψ) torsions.

    Parameters
    ----------
    phi, psi : float
        Backbone torsion angles in radians.
    theta_local_ang : float
        Local diamond size Θ in Å for the residue (from `theta_local` or
        `theta_for_atom`).
    temperature : float, optional
        Temperature in K. If None, behaves as a T-independent HQIV energy.
    a_loc, gamma : float
        Lapse / damping parameters; see `damping_force_magnitude`.

    Returns
    -------
    energy_ev : float
        HQIV energy in eV for this residue's backbone torsions.
    """
    angles = np.array([phi, psi], dtype=float)
    E_phi_psi = _torsion_energy_ev(
        "phi+psi",
        angles_rad=angles,
        theta_local_ang=theta_local_ang,
        temperature_K=temperature,
        a_loc=a_loc,
        gamma=gamma,
    )
    # Combine φ and ψ contributions additively.
    return float(np.sum(E_phi_psi))


def hqiv_energy_for_angles_batch(
    phi: np.ndarray,
    psi: np.ndarray,
    theta_local_ang: np.ndarray,
    temperature: Optional[float] = None,
    *,
    a_loc: float = A_LOC_ANG,
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    Vectorized HQIV backbone energy for many residues.

    Parameters
    ----------
    phi, psi : array_like
        Arrays of backbone torsion angles in radians (broadcastable).
    theta_local_ang : array_like
        Θ in Å per residue (broadcastable with `phi` / `psi`).
    temperature : float, optional
        Temperature in K. If None, uses T-independent HQIV baseline.

    Returns
    -------
    energies_ev : np.ndarray
        HQIV energy in eV per residue, shape broadcast from inputs.
    """
    phi_arr = np.asarray(phi, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    theta_arr = np.asarray(theta_local_ang, dtype=float)
    # Broadcast all inputs to a common shape.
    phi_b, psi_b, theta_b = np.broadcast_arrays(phi_arr, psi_arr, theta_arr)
    # Flatten for evaluation, then reshape to original broadcast shape.
    flat_phi = phi_b.ravel()
    flat_psi = psi_b.ravel()
    flat_theta = theta_b.ravel()
    angles = np.stack([flat_phi, flat_psi], axis=-1)
    energies_flat = _torsion_energy_ev(
        "phi+psi",
        angles_rad=angles,
        theta_local_ang=float(np.mean(flat_theta)),
        temperature_K=temperature,
        a_loc=a_loc,
        gamma=gamma,
    )
    energies = np.sum(energies_flat.reshape(phi_b.shape + (2,)), axis=-1)
    return energies


def coupling_angle_energy_profile(
    dof_type: str,
    theta_local_ang: float,
    temperature: Optional[float] = None,
    *,
    n_states: int = 32,
) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
    """
    Discrete HQIV energy profile for a single coupling/torsion DOF.

    Parameters
    ----------
    dof_type : str
        Identifier of the coupling DOF, e.g. "phi", "psi", "omega", "chi",
        or group DOFs like "helix_tilt", "loop_hinge".
    theta_local_ang : float
        Local diamond size Θ in Å (or effective Θ for the residue/segment).
    temperature : float, optional
        Effective temperature in K. If None, use T-independent HQIV behavior.
    n_states : int
        Number of discrete angle states sampled over [-π, π) in radians.

    Returns
    -------
    angles : list[float]
        Allowed angle states in radians, sorted and evenly spaced.
    energies : list[float]
        HQIV energy in eV at each angle state.
    deltaE_neighbors : list[float]
        First difference E[i+1] - E[i] (last element set to NaN).
    """
    n = max(int(n_states), 2)
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=float)
    energies = _torsion_energy_ev(
        dof_type,
        angles_rad=angles,
        theta_local_ang=theta_local_ang,
        temperature_K=temperature,
    )
    deltaE = np.empty_like(energies)
    deltaE[:-1] = energies[1:] - energies[:-1]
    deltaE[-1] = np.nan
    return angles.tolist(), energies.tolist(), deltaE.tolist()


def coupling_angle_energy_profile_batch(
    dof_type: str,
    theta_local_ang: np.ndarray,
    temperature: Optional[float] = None,
    *,
    n_states: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized discrete HQIV energy profile for a coupling DOF.

    Parameters
    ----------
    dof_type : str
        Identifier of the coupling DOF, e.g. "phi", "psi", "omega", "chi",
        "helix_tilt", "loop_hinge", "domain_twist".
    theta_local_ang : np.ndarray
        Array of Θ in Å, shape (N,).
    temperature : float, optional
        Effective temperature in K. If None, uses T-independent HQIV baseline.
    n_states : int
        Number of discrete angle states sampled over [-π, π) in radians.

    Returns
    -------
    angles : np.ndarray
        Angle states in radians, shape (n_states,). Shared across all inputs.
    energies : np.ndarray
        HQIV energy in eV at each angle state, shape (N, n_states).
    deltaE_neighbors : np.ndarray
        First difference E[i+1] - E[i] along the angle axis, shape (N, n_states),
        with the last column set to NaN.
    """
    theta_arr = np.asarray(theta_local_ang, dtype=float).ravel()
    n = max(int(n_states), 2)
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=float)
    # For now the energy scale only depends on Θ, so we can evaluate per Θ and DOF.
    energies_list = []
    delta_list = []
    for theta_val in theta_arr:
        e = _torsion_energy_ev(
            dof_type,
            angles_rad=angles,
            theta_local_ang=float(theta_val),
            temperature_K=temperature,
        )
        d = np.empty_like(e)
        d[:-1] = e[1:] - e[:-1]
        d[-1] = np.nan
        energies_list.append(e)
        delta_list.append(d)
    energies = np.stack(energies_list, axis=0)
    deltaE = np.stack(delta_list, axis=0)
    return angles, energies, deltaE
