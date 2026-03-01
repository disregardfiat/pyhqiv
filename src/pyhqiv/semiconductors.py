"""
High-level semiconductor property extractors with HQIV corrections.

Band gap, DOS, effective mass, conductivity tensor, dielectric/optical response.
Intended for use with DFT band structures plus HQIV potential shift.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import GAMMA
from pyhqiv.crystal import hqiv_potential_shift
from pyhqiv.response import compute_conductivity, response_tensor_diagonal


def compute_band_gap(
    eigenvalues: np.ndarray,
    k_weights: Optional[np.ndarray] = None,
    phi_avg: float = 0.0,
    dot_delta_theta_avg: float = 0.0,
    gamma: float = GAMMA,
) -> Tuple[float, str]:
    """
    Band gap from eigenvalue array with optional HQIV shift.

    Parameters
    ----------
    eigenvalues : (n_k, n_bands) or (n_bands,) array
        Single k or multiple k; last axis = bands.
    k_weights : (n_k,) array, optional
        Weights per k-point (e.g. for irreducible wedge).
    phi_avg : float
        Average φ for HQIV potential shift (default 0 = no shift).
    dot_delta_theta_avg : float
        Average δ̇θ′ for HQIV shift.
    gamma : float
        HQIV monogamy coefficient.

    Returns
    -------
    gap : float
        Band gap (eV or same units as eigenvalues).
    gap_type : str
        "direct" or "indirect" (indirect if min gap is not at same k).
    """
    ev = np.asarray(eigenvalues, dtype=float)
    if ev.ndim == 1:
        ev = ev.reshape(1, -1)
    v_shift = hqiv_potential_shift(phi_avg, dot_delta_theta_avg, gamma=gamma)
    ev_shifted = ev + v_shift

    if k_weights is not None:
        w = np.asarray(k_weights, dtype=float).ravel()
        w = w / w.sum()
    else:
        w = np.ones(ev.shape[0]) / ev.shape[0]

    # Valence max and conduction min over bands (assume last axis is bands)
    n_bands = ev_shifted.shape[-1]
    n_v = n_bands // 2  # simple half-filling
    vb = np.max(ev_shifted[..., :n_v], axis=-1)
    cb = np.min(ev_shifted[..., n_v:], axis=-1)
    gap_per_k = cb - vb
    gap = float(np.min(gap_per_k))
    gap_direct = float(np.min(ev_shifted[..., n_v] - np.max(ev_shifted[..., :n_v], axis=-1)))
    gap_type = "direct" if abs(gap - gap_direct) < 1e-10 else "indirect"
    return gap, gap_type


def dos(
    eigenvalues: np.ndarray,
    energies: np.ndarray,
    sigma: float = 0.05,
    k_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Density of states (smoothed) from band eigenvalues.

    ρ(E) = sum_{n,k} w_k delta_sigma(E - E_nk). Uses Gaussian broadening.

    Parameters
    ----------
    eigenvalues : (n_k, n_bands) array
        Band energies per k-point.
    energies : 1D array
        Energy grid (eV) for DOS.
    sigma : float
        Gaussian broadening (eV).
    k_weights : (n_k,) array, optional
        Weights per k-point.

    Returns
    -------
    rho : 1D array
        DOS on energy grid (states/eV per unit cell or per formula).
    """
    ev = np.asarray(eigenvalues, dtype=float).reshape(-1)
    grid = np.asarray(energies, dtype=float)
    if k_weights is not None:
        w = np.asarray(k_weights, dtype=float).ravel()
        w = np.repeat(w, ev.shape[0] // len(w))
        w = w / w.sum() * len(w)
    else:
        w = np.ones_like(ev) / ev.size
    rho = np.zeros_like(grid, dtype=float)
    for i, e in enumerate(grid):
        rho[i] = np.sum(w * np.exp(-0.5 * ((ev - e) / max(sigma, 1e-30)) ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return rho


def effective_mass(
    eigenvalues: np.ndarray,
    k_points: np.ndarray,
    band_index: int,
    direction: Union[int, Tuple[float, float, float]] = 0,
    dk: float = 1e-5,
) -> float:
    """
    Effective mass (in units of m_e) from curvature of E(k) along a direction.

    m* = ℏ² / (d²E/dk²). Uses finite difference along direction.

    Parameters
    ----------
    eigenvalues : (n_k, n_bands) array
        Band energies. Must have at least 3 k-points along the path for curvature.
    k_points : (n_k, 3) array
        K-points in 1/Å (or same units as curvature).
    band_index : int
        Band index (0-based).
    direction : int or (3,) array
        Axis (0,1,2) or direction vector for dk.
    dk : float
        Finite-difference step in k.

    Returns
    -------
    m_star : float
        Effective mass (relative to m_e). Uses ℏ²/m_e in eV·Å² so 1/m* = (1/ℏ²) d²E/dk².
    """
    # ℏ²/(2m_e) ≈ 7.62 eV·Å² so m*/m_e = 7.62 / (d²E/dk² in eV/Å²)
    HBAR2_OVER_2ME_EV_ANG2 = 7.62
    ev = np.asarray(eigenvalues, dtype=float)
    kpt = np.asarray(k_points, dtype=float)
    if ev.ndim == 1:
        ev = ev.reshape(1, -1)
    nk, nb = ev.shape
    if band_index >= nb or nk < 3:
        return np.nan
    idx = min(band_index, nb - 1)
    if isinstance(direction, int):
        ax = direction
    else:
        ax = 0
    i = nk // 2
    if i == 0 or i >= nk - 1:
        return np.nan
    k0 = kpt[i]
    e0 = ev[i, idx]
    k_plus = k0.copy()
    k_plus[ax] += dk
    k_minus = k0.copy()
    k_minus[ax] -= dk

    def e_at(kq: np.ndarray) -> float:
        dist = np.linalg.norm(kpt - kq, axis=1)
        j = np.argmin(dist)
        return float(ev[j, idx])

    e_plus = e_at(k_plus)
    e_minus = e_at(k_minus)
    d2edk2 = (e_plus - 2 * e0 + e_minus) / (dk ** 2)
    if abs(d2edk2) < 1e-20:
        return np.nan
    m_star = HBAR2_OVER_2ME_EV_ANG2 / d2edk2
    return float(m_star)


def compute_conductivity_tensor(
    omega: float,
    T: float = 300.0,
    sigma_0: float = 1.0,
    phi_avg: float = 0.0,
    gamma: float = GAMMA,
    dim: int = 3,
) -> np.ndarray:
    """
    Conductivity tensor σ(ω, T) with HQIV correction.

    Wrapper around response_tensor_diagonal; T can be used for temperature-dependent σ_0
    in extended models.
    """
    return response_tensor_diagonal(omega, dim=dim, sigma_0=sigma_0, phi_avg=phi_avg, gamma=gamma)


def dielectric_function_epsilon(
    omega: np.ndarray,
    sigma_0: float = 1.0,
    phi_avg: float = 0.0,
    gamma: float = GAMMA,
    epsilon_inf: float = 1.0,
) -> np.ndarray:
    """
    Dielectric function ε(ω) from Drude-like response with HQIV correction.

    ε(ω) = ε_inf - σ(ω)/(i ω ε_0) in SI-style; here returns real and imag parts
    as (2, len(omega)) or complex array. Simplified: returns 1 - sigma/(i*omega)
    in natural units for illustration.
    """
    omega = np.asarray(omega, dtype=float)
    sigma = compute_conductivity(omega, sigma_0, phi_avg=phi_avg, gamma=gamma)
    eps = epsilon_inf - sigma / (1j * np.maximum(omega, 1e-30))
    return np.asarray(eps, dtype=complex)
