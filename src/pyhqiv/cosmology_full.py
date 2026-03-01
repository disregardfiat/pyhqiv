"""
Optional heavy cosmology module: full universe evolution → CMB map.

Perturbations and the lattice stay in **main** (pyhqiv.perturbations, pyhqiv.lattice).
This module implements: growth D(z), σ₈, universe_evolver, C_ℓ, LOS/ISW, and
optional Healpy full-sky map.

Install: pip install pyhqiv[cosmology] for healpy. Core logic (σ₈, C_ℓ, LOS/ISW)
works without healpy.
Design: docs/HQIV_CMB_Pipeline.md
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyhqiv.constants import ALPHA, GAMMA
from pyhqiv.cosmology import HQIVCosmology
from pyhqiv.lattice import curvature_imprint_delta_E
from pyhqiv.perturbations import HQIVPerturbations

# NumPy 2.0 removed trapz; use trapezoid when available
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapz is None:
    from scipy.integrate import trapezoid as _trapz  # type: ignore

# CMB monopole in μK (for kinematic dipole)
T_CMB_MUK = 2.725e6

__all__ = [
    "universe_evolver",
    "hqiv_cmb",
    "sigma8",
    "c_ell_spectrum",
    "full_sky_healpy_map",
    "line_of_sight_isw_rees_sciama",
    "add_kinematic_dipole",
]

# Fiducial cosmology (flat ΛCDM-like for transfer/growth; HQIV modifies via f(φ))
H0_KM_S_MPC = 67.36
H0_SI = H0_KM_S_MPC * 1e3 / 3.086e22  # 1/s
HUBBLE_MPC = 3000.0 / H0_KM_S_MPC  # c/H0 in Mpc
OMEGA_M0 = 0.315
OMEGA_L0 = 1.0 - OMEGA_M0
N_S = 0.9649  # scalar spectral index
K_PIVOT = 0.05  # 1/Mpc
R8_MPC = 8.0 / (H0_KM_S_MPC / 100.0)  # 8 h⁻¹ Mpc in Mpc
T_PL_K = 1.22e19 * 1.16e13  # Planck T in K (approx)


def _get_cosmology(cosmology: Optional[Any]) -> HQIVCosmology:
    if cosmology is not None and isinstance(cosmology, HQIVCosmology):
        return cosmology
    return HQIVCosmology(gamma=GAMMA, alpha=ALPHA)


def _get_background(
    bulk_seed: Optional[Any],
    cosmology: Optional[Any],
) -> dict:
    """
    Background (Ω_k, H₀, lapse) for CMB pipeline.
    When bulk_seed is provided (from HQIV bulk.py), use it as authoritative until baryogenesis complete.
    Otherwise use lattice evolve_to_cmb() and default H₀.
    """
    if bulk_seed is not None and isinstance(bulk_seed, dict):
        return {
            "Omega_true_k": bulk_seed.get("Omega_true_k", bulk_seed.get("omega_k_true")),
            "H0_km_s_Mpc": bulk_seed.get("H0_km_s_Mpc", H0_KM_S_MPC),
            "lapse_compression": bulk_seed.get("lapse_compression", 3.96),
        }
    cosmo = _get_cosmology(cosmology)
    result = cosmo.evolve_to_cmb()
    return {
        "Omega_true_k": result["Omega_true_k"],
        "H0_km_s_Mpc": H0_KM_S_MPC,
        "lapse_compression": result["lapse_compression"],
    }


def _lapse_f_from_lattice(k: float, z: float) -> float:
    """Lapse factor f(z) from curvature imprint at effective shell m ∝ k."""
    m = np.clip(int(k * 50), 0, 499)
    T = 2.725 * (1.0 + z)
    delta_E = curvature_imprint_delta_E(np.array([m]), np.array([T]), alpha=ALPHA)
    f = 1.0 / (1.0 + float(np.asarray(delta_E).flat[0]) / 1e6)
    return float(np.clip(f, 0.1, 1.0))


def _growth_factor_hqiv(
    z_arr: np.ndarray,
    cosmology: HQIVCosmology,
    k_ref: float = 0.1,
) -> np.ndarray:
    """
    Growth factor D(z) with HQIV lapse modulation.
    D(z) normalized to D(0) = 1. Uses cosmological_perturbation (delta_growth, f).
    """
    pert = HQIVPerturbations(cosmology)
    # Sample (delta_growth, f) at each z; build D ∝ delta_growth normalized at z=0
    # delta_growth from perturbations is (standard * f); f typically larger at high z
    # so delta_growth is larger at high z. Growth D(z) should be 1 today and < 1 in past.
    # Use D(z) = d0 / d_vals so D increases toward z=0 (structure grows).
    d_vals = np.zeros_like(z_arr)
    for i, z in enumerate(z_arr):
        dg, _ = pert.cosmological_perturbation(k_ref, z)
        d_vals[i] = max(dg, 1e-10)
    idx0 = np.argmin(np.abs(z_arr))
    d0 = d_vals[idx0]
    if d0 <= 0:
        d0 = 1.0
    return d0 / np.maximum(d_vals, 1e-30)


def _transfer_simple(k_mpc: np.ndarray, k_eq: float = 0.01) -> np.ndarray:
    """Simple transfer function T(k) (matter domination). k_mpc in 1/Mpc."""
    k = np.maximum(np.asarray(k_mpc), 1e-8)
    # T(k) ~ 1 at k << k_eq, ~ (k_eq/k)^2 at k >> k_eq
    return 1.0 / (1.0 + (k / k_eq) ** 2) ** 0.5


def _tophat_filter(k_mpc: np.ndarray, R_mpc: float) -> np.ndarray:
    """Top-hat filter W(kR) = 3*(sin(kR)-kR*cos(kR))/(kR)^3."""
    x = np.maximum(k_mpc * R_mpc, 1e-20)
    return 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)


def sigma8(
    z: float = 0.0,
    cosmology: Optional[Any] = None,
    sigma8_z0_ref: float = 0.810,
    n_z: int = 100,
    bulk_seed: Optional[Any] = None,
    **kwargs: Any,
) -> float:
    """
    Amplitude of matter fluctuations at R = 8 h⁻¹ Mpc.

    σ₈ from HQIV growth factor D(z) with f(φ) and curvature imprint.
    When bulk_seed is provided, H₀ from bulk is used for R = 8 h⁻¹ Mpc.
    """
    cosmo = _get_cosmology(cosmology)
    bg = _get_background(bulk_seed, cosmology)
    H0_km_s_Mpc = bg["H0_km_s_Mpc"]
    R8_MPC_local = 8.0 / (H0_km_s_Mpc / 100.0)
    z_arr = np.linspace(0.0, max(z, 0.01), max(n_z, 2))
    D = _growth_factor_hqiv(z_arr, cosmo)
    # D(z) at requested z (interpolate)
    if z <= z_arr[0]:
        D_z = D[0]
    elif z >= z_arr[-1]:
        D_z = D[-1]
    else:
        D_z = np.interp(z, z_arr, D)

    # P(k) = A_s (k/k_pivot)^(n_s-1) T²(k) D²(z). σ_R² = (1/2π²) ∫ P(k) W²(kR) k² dk
    k_min, k_max = 1e-5, 1e2
    n_k = 400
    k_mpc = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    T_k = _transfer_simple(k_mpc)
    W = _tophat_filter(k_mpc, R8_MPC_local)
    # Normalize A_s so that at z=0, σ8 = sigma8_z0_ref
    D0 = _growth_factor_hqiv(np.array([0.0]), cosmo)[0]
    P0 = (k_mpc / K_PIVOT) ** (N_S - 1.0) * (T_k**2) * (D0**2)
    integrand0 = P0 * (W**2) * (k_mpc**2)
    sigma_sq_0 = _trapz(integrand0, k_mpc) / (2.0 * np.pi**2)
    if sigma_sq_0 <= 0:
        return 0.0
    A_s_norm = (sigma8_z0_ref**2) / sigma_sq_0

    P_z = A_s_norm * (k_mpc / K_PIVOT) ** (N_S - 1.0) * (T_k**2) * (D_z**2)
    integrand = P_z * (W**2) * (k_mpc**2)
    sigma_sq = _trapz(integrand, k_mpc) / (2.0 * np.pi**2)
    return float(np.sqrt(max(sigma_sq, 0.0)))


def universe_evolver(
    z_start: float = 1100.0,
    z_end: float = 0.0,
    n_steps: int = 500,
    cosmology: Optional[Any] = None,
    bulk_seed: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evolve universe from z_start (e.g. recombination) to z_end (now).

    When bulk_seed is provided (from HQIV bulk.py), background Ω_k and lapse use that
    as the authoritative seed until baryogenesis complete. Otherwise seeds from lattice.
    Returns z_grid, a_grid, D(z), f(z) for use in LOS and C_ℓ.
    """
    cosmo = _get_cosmology(cosmology)
    bg = _get_background(bulk_seed, cosmology)
    z_grid = np.linspace(z_end, z_start, n_steps)
    a_grid = 1.0 / (1.0 + z_grid)
    D = _growth_factor_hqiv(z_grid, cosmo)

    pert = HQIVPerturbations(cosmo)
    f_grid = np.zeros_like(z_grid)
    for i, z in enumerate(z_grid):
        _, f_grid[i] = pert.cosmological_perturbation(0.1, z)

    return {
        "z": z_grid,
        "a": a_grid,
        "D": D,
        "f_lapse": f_grid,
        "Omega_k_true": bg["Omega_true_k"],
        "lapse_compression": bg["lapse_compression"],
        "z_start": z_start,
        "z_end": z_end,
    }


def c_ell_spectrum(
    spectrum_type: str = "TT",
    max_ell: int = 2000,
    cosmology: Optional[Any] = None,
    ell_pivot: float = 30.0,
    amplitude: float = 5800.0,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    C_ℓ multipole power spectrum (phenomenological HQIV template).

    spectrum_type: 'TT', 'EE', 'TE', 'BB'. TT uses Sachs-Wolfe + first acoustic
    peak shape; EE/TE/BB scaled from TT. Returns (ell, C_ell) in μK².
    """
    ell = np.arange(2, max_ell + 1, dtype=float)
    # C_ℓ^TT ∝ 1/(ell(ell+1)) * (1 + first peak); amplitude in μK²
    sw = 1.0 / (ell * (ell + 1))
    first_peak_ell = 220.0
    peak = 1.0 + 2.5 * np.exp(-((ell - first_peak_ell) ** 2) / (2.0 * 80.0**2))
    C_tt = amplitude * sw * peak
    C_tt = np.maximum(C_tt, 1e-10)

    if spectrum_type.upper() == "TT":
        return ell, C_tt
    if spectrum_type.upper() == "EE":
        return ell, C_tt * 0.02  # EE ~ 2% of TT
    if spectrum_type.upper() == "TE":
        return ell, np.sqrt(C_tt * C_tt * 0.02) * 0.5  # TE correlation
    if spectrum_type.upper() == "BB":
        return ell, C_tt * 1e-4  # lensing B-mode level
    return ell, C_tt


def line_of_sight_isw_rees_sciama(
    ell: np.ndarray,
    z_range: Tuple[float, float] = (0.0, 1100.0),
    n_z: int = 200,
    cosmology: Optional[Any] = None,
    bulk_seed: Optional[Any] = None,
    amplitude: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Line-of-sight: ISW + Rees–Sciama contribution to C_ℓ (additive ΔC_ℓ).

    Uses HQIV growth D(z) and f(φ). When bulk_seed is provided, background uses bulk.
    """
    cosmo = _get_cosmology(cosmology)
    ev = universe_evolver(
        z_start=z_range[1],
        z_end=z_range[0],
        n_steps=n_z,
        cosmology=cosmo,
        bulk_seed=bulk_seed,
    )
    ell = np.asarray(ell, dtype=float)
    ell = np.maximum(ell, 2.0)
    D_mean = float(np.mean(ev["D"]))
    f_mean = float(np.mean(ev["f_lapse"]))
    delta_cl = amplitude * (D_mean * f_mean) / (ell * (ell + 1))
    return delta_cl


def add_kinematic_dipole(
    t_map_muK: np.ndarray,
    n_side: int,
    v_km_s: float,
    gal_l_deg: float = 264.0,
    gal_b_deg: float = 48.0,
    T_cmb_muK: float = T_CMB_MUK,
) -> np.ndarray:
    """
    Add kinematic dipole to a temperature map (μK) by boosting the reference frame.

    Simulates the low-ℓ (dipole) region from observer velocity: ΔT/T = (v/c) cos θ,
    so ΔT_muK = T_cmb_muK * (v_km_s / c_km_s) * cos(θ) where θ is angle between
    pixel and velocity apex. Uses galactic (l, b) for apex; Planck dipole ≈
    (l=264°, b=48°, v≈370 km/s). Returns a new map (does not modify in place).
    """
    try:
        import healpy as hp
    except ImportError as e:
        raise ImportError(
            "healpy is required for add_kinematic_dipole. "
            "Install with: pip install pyhqiv[cosmology]"
        ) from e
    c_km_s = 299792.458
    theta_apex = np.pi / 2.0 - np.radians(gal_b_deg)
    phi_apex = np.radians(gal_l_deg)
    apex_vec = np.array(hp.ang2vec(theta_apex, phi_apex), dtype=float)
    npix = hp.nside2npix(n_side)
    # Vectorized: pixel directions (3, npix) from healpy, then cos(θ) = apex · pix
    pix_vecs = np.asarray(hp.pix2vec(n_side, np.arange(npix)))  # (3, npix)
    cos_theta = np.dot(apex_vec, pix_vecs)  # (npix,)
    out = np.asarray(t_map_muK, dtype=float).copy()
    out += T_cmb_muK * (v_km_s / c_km_s) * cos_theta
    return out


def full_sky_healpy_map(
    n_side: int = 256,
    map_type: str = "T",
    include_isw_rees_sciama: bool = True,
    cosmology: Optional[Any] = None,
    bulk_seed: Optional[Any] = None,
    max_ell: int = 1500,
    frame_velocity_km_s: Optional[float] = None,
    frame_gal_l_deg: float = 264.0,
    frame_gal_b_deg: float = 48.0,
    **kwargs: Any,
) -> Any:
    """
    Full-sky HEALPix map (T and/or Q, U) with lapse and secondaries.

    Multipoles run out to max_ell (default 1500). If frame_velocity_km_s is set,
    adds the kinematic dipole to the T map so the low-ℓ region is generated
    (accelerated reference frame). Direction in galactic (l, b) degrees.
    """
    try:
        import healpy as hp
    except ImportError as e:
        raise ImportError(
            "healpy is required for full_sky_healpy_map. "
            "Install with: pip install pyhqiv[cosmology]"
        ) from e

    max_ell = max(int(max_ell), 1500)
    ell_tt, c_tt = c_ell_spectrum("TT", max_ell=max_ell, cosmology=cosmology)
    if include_isw_rees_sciama:
        delta_cl = line_of_sight_isw_rees_sciama(ell_tt, cosmology=cosmology, bulk_seed=bulk_seed)
        c_tt = c_tt + delta_cl

    # Build C_ℓ array for healpy (indexed by ell from 0 to max_ell)
    n_ell = max_ell + 1
    cl_arr = np.zeros(n_ell)
    cl_arr[2:] = c_tt[: n_ell - 2]

    if map_type.upper() == "T":
        t_map = hp.synfast(cl_arr, n_side, verbose=False)
        if frame_velocity_km_s is not None and frame_velocity_km_s != 0:
            t_map = add_kinematic_dipole(
                t_map,
                n_side,
                frame_velocity_km_s,
                gal_l_deg=frame_gal_l_deg,
                gal_b_deg=frame_gal_b_deg,
            )
        return t_map
    if map_type.upper() in ("QU", "Q", "U"):
        ell_ee, c_ee = c_ell_spectrum("EE", max_ell=max_ell, cosmology=cosmology)
        ell_bb, c_bb = c_ell_spectrum("BB", max_ell=max_ell, cosmology=cosmology)
        cl_ee = np.zeros(n_ell)
        cl_bb = np.zeros(n_ell)
        cl_ee[2:] = c_ee[: n_ell - 2]
        cl_bb[2:] = c_bb[: n_ell - 2]
        te, eb, tb = np.zeros(n_ell), np.zeros(n_ell), np.zeros(n_ell)
        t_map, q_map, u_map = hp.synfast([cl_arr, cl_ee, cl_bb, te, eb, tb], n_side, verbose=False)
        if frame_velocity_km_s is not None and frame_velocity_km_s != 0:
            t_map = add_kinematic_dipole(
                t_map,
                n_side,
                frame_velocity_km_s,
                gal_l_deg=frame_gal_l_deg,
                gal_b_deg=frame_gal_b_deg,
            )
        if map_type.upper() == "QU":
            return t_map, q_map, u_map
        if map_type.upper() == "Q":
            return q_map
        return u_map
    t_map = hp.synfast(cl_arr, n_side, verbose=False)
    if frame_velocity_km_s is not None and frame_velocity_km_s != 0:
        t_map = add_kinematic_dipole(
            t_map,
            n_side,
            frame_velocity_km_s,
            gal_l_deg=frame_gal_l_deg,
            gal_b_deg=frame_gal_b_deg,
        )
    return t_map


def hqiv_cmb(
    n_side: int = 256,
    max_ell: int = 1500,
    include_polarization: bool = True,
    cosmology: Optional[Any] = None,
    bulk_seed: Optional[Any] = None,
    frame_velocity_km_s: Optional[float] = None,
    frame_gal_l_deg: float = 264.0,
    frame_gal_b_deg: float = 48.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run full HQIV CMB pipeline: C_ℓ → optional map (phenomenological).

    When bulk_seed is provided (from pyhqiv.bulk_seed.get_bulk_seed()), the background
    Ω_k and H₀ are taken from HQIV bulk.py (authoritative until baryogenesis complete).
    Multipoles run out to max_ell (default 1500, minimum 1500). Set frame_velocity_km_s
    to add the kinematic dipole (e.g. 370 for Solar System rest frame).
    """
    max_ell = max(int(max_ell), 1500)
    cosmo = _get_cosmology(cosmology)
    ell_tt, c_tt = c_ell_spectrum("TT", max_ell=max_ell, cosmology=cosmo)
    sigma8_z0 = sigma8(0.0, cosmology=cosmo, bulk_seed=bulk_seed)
    result = {
        "ell": ell_tt,
        "C_ell_TT": c_tt,
        "sigma8": sigma8_z0,
        "cosmology": cosmo,
    }
    if include_polarization:
        _, c_ee = c_ell_spectrum("EE", max_ell=max_ell, cosmology=cosmo)
        _, c_te = c_ell_spectrum("TE", max_ell=max_ell, cosmology=cosmo)
        _, c_bb = c_ell_spectrum("BB", max_ell=max_ell, cosmology=cosmo)
        result["C_ell_EE"] = c_ee
        result["C_ell_TE"] = c_te
        result["C_ell_BB"] = c_bb
    try:
        t_map = full_sky_healpy_map(
            n_side=n_side,
            map_type="T",
            include_isw_rees_sciama=True,
            cosmology=cosmo,
            bulk_seed=bulk_seed,
            max_ell=max_ell,
            frame_velocity_km_s=frame_velocity_km_s,
            frame_gal_l_deg=frame_gal_l_deg,
            frame_gal_b_deg=frame_gal_b_deg,
        )
        result["T_map"] = t_map
        result["n_side"] = n_side
    except ImportError:
        result["T_map"] = None
        result["n_side"] = None
    return result
