"""
HQIV CMB: full T_Pl → now pipeline in one place.

- HQIVCMBMap: axiom-pure, respects Ω_k^true = +0.0098 (curved LOS, ISW, growth_to_sigma8).
- run_hqiv_cmb_to_map: thin wrapper over cosmology_full (phenomenological).

Why CLASS-HQIV gets peaks and we didn't (before the fix):
  CLASS computes C_ℓ in harmonic space with j_ℓ(k χ). We now do the same.

Lapse order (run in real time, then z_shift at observe):
  CLASS runs the full ~52 Gyr (wall-clock) dynamics, then you \"see\" the CMB; the
  lapse/z-shift is applied only when mapping to observer (apparent age 13.8 Gyr).
  Here we do the same: dynamics (transfer, χ_rec) use real time — no lapse in
  T_recomb or χ_rec. Apply z_shift only at the observe step: use result[\"lapse_compression\"]
  to get observer multipole ℓ_eff = ℓ / lapse_compression for Planck comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pyhqiv.lattice import DiscreteNullLattice
from pyhqiv.perturbations import HQIVPerturbations
from pyhqiv.cosmology.background import HQIVCosmology

from pyhqiv.constants import T_CMB_K, T_CMB_MUK, Z_RECOMB

try:
    import healpy as hp
except ImportError:
    hp = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from scipy.special import spherical_jn
except ImportError:
    spherical_jn = None


class HQIVCMBMap:
    """
    Full T_Pl → now pipeline. Produces clean multipole + σ₈ + map.

    Orchestrator that:
    1. Runs dynamics in real (wall-clock) time — no lapse in transfer or χ_rec.
    2. Primordial power, transfer T(k), comoving distance χ_rec (uncompressed).
    3. C_ℓ in harmonic space: C_ℓ ∝ ∫ (dk/k) P(k) T(k)² j_ℓ(k χ_rec)².
    4. Builds map from C_ℓ via synfast; adds ISW; σ₈.
    5. Z-shift / lapse applied only at observe step (lapse_compression in result for ℓ_eff = ℓ / lapse).

    boost_scale: ISW/acceleration-artifact dipole is multiplied by this (default 0.1) so the
    map gradient isn't extreme and acoustic structure can show; set to 1.0 for raw boost.
    normalize_map_rms: if set (default 70 μK), scale T_map_muK so std = this for sensible
    mollview; set to None to keep raw observer-scaled map. Map is δT (fluctuations), so
    negative values are physical (cold spots).
    """

    def __init__(self, nside: int = 1024, gamma: float = 0.40, lmax: int = 2500) -> None:
        self.lattice = DiscreteNullLattice(gamma=gamma)
        self.cosmo = HQIVCosmology(gamma=gamma)
        self.pert = HQIVPerturbations(background=self.cosmo, gamma=gamma)
        self.nside = nside
        self.lmax = lmax

    def _cl_from_harmonic_integral(
        self,
        k: np.ndarray,
        Pk_prim: np.ndarray,
        delta_T_transfer: np.ndarray,
        chi_rec: float,
        lmax: int,
    ) -> np.ndarray:
        """
        C_ℓ from harmonic-space integral (CLASS-like): C_ℓ = (T_CMB_μK)² (4π) ∫ (dk/k) P(k) T(k)² j_ℓ(k χ_rec)².
        This produces acoustic peaks; using j0 only would not.
        """
        if spherical_jn is None:
            raise ImportError("scipy.special.spherical_jn required for _cl_from_harmonic_integral")
        ell = np.arange(0, lmax + 1, dtype=float)
        cl = np.zeros_like(ell)
        # dk/k integrand: use log spacing so d(ln k) = dk/k
        lnk = np.log(np.maximum(k, 1e-30))
        dlnk = np.diff(lnk)
        for i in range(1, int(lmax) + 1):
            j_ell = spherical_jn(i, k * chi_rec)
            j_ell = np.maximum(np.asarray(j_ell, dtype=float), 0.0)  # j_ℓ²
            integrand = Pk_prim * (delta_T_transfer ** 2) * (j_ell ** 2)
            # ∫ (dk/k) f = ∫ f d(ln k)
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if _trapz is None:
                from scipy.integrate import trapezoid as _trapz
            cl[i] = _trapz(integrand, lnk)
        cl[0] = cl[1]  # avoid ℓ=0 singularity
        # (4π) and (T_CMB_μK)² to get μK²
        cl *= (4.0 * np.pi) * (T_CMB_MUK ** 2)
        return cl

    def run_from_T_Pl_to_now(
        self,
        boost_scale: float = 0.1,
        normalize_map_rms: Optional[float] = 70.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if hp is None:
            raise ImportError("healpy is required for HQIVCMBMap.run_from_T_Pl_to_now")

        # 1. Background (perfect)
        bg = self.lattice.evolve_to_cmb(T0_K=T_CMB_K)

        # 2. Primordial power from lattice invariant
        k = np.logspace(-5, 0, 800)
        Pk_prim = self.lattice.primordial_power_from_invariant(k)

        # 3. Lapse + curvature-aware transfer (fluid.py-style: f(φ), coherent fluid)
        delta_T_transfer = self.pert.cosmological_transfer(
            k, z_recomb=Z_RECOMB, omega_k=self.cosmo.Ok0
        )

        # 4. Comoving distance to recombination in real time (no lapse: dynamics first, z_shift at observe)
        chi_rec = self.cosmo.comoving_distance(Z_RECOMB, omega_k=self.cosmo.Ok0)

        # 5. C_ℓ in harmonic space (CLASS-like: j_ℓ(k χ_rec) so peaks appear)
        cl = self._cl_from_harmonic_integral(
            k, Pk_prim, delta_T_transfer, chi_rec, self.lmax
        )

        # 6. Map from C_ℓ (synfast); add ISW dipole (acceleration artifact) scaled so CMB structure shows
        cl_arr = np.zeros(int(self.lmax) + 1)
        cl_arr[: len(cl)] = cl
        cl_arr[0] = 0.0
        cl_arr[1] = 0.0
        delta_T = hp.synfast(cl_arr, self.nside, lmax=self.lmax)
        # Boost/ISW as acceleration artifact: scale down so gradient isn't extreme and peaks can show
        isw_amp = self.pert.isw_from_peculiar_velocity(0.0, 0.0, omega_k=self.cosmo.Ok0)
        npix = hp.nside2npix(self.nside)
        for ipix in range(npix):
            theta, phi = hp.pix2ang(self.nside, ipix)
            delta_T[ipix] += boost_scale * isw_amp * T_CMB_MUK * (1.0 + 0.3 * np.cos(theta))

        # 7. σ₈ from lapse-corrected growth
        sigma8 = self.pert.growth_to_sigma8(omega_k=self.cosmo.Ok0) * np.sqrt(
            np.mean(Pk_prim)
        )

        # Z-shift at observe step: map in observer-frame (lapse-compressed) so scale matches what we see
        lapse_comp = float(bg.get("lapse_compression", 1.0))
        delta_T *= 1.0 / max(lapse_comp, 1.0)  # observer-frame δT in μK

        # Optional: scale map to target RMS (μK) for sensible mollview when C_ℓ amplitude is off
        if normalize_map_rms is not None:
            rms = np.std(delta_T)
            if rms > 1e-6:
                delta_T *= normalize_map_rms / rms

        # Map is δT (fluctuations), not T; negative values are physical (cold spots)
        return {
            "T_map_muK": delta_T,
            "Cl_TT": cl,
            "sigma8": float(sigma8),
            "Omega_k_true": float(self.cosmo.Ok0),
            "background": bg,
            "lapse_compression": lapse_comp,
            "age_wall_Gyr": float(bg.get("age_wall_Gyr", 0.0)),
            "age_apparent_Gyr": float(bg.get("age_apparent_Gyr", 0.0)),
        }

    def observer_multipoles(
        self, result: Dict[str, Any]
    ) -> tuple:
        """
        Z-shift at observe step: return (ell_eff, D_ell_eff) for observer frame.
        ℓ_eff = ℓ / lapse_compression so the spectrum can be compared to Planck.
        """
        ell = np.arange(len(result["Cl_TT"]), dtype=float)
        lapse = result.get("lapse_compression", 1.0)
        ell_eff = ell / max(lapse, 1e-10)
        d_ell = ell * (ell + 1) * result["Cl_TT"] / (2.0 * np.pi)
        return ell_eff, np.maximum(d_ell, 1e-20)

    def plot_multipole(
        self,
        result: Dict[str, Any],
        show: bool = True,
        out_path: Optional[str] = None,
    ) -> None:
        if plt is None:
            raise ImportError("matplotlib is required for plot_multipole")
        ell = np.arange(len(result["Cl_TT"]))
        d_ell = np.maximum(ell * (ell + 1) * result["Cl_TT"] / (2.0 * np.pi), 1e-10)
        plt.figure(figsize=(12, 7))
        plt.loglog(
            ell,
            d_ell,
            label=f"HQIV (Ω_k = {result['Omega_k_true']:.4f}, σ₈ = {result['sigma8']:.3f})",
        )
        plt.xlabel("Multipole ℓ")
        plt.ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$  [μK²]")
        plt.title("HQIV CMB Multipole — T_Pl to now (clean)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        if out_path:
            plt.savefig(out_path, dpi=120)
        if show:
            plt.show()
        else:
            plt.close()


def run_hqiv_cmb_to_map(
    n_side: int = 256,
    max_ell: int = 1500,
    include_polarization: bool = True,
    cosmology: Optional[HQIVCosmology] = None,
    bulk_seed: Optional[Any] = None,
    include_isw_rees_sciama: bool = True,
    frame_velocity_km_s: Optional[float] = None,
    frame_gal_l_deg: float = 264.0,
    frame_gal_b_deg: float = 48.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run HQIV CMB pipeline: C_ℓ → full-sky map in μK (cosmology_full wrapper).

    When bulk_seed is provided (from pyhqiv.bulk_seed.get_bulk_seed()), the pipeline
    uses HQIV bulk.py output as the authoritative seed until baryogenesis complete.
    """
    from pyhqiv import cosmology_full

    return cosmology_full.hqiv_cmb(
        n_side=n_side,
        max_ell=max(max_ell, 1500),
        include_polarization=include_polarization,
        cosmology=cosmology,
        bulk_seed=bulk_seed,
        frame_velocity_km_s=frame_velocity_km_s,
        frame_gal_l_deg=frame_gal_l_deg,
        frame_gal_b_deg=frame_gal_b_deg,
        **kwargs,
    )


__all__ = ["HQIVCMBMap", "run_hqiv_cmb_to_map"]
