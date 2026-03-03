"""
Redshift decomposition from cosmic birefringence (polarization twist β).

The HQIV framework links a polarization rotation angle β to the pure
recession / expansion redshift component z_rec via:

    Z_rec = exp(beta_rad / KAPPA_BETA) - 1

where beta_rad is the cosmic birefringence angle in radians and
KAPPA_BETA is an HQIV constant. The remaining apparent redshift is
attributed to horizon-lapse / mass-induced effects Z_lapse such that

    log(1 + Z_obs) = log(1 + Z_rec) + log(1 + Z_lapse) + Δlog N(z),

where Δlog N encodes the ADM lapse / horizon-compression term.

This module provides a user-facing API:

    from pyhqiv import HQIVCosmology
    from pyhqiv.polarization import decompose_redshift, RedshiftDecomposition

    cosmo = HQIVCosmology()
    result = decompose_redshift(z_obs=1100.0, beta=0.215, source_type="cmb")
    print(result.z_rec, result.z_lapse, result.true_recession_distance)
    result.plot_decomposition()
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from pyhqiv.constants import (
    H0_KM_S_MPC_PAPER,
    LAPSE_COMPRESSION_PAPER,
    SEC_PER_GYR,
)
from pyhqiv.cosmology import HQIVCosmology

try:  # Optional dependency: astropy for unit compatibility
    import astropy.units as u
except ImportError:  # pragma: no cover - astropy not required for core logic
    u = None  # type: ignore


Number = Union[float, np.floating]
ArrayLike = Union[Number, Sequence[Number], np.ndarray]


def _get_kappa_beta() -> float:
    """
    Retrieve KAPPA_BETA from pyhqiv.constants if available.

    The constant is defined in the HQIV paper; this helper avoids hard-coding
    the numerical value while remaining backwards-compatible with older
    versions of pyhqiv that may not yet expose it.
    """
    try:
        from pyhqiv import constants as _c  # type: ignore

        kappa = getattr(_c, "KAPPA_BETA")
        return float(kappa)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise AttributeError(
            "pyhqiv.constants.KAPPA_BETA is required for polarization-based "
            "redshift decomposition but was not found. Please update "
            "pyhqiv.constants to define KAPPA_BETA from the HQIV paper."
        ) from exc


def _ensure_cosmology(cosmology: Optional[HQIVCosmology]) -> HQIVCosmology:
    """Return a valid HQIVCosmology instance."""
    if cosmology is not None and isinstance(cosmology, HQIVCosmology):
        return cosmology
    return HQIVCosmology()


def _delta_log_n_term(
    z_obs: np.ndarray,
    cosmology: HQIVCosmology,
) -> np.ndarray:
    """
    Approximate Δlog N(z) from the cosmology lapse factor.

    We use the public HQIVCosmology.lapse_factor(z) and lapse_now as:

        Δlog N(z) = log[f(0) / f(z)],

    where f(z) is the local lapse factor at redshift z.
    """
    # Vectorize lapse_factor over z_obs while keeping the public scalar API.
    f_now = max(cosmology.lapse_now, 1e-30)

    def _f(z_val: float) -> float:
        return max(cosmology.lapse_factor(float(z_val)), 1e-30)

    f_vec = np.vectorize(_f, otypes=[float])(z_obs)
    return np.log(f_now / np.maximum(f_vec, 1e-30))


def _true_recession_distance_mpc(
    z_rec: np.ndarray,
    cosmology: HQIVCosmology,
) -> np.ndarray:
    """
    True recession distance χ(z_rec) in Mpc using HQIV comoving distance.

    Uses HQIVCosmology.comoving_distance, which already incorporates the
    curved background and the H0 derived from the HQIV age.
    """
    # Vectorize comoving_distance over z_rec
    def _chi(z_val: float) -> float:
        return float(cosmology.comoving_distance(float(z_val)))

    return np.vectorize(_chi, otypes=[float])(z_rec)


def _to_quantity_mpc(distance_mpc: np.ndarray) -> Any:
    """
    Convert a float distance in Mpc to an astropy Quantity when available.

    Returns either a numpy array of float (Mpc) or a Quantity with physical units.
    """
    if u is None:
        return distance_mpc
    return distance_mpc * u.Mpc


@dataclass
class RedshiftDecomposition:
    """Container for birefringence-based redshift decomposition.

    Attributes:
        z_obs: Observed spectroscopic redshift.
        z_rec: Pure recession / expansion component inferred from β.
        z_lapse: Horizon-lapse / mass-induced component.
        z_mass: Alias for z_lapse.
        true_recession_distance: HQIV comoving distance for z_rec (Mpc or Quantity).
        implied_mass_factor: Effective mass amplification factor from lapse.
        beta_input_deg: Input cosmic birefringence angle (degrees), if provided.
        beta_predicted_from_z_rec: Predicted β (degrees) from z_rec.
        source_type: Source classification ('cmb', 'galaxy', 'quasar', 'cdg2_like', ...).
        sky_position: Optional sky coordinates (reserved for anisotropy extensions).
        z_obs_err: Optional observational uncertainty on z_obs.
        beta_err_deg: Optional observational uncertainty on β (degrees).
        mc_stats: Optional Monte-Carlo uncertainty summary (per-field std).
    """

    z_obs: np.ndarray
    z_rec: np.ndarray
    z_lapse: np.ndarray
    true_recession_distance: Any
    implied_mass_factor: np.ndarray
    beta_predicted_from_z_rec: np.ndarray
    z_mass: np.ndarray = field(init=False)
    beta_input_deg: Optional[np.ndarray] = None
    source_type: str = "generic"
    sky_position: Optional[Any] = None
    z_obs_err: Optional[np.ndarray] = None
    beta_err_deg: Optional[np.ndarray] = None
    mc_stats: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.z_obs = np.asarray(self.z_obs, dtype=float)
        self.z_rec = np.asarray(self.z_rec, dtype=float)
        self.z_lapse = np.asarray(self.z_lapse, dtype=float)
        self.z_mass = self.z_lapse
        self.implied_mass_factor = np.asarray(self.implied_mass_factor, dtype=float)
        self.beta_predicted_from_z_rec = np.asarray(
            self.beta_predicted_from_z_rec,
            dtype=float,
        )
        if self.beta_input_deg is not None:
            self.beta_input_deg = np.asarray(self.beta_input_deg, dtype=float)
        if self.z_obs_err is not None:
            self.z_obs_err = np.asarray(self.z_obs_err, dtype=float)
        if self.beta_err_deg is not None:
            self.beta_err_deg = np.asarray(self.beta_err_deg, dtype=float)

    # ------------------------------------------------------------------
    # Convenience representations
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        data = asdict(self)
        # astropy Quantity is not JSON-serialisable; store magnitude for dict
        trd = data.get("true_recession_distance")
        if u is not None and hasattr(trd, "to"):
            try:
                data["true_recession_distance"] = float(trd.to(u.Mpc).value)
            except Exception:
                pass
        return data

    def to_dataframe(self) -> Any:
        """Return a pandas.DataFrame with one row (or raise if pandas missing)."""
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required for RedshiftDecomposition.to_dataframe(). "
                "Install pandas or use .to_dict() instead."
            ) from exc
        data = self.to_dict()
        return pd.DataFrame([data])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_decomposition(self) -> Any:
        """Plot the fractional contribution of Z_rec and Z_lapse."""
        import matplotlib.pyplot as plt  # local import to avoid mandatory dependency

        z_rec_total = np.mean(self.z_rec)
        z_lapse_total = np.mean(self.z_lapse)
        z_total = z_rec_total + z_lapse_total
        if z_total <= 0:
            z_total = 1.0
        frac_rec = z_rec_total / z_total
        frac_lapse = z_lapse_total / z_total

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(
            ["Z_rec", "Z_lapse"],
            [frac_rec, frac_lapse],
            color=["tab:blue", "tab:orange"],
        )
        ax.set_ylabel("Fraction of total redshift")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Redshift decomposition (source_type={self.source_type})")
        return fig

    # ------------------------------------------------------------------
    # Named example for Perseus / CDG2-like systems
    # ------------------------------------------------------------------
    @classmethod
    def for_cdg2_example(cls) -> "RedshiftDecomposition":
        """Convenience constructor for the Perseus-cluster / CDG2-like example.

        Uses representative values:

        - z_obs ≈ 0.018 (Perseus cluster)
        - β ≈ 5.5e-4 deg
        - source_type = 'cdg2_like'

        The returned object encodes the SMBH-vs-diffuse diagnostic via
        implied_mass_factor and z_lapse.
        """
        result = decompose_redshift(
            z_obs=0.018,
            beta=5.5e-4,
            source_type="cdg2_like",
        )
        return result


def _broadcast_to_array(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    """Return x as a numpy array (or None)."""
    if x is None:
        return None
    return np.asarray(x, dtype=float)


def _monte_carlo_uncertainties(
    z_obs: np.ndarray,
    beta_deg: Optional[np.ndarray],
    z_obs_err: Optional[np.ndarray],
    beta_err_deg: Optional[np.ndarray],
    n_samples: int,
    source_type: str,
    cosmology: HQIVCosmology,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple Monte-Carlo error propagation for the decomposition.

    Draws Gaussian samples around z_obs and beta_deg (where errors provided)
    and recomputes the decomposition to estimate 1σ uncertainties.
    """
    rng = np.random.default_rng()
    n_samples = max(int(n_samples), 10)

    # Use scalar means for the MC; array-like inputs are averaged.
    z_mean = float(np.mean(z_obs))
    if z_obs_err is None:
        z_sigma = 0.0
    else:
        z_sigma = float(np.mean(np.abs(z_obs_err)))

    if beta_deg is None:
        beta_mean = None
        beta_sigma = 0.0
    else:
        beta_mean = float(np.mean(beta_deg))
        if beta_err_deg is None:
            beta_sigma = 0.0
        else:
            beta_sigma = float(np.mean(np.abs(beta_err_deg)))

    z_samples = (
        rng.normal(loc=z_mean, scale=z_sigma, size=n_samples)
        if z_sigma > 0
        else np.full(n_samples, z_mean, dtype=float)
    )
    if beta_mean is None:
        beta_samples = None
    else:
        beta_samples = (
            rng.normal(loc=beta_mean, scale=beta_sigma, size=n_samples)
            if beta_sigma > 0
            else np.full(n_samples, beta_mean, dtype=float)
        )

    z_rec_s = np.zeros(n_samples, dtype=float)
    z_lapse_s = np.zeros(n_samples, dtype=float)
    dist_s = np.zeros(n_samples, dtype=float)
    mass_factor_s = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        res_i = decompose_redshift(
            z_obs=float(z_samples[i]),
            beta=None if beta_samples is None else float(beta_samples[i]),
            source_type=source_type,
            cosmology=cosmology,
            monte_carlo_samples=0,
        )
        z_rec_s[i] = float(np.mean(res_i.z_rec))
        z_lapse_s[i] = float(np.mean(res_i.z_lapse))
        # true_recession_distance may be Quantity or float
        trd = res_i.true_recession_distance
        if u is not None and hasattr(trd, "to"):
            try:
                dist_s[i] = float(trd.to(u.Mpc).value)
            except Exception:
                dist_s[i] = float(np.mean(np.asarray(trd)))
        else:
            dist_s[i] = float(np.mean(np.asarray(trd)))
        mass_factor_s[i] = float(np.mean(res_i.implied_mass_factor))

    stats = {
        "z_rec_std": float(np.std(z_rec_s)),
        "z_lapse_std": float(np.std(z_lapse_s)),
        "true_recession_distance_std_Mpc": float(np.std(dist_s)),
        "implied_mass_factor_std": float(np.std(mass_factor_s)),
    }
    return stats, z_rec_s, z_lapse_s, dist_s, mass_factor_s


def decompose_redshift(
    z_obs: ArrayLike,
    beta: Optional[ArrayLike] = None,
    beta_err: Optional[ArrayLike] = None,
    z_obs_err: Optional[ArrayLike] = None,
    sky_position: Optional[Any] = None,
    source_type: str = "generic",
    cosmology: Optional[HQIVCosmology] = None,
    monte_carlo_samples: int = 0,
) -> RedshiftDecomposition:
    """Decompose an observed redshift into recession and lapse components.

    Args:
        z_obs: Observed spectroscopic redshift (dimensionless).
        beta: Cosmic birefringence angle β in degrees. If None, falls back
            to spectroscopic-only decomposition (z_rec ≈ z_obs, z_lapse ≈ 0)
            and issues a runtime warning.
        beta_err: 1σ uncertainty on β (degrees); optional.
        z_obs_err: 1σ uncertainty on z_obs; optional.
        sky_position: Optional sky coordinates (e.g. (ra, dec) or (lon, lat)).
        source_type: Source classification label.
        cosmology: Optional HQIVCosmology instance; if None, a default is used.
        monte_carlo_samples: If > 0, perform simple Monte-Carlo error
            propagation with the given number of samples.

    Returns:
        RedshiftDecomposition instance containing the decomposition and
        derived quantities (recession distance, implied mass factor, etc.).
    """
    cosmo = _ensure_cosmology(cosmology)
    z_obs_arr = np.asarray(z_obs, dtype=float)
    beta_arr = _broadcast_to_array(beta)
    beta_err_arr = _broadcast_to_array(beta_err)
    z_obs_err_arr = _broadcast_to_array(z_obs_err)

    kappa_beta = _get_kappa_beta()

    if beta_arr is None:
        import warnings

        warnings.warn(
            "beta is None: falling back to spectroscopic-only redshift "
            "decomposition with z_rec ≈ z_obs and z_lapse ≈ 0.",
            RuntimeWarning,
        )
        z_rec_arr = z_obs_arr.copy()
        z_lapse_arr = np.zeros_like(z_obs_arr)
        beta_input_deg = None
    else:
        beta_rad = np.radians(beta_arr)
        z_rec_arr = np.exp(beta_rad / kappa_beta) - 1.0
        beta_input_deg = beta_arr
        # Compute lapse component in log-additive space.
        one_plus_z_obs = 1.0 + z_obs_arr
        one_plus_z_rec = 1.0 + z_rec_arr
        delta_log_n = _delta_log_n_term(z_obs_arr, cosmo)
        log_one_plus_z_lapse = (
            np.log(np.maximum(one_plus_z_obs, 1e-30))
            - np.log(np.maximum(one_plus_z_rec, 1e-30))
            - delta_log_n
        )
        one_plus_z_lapse = np.exp(log_one_plus_z_lapse)
        z_lapse_arr = np.maximum(one_plus_z_lapse - 1.0, 0.0)

    # True recession distance χ(z_rec) in Mpc (or astropy Quantity).
    dist_mpc = _true_recession_distance_mpc(z_rec_arr, cosmo)
    dist = _to_quantity_mpc(dist_mpc)

    # Effective mass amplification factor from lapse.
    implied_mass_factor = 1.0 + z_lapse_arr

    # Predicted β from z_rec (consistency check).
    beta_pred_rad = kappa_beta * np.log(1.0 + np.maximum(z_rec_arr, 0.0))
    beta_pred_deg = np.degrees(beta_pred_rad)

    mc_stats: Optional[Dict[str, float]] = None
    if monte_carlo_samples > 0:
        mc_stats, _, _, _, _ = _monte_carlo_uncertainties(
            z_obs=z_obs_arr,
            beta_deg=beta_arr,
            z_obs_err=z_obs_err_arr,
            beta_err_deg=beta_err_arr,
            n_samples=monte_carlo_samples,
            source_type=source_type,
            cosmology=cosmo,
        )

    return RedshiftDecomposition(
        z_obs=z_obs_arr,
        z_rec=z_rec_arr,
        z_lapse=z_lapse_arr,
        true_recession_distance=dist,
        implied_mass_factor=implied_mass_factor,
        beta_predicted_from_z_rec=beta_pred_deg,
        beta_input_deg=beta_input_deg,
        source_type=source_type,
        sky_position=sky_position,
        z_obs_err=z_obs_err_arr,
        beta_err_deg=beta_err_arr,
        mc_stats=mc_stats,
    )


__all__ = [
    "RedshiftDecomposition",
    "decompose_redshift",
]

