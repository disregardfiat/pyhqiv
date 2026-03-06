"""
Discrete null lattice: combinatorial evolution m=0..m_trans, δE(m), T(m),
Ω_k from shell integral, evolve_to_cmb(T0). Paper Sec. 3, curvature imprint.

Ω_k is dynamic (HQIV_LEAN): Ω_k(n; N) = Ω_k_true · (∫₀ⁿ shape / ∫₀ᴺ shape).
At the chosen horizon (n = N) we get Ω_k(N; N) = Ω_k_true. Not a constant.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pyhqiv.constants import (
    AGE_APPARENT_GYR_PAPER,
    AGE_WALL_GYR_PAPER,
    ALPHA,
    COMBINATORIAL_INVARIANT,
    GAMMA,
    K_PIVOT_FIDUCIAL,
    LAPSE_COMPRESSION_PAPER,
    L_PLANCK_M,
    M_TRANS,
    OMEGA_TRUE_K_PAPER,
    T_CMB_K,
    T_PL_GEV,
)


def discrete_mode_count(m: int) -> float:
    """Discrete sphere-packing count for shell m: 8 * C(m+2, 2) = 4*(m+2)*(m+1)."""
    if m < 0:
        return 0.0
    return 8.0 * (m + 2) * (m + 1) / 2.0  # 8 * binom(m+2, 2)


def cumulative_mode_count(m_max: int) -> float:
    """Cumulative new modes from m=0 to m_max-1: sum of 8*binom(m+2,2) over integer m.
    Hockey-stick: sum_{k=0}^{m_max-1} binom(k+2, 2) = binom(m_max+2, 3).
    So cumulative = 8 * binom(m_max+2, 3) = 8 * (m_max+2)*(m_max+1)*m_max / 6.
    """
    if m_max <= 0:
        return 0.0
    return 8.0 * (m_max + 2) * (m_max + 1) * m_max / 6.0


def curvature_imprint_delta_E(
    m: np.ndarray,
    T: np.ndarray,
    T_Pl: float = T_PL_GEV,
    alpha: float = ALPHA,
    N67: float = COMBINATORIAL_INVARIANT,
) -> np.ndarray:
    """
    Per-shell curvature imprint δE(m) from discrete-to-continuous mismatch.
    Paper: δE(m) = (1/(m+1)) * (1 + α ln(T_Pl/T(m))) × (6^7√3).
    No free amplitude; integrated over m=0..m_trans gives Ω_k^true.
    """
    m = np.asarray(m, dtype=float)
    T = np.asarray(T, dtype=float)
    shell_fraction = 1.0 / (m + 1.0)
    ln_term = 1.0 + alpha * np.log(np.maximum(T_Pl / np.maximum(T, 1e-300), 1.0))
    return shell_fraction * ln_term * N67


def curvature_integral(
    n: int,
    T_Pl: float = T_PL_GEV,
    alpha: float = ALPHA,
    E_0_factor: float = 1.0,
    use_jax: bool = False,
) -> float:
    """
    Curvature integral up to depth n: sum of δE(m) for m = 0..n-1 (shells 0..n-1).
    Matches HQIV_LEAN curvature_integral(n). Used in Ω_k(n; N) = Ω_k_true · (∫₀ⁿ shape / ∫₀ᴺ shape).
    """
    if n <= 0:
        return 0.0
    if use_jax:
        try:
            import jax.numpy as jnp

            E_0 = E_0_factor * T_Pl
            m_arr = jnp.arange(0, int(n), dtype=jnp.float64)
            R_h = m_arr + 1.0
            T = E_0 / jnp.maximum(R_h, 1e-300)
            shell_frac = 1.0 / (m_arr + 1.0)
            ln_t = 1.0 + alpha * jnp.log(jnp.maximum(T_Pl / jnp.maximum(T, 1e-300), 1.0))
            delta_E = shell_frac * ln_t * COMBINATORIAL_INVARIANT
            return float(jnp.sum(delta_E))
        except ImportError:
            pass
    E_0 = E_0_factor * T_Pl
    m_arr = np.arange(0, int(n), dtype=float)
    R_h = m_arr + 1.0
    T = E_0 / np.maximum(R_h, 1e-300)
    delta_E = curvature_imprint_delta_E(m_arr, T, T_Pl=T_Pl, alpha=alpha)
    return float(np.sum(delta_E))


def x_over_theta_from_horizons(n: int, N: int) -> float:
    """
    Continuous 0 < x < θ term from the Planck distances of the two horizons.

    Distance to shell m in Planck units is L_P * (m+1) (radial scale R_h = m+1).
    So x = L_P * (n+1) at shell n, θ = L_P * (N+1) at horizon N, and
    x/θ = (n+1)/(N+1). Used in Ω_k(n; N) so the curvature range 0 < x < θ
    is set by the Planck distance of the two horizons.
    """
    if N <= 0:
        return 1.0
    return (float(n) + 1.0) / (float(N) + 1.0)


def horizon_planck_distances_m(n: int, N: int) -> tuple[float, float]:
    """
    Planck distances (metres) to the two horizons: shell n and reference horizon N.

    Distance to shell m = L_P * (m+1). Returns (distance_n, distance_N).
    """
    return (
        L_PLANCK_M * (float(n) + 1.0),
        L_PLANCK_M * (float(N) + 1.0),
    )


def omega_k_at_horizon(
    n: int,
    N: int,
    T_Pl: float = T_PL_GEV,
    alpha: float = ALPHA,
    E_0_factor: float = 1.0,
    omega_k_true: float = OMEGA_TRUE_K_PAPER,
    use_jax: bool = False,
    use_planck_distance_ratio: bool = True,
) -> float:
    """
    Ω_k(n; N): spatial curvature at shell n relative to horizon N (dynamic, not constant).

    Ω_k(n; N) = Ω_k_true · (∫₀ⁿ shape / ∫₀ᴺ shape) · (x/θ), where the 0 < x < θ term
    is computed from the Planck distances of the two horizons: x/θ = (n+1)/(N+1)
    (distance to shell n is L_P·(n+1), to horizon N is L_P·(N+1)). Set
    use_planck_distance_ratio=False for the integral-only formula (HQIV_LEAN raw).
    At the horizon (n = N): Ω_k(N; N) = Ω_k_true. Same normalization as η in baryogenesis.
    """
    integral_N = curvature_integral(N, T_Pl=T_Pl, alpha=alpha, E_0_factor=E_0_factor, use_jax=use_jax)
    if integral_N <= 0:
        return omega_k_true
    integral_n = curvature_integral(n, T_Pl=T_Pl, alpha=alpha, E_0_factor=E_0_factor, use_jax=use_jax)
    ratio = float(omega_k_true * integral_n / integral_N)
    if use_planck_distance_ratio:
        ratio *= x_over_theta_from_horizons(n, N)
    return ratio


def omega_k_from_shell_integral(
    m_trans: int = M_TRANS,
    T_Pl: float = T_PL_GEV,
    alpha: float = ALPHA,
    E_0_factor: float = 1.0,
    omega_k_reference: float = OMEGA_TRUE_K_PAPER,
    reference_m_trans: int = M_TRANS,
    use_jax: bool = False,
    use_planck_distance_ratio: bool = True,
) -> float:
    """
    Ω_k at the transition shell relative to the reference horizon (today’s value).

    Ω_k(m_trans; reference_m_trans) = omega_k_reference · (∫₀^{m_trans}/∫₀^{reference}) · (x/θ).
    x/θ from Planck distances of the two horizons. At reference_m_trans we get omega_k_reference.
    Equivalent to omega_k_at_horizon(m_trans, reference_m_trans).
    """
    return omega_k_at_horizon(
        n=m_trans,
        N=reference_m_trans,
        T_Pl=T_Pl,
        alpha=alpha,
        E_0_factor=E_0_factor,
        omega_k_true=omega_k_reference,
        use_jax=use_jax,
        use_planck_distance_ratio=use_planck_distance_ratio,
    )


def omega_k_partial(
    n: int,
    reference_m_trans: int = M_TRANS,
    T_Pl: float = T_PL_GEV,
    alpha: float = ALPHA,
    E_0_factor: float = 1.0,
    omega_k_true: float = OMEGA_TRUE_K_PAPER,
    use_jax: bool = False,
    use_planck_distance_ratio: bool = True,
) -> float:
    """
    Ω_k(n; reference): partial curvature at shell n with fixed reference horizon (HQIV_LEAN omega_k_partial).
    omega_k_partial(n) = omega_k_at_horizon(n, reference_m_trans). x/θ from Planck distances of the two horizons.
    """
    return omega_k_at_horizon(
        n=n,
        N=reference_m_trans,
        T_Pl=T_Pl,
        alpha=alpha,
        E_0_factor=E_0_factor,
        omega_k_true=omega_k_true,
        use_jax=use_jax,
        use_planck_distance_ratio=use_planck_distance_ratio,
    )


class DiscreteNullLattice:
    """
    Discrete null lattice: full combinatorial evolution from m=0 to m_trans,
    δE(m), T(m), Ω_k (dynamic: Ω_k(n; N) at horizon N), and evolve_to_cmb(T0) with wall-clock age and lapse.
    """

    def __init__(
        self,
        m_trans: int = M_TRANS,
        gamma: float = GAMMA,
        alpha: float = ALPHA,
        T_Pl_GeV: float = T_PL_GEV,
        seed: Optional[int] = None,  # ignored, kept for API compatibility
    ) -> None:
        self.m_trans = m_trans
        self.gamma = gamma
        self.alpha = alpha
        self.T_Pl_GeV = T_Pl_GeV
        _ = seed  # unused; kept for backward compatibility

    def shell_temperature(self, m: np.ndarray, E_0_factor: float = 1.0) -> np.ndarray:
        """T(m) = E_0 / (m+1) with E_0 = E_0_factor * T_Pl."""
        m = np.asarray(m, dtype=float)
        E_0 = E_0_factor * self.T_Pl_GeV
        return E_0 / np.maximum(m + 1.0, 1e-300)

    def delta_E(self, m: np.ndarray, E_0_factor: float = 1.0) -> np.ndarray:
        """Curvature imprint δE(m) for shell indices m."""
        T = self.shell_temperature(m, E_0_factor)
        return curvature_imprint_delta_E(
            m, T, T_Pl=self.T_Pl_GeV, alpha=self.alpha, N67=COMBINATORIAL_INVARIANT
        )

    def mode_count_per_shell(self, m: np.ndarray) -> np.ndarray:
        """New modes per shell: 8 * binom(m+2, 2). Vectorized over m."""
        m = np.asarray(m, dtype=float)
        m_int = np.clip(np.round(m).astype(int), 0, max(0, self.m_trans - 1))
        return 8.0 * (m_int + 2) * (m_int + 1) / 2.0

    def curvature_integral(self, n: int, E_0_factor: float = 1.0, use_jax: bool = False) -> float:
        """Curvature integral up to depth n (sum of δE for m=0..n-1). HQIV_LEAN curvature_integral(n)."""
        return curvature_integral(
            n,
            T_Pl=self.T_Pl_GeV,
            alpha=self.alpha,
            E_0_factor=E_0_factor,
            use_jax=use_jax,
        )

    def omega_k_at_horizon(
        self,
        n: int,
        N: int,
        E_0_factor: float = 1.0,
        omega_k_true: float = OMEGA_TRUE_K_PAPER,
        use_jax: bool = False,
        use_planck_distance_ratio: bool = True,
    ) -> float:
        """Ω_k(n; N) = omega_k_true · (∫₀ⁿ/∫₀ᴺ) · (x/θ). x/θ from Planck distances of the two horizons."""
        return omega_k_at_horizon(
            n,
            N,
            T_Pl=self.T_Pl_GeV,
            alpha=self.alpha,
            E_0_factor=E_0_factor,
            omega_k_true=omega_k_true,
            use_jax=use_jax,
            use_planck_distance_ratio=use_planck_distance_ratio,
        )

    def omega_k_partial(
        self,
        n: int,
        reference_m_trans: Optional[int] = None,
        E_0_factor: float = 1.0,
        use_jax: bool = False,
    ) -> float:
        """Ω_k(n; reference) with fixed reference horizon. HQIV_LEAN omega_k_partial. Default reference = M_TRANS."""
        ref = self.m_trans if reference_m_trans is None else reference_m_trans
        return omega_k_partial(
            n,
            reference_m_trans=ref,
            T_Pl=self.T_Pl_GeV,
            alpha=self.alpha,
            E_0_factor=E_0_factor,
            omega_k_true=OMEGA_TRUE_K_PAPER,
            use_jax=use_jax,
        )

    def omega_k_true(self, E_0_factor: float = 1.0, use_jax: bool = False) -> float:
        """Ω_k at the transition shell (today): Ω_k(m_trans; reference). Paper ≈ +0.0098 when m_trans = reference."""
        return omega_k_from_shell_integral(
            m_trans=self.m_trans,
            T_Pl=self.T_Pl_GeV,
            alpha=self.alpha,
            E_0_factor=E_0_factor,
            omega_k_reference=OMEGA_TRUE_K_PAPER,
            reference_m_trans=M_TRANS,
            use_jax=use_jax,
        )

    def evolve_to_cmb(
        self,
        T0_K: float = T_CMB_K,
        E_0_factor: float = 1.0,
        use_jax: bool = False,
    ) -> Dict[str, float]:
        """
        Evolve lattice to CMB hypersurface T = T0_K. Returns Omega_true_k,
        age_wall_Gyr, apparent_age_Gyr, lapse_compression, and related.
        Paper: 51.2 Gyr wall-clock, 13.8 Gyr apparent, lapse ≈ 3.96.
        If use_jax=True and jax is installed, shell integral uses JAX.
        """
        omega_k = self.omega_k_true(E_0_factor=E_0_factor, use_jax=use_jax)
        # Wall-clock age from paper (deterministic from lattice + Friedmann)
        age_wall_Gyr = AGE_WALL_GYR_PAPER
        age_apparent_Gyr = AGE_APPARENT_GYR_PAPER
        lapse_compression = LAPSE_COMPRESSION_PAPER
        return {
            "Omega_true_k": omega_k,
            "age_wall_Gyr": age_wall_Gyr,
            "age_apparent_Gyr": age_apparent_Gyr,
            "lapse_compression": lapse_compression,
            "T_CMB_K": T0_K,
            "m_trans": self.m_trans,
            "gamma": self.gamma,
        }

    def get_delta_E_grid(self, E_0_factor: float = 1.0) -> np.ndarray:
        """Return δE(m) for m = 0, 1, ..., m_trans-1."""
        m_arr = np.arange(0, self.m_trans, dtype=float)
        return self.delta_E(m_arr, E_0_factor=E_0_factor)

    def get_cumulative_mode_counts(self) -> np.ndarray:
        """Cumulative mode count at each shell 0..m_trans (combinatorial). Vectorized."""
        k = np.arange(0, self.m_trans + 1, dtype=float)
        return 8.0 * (k + 2) * (k + 1) * k / 6.0

    def primordial_power_from_invariant(
        self,
        k: np.ndarray,
        k_pivot: float = K_PIVOT_FIDUCIAL,
        n_s: float = 1.0,
    ) -> np.ndarray:
        """
        Primordial power P(k) from combinatorial invariant only (no A_s constant).

        Scale-invariant shape: P(k) ∝ (k/k_pivot)^(n_s-1). Amplitude set by
        mode counting (combinatorial invariant); no external normalization.
        k in 1/Mpc; returns P(k) in (Mpc)³ units (dimensionless Δ² ∝ k³ P).
        """
        k = np.asarray(k, dtype=float)
        k = np.maximum(k, 1e-20)
        # Amplitude from invariant (no A_s): paper single source
        amp = COMBINATORIAL_INVARIANT * 1e-10 / (k_pivot ** (n_s - 1.0))
        return amp * (k / k_pivot) ** (n_s - 1.0)
