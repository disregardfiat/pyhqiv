"""
HQIV waveguide: phase-horizon corrected Maxwell in a guide.

The single-source axiom E_tot = m c² + ħ c/Δx with Δx ≤ Θ_local(x) yields
φ = 2c²/Θ_local and the phase-horizon clock δθ′; all corrections here follow from
that. Modified curl equations use D/Dt = ∂/∂t + δ̇θ′ ∂/∂δθ′, with phase-fiber
coupling ∂/∂δθ′ → im, giving (∇⊥² + k_c²) E_t = 0 and
k_c² = ω²/c² - β² + 2imωδ̇θ′ + m²(δ̇θ′)² (paper Sec. 3.3, 3.7, 5.1). Geometry
design: constant-φ circle, adiabatic taper, hyperbolic cross-section, rectangular correction.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pyhqiv.constants import C_SI, GAMMA


def dot_delta_theta_from_phi(
    phi: Union[float, np.ndarray],
    gamma: float = GAMMA,
    c: float = C_SI,
) -> Union[float, np.ndarray]:
    """
    δ̇θ′ = γ φ / c (homogeneous HQVM limit + monogamy). Used in waveguide k_c².
    """
    return gamma * np.asarray(phi, dtype=float) / c


def kc_squared_hqiv(
    omega: float,
    beta: float,
    m_phase: int,
    dot_delta_theta: Union[float, np.ndarray],
    c: float = C_SI,
) -> Union[float, np.ndarray]:
    """
    Position-dependent cutoff wavenumber squared (complex):

    k_c² = (ω²/c² - β²) + 2imωδ̇θ′ + m²(δ̇θ′)².

    Paper: (∇⊥² + k_c²) E_t = 0. Non-Hermitian when m≠0 → complex, leaky, or phase-locked modes.
    """
    k0_sq = (omega / c) ** 2
    base = k0_sq - beta**2
    d = np.asarray(dot_delta_theta, dtype=complex)
    return base + 2j * m_phase * omega * d + (m_phase * d) ** 2


def waveguide_radius_constant_phi(
    phi_target: float,
    c: float = C_SI,
) -> float:
    """
    Radius a for constant-φ circular guide: a = Θ_desired = 2c²/φ_target.

    Only shape with uniform Θ_local (distance to wall) in the cross-section.
    """
    return 2.0 * (c**2) / max(phi_target, 1e-30)


def waveguide_te11_cutoff_beta(
    omega: float,
    a: float,
    m_phase: int,
    dot_delta_theta: float,
    c: float = C_SI,
) -> complex:
    """
    Propagation constant β for TE11 (first circular mode): standard cutoff 1.841/a,
    shifted by HQIV phase terms. β² = ω²/c² - (1.841/a)² + 2imωδ̇θ′ + m²(δ̇θ′)².
    """
    kc_te11 = 1.841 / max(a, 1e-30)
    kc_sq = (
        (omega / c) ** 2
        - kc_te11**2
        + 2j * m_phase * omega * dot_delta_theta
        + (m_phase * dot_delta_theta) ** 2
    )
    return np.sqrt(kc_sq + 0j)


def waveguide_taper_slope(
    a: float,
    Theta_local: float,
    dTheta_dz: float,
) -> float:
    """
    Adiabatic taper: da/dz = -a/Θ_local * dΘ_local/dz.
    Keeps phase-lift uniform along z.
    """
    if abs(Theta_local) < 1e-30:
        return 0.0
    return -a * dTheta_dz / Theta_local


def hyperbolic_boundary_r(
    theta: Union[float, np.ndarray],
    Theta0: float,
    kappa: float,
) -> Union[float, np.ndarray]:
    """
    Hyperbolic cross-section (observer-centric H³ fibre): r(θ) = Θ0 * cosh(κ θ).
    κ = sqrt(|sectional curvature|) ≈ H_local. Makes Θ_local constant on boundary.
    """
    return Theta0 * np.cosh(kappa * np.asarray(theta, dtype=float))


def rectangular_cutoff_kc_squared(
    m_mode: int,
    n_mode: int,
    w: float,
    h: float,
    omega: float,
    beta: float,
    m_phase: int,
    dot_delta_theta_avg: float,
    c: float = C_SI,
) -> complex:
    """
    Rectangular guide (w×h) with HQIV correction:
    k_c,mn² = (mπ/w)² + (nπ/h)² + 2im_phase ω δ̇θ′_avg + m_phase² (δ̇θ′_avg)².
    (Conventional mode indices m_mode, n_mode; m_phase = phase-winding number.)
    """
    kc_geom_sq = (m_mode * np.pi / max(w, 1e-30)) ** 2 + (n_mode * np.pi / max(h, 1e-30)) ** 2
    phase_term = 2j * m_phase * omega * dot_delta_theta_avg + (m_phase * dot_delta_theta_avg) ** 2
    return kc_geom_sq + phase_term


def distance_to_boundary_rect(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    w: float,
    h: float,
) -> np.ndarray:
    """
    Euclidean distance to nearest boundary of a rectangle [0,w]×[0,h].
    Used as Θ_local for rectangular guides.
    """
    x = np.asarray(grid_x)
    y = np.asarray(grid_y)
    dx = np.minimum(x, w - x)
    dy = np.minimum(y, h - y)
    return np.maximum(np.minimum(dx, dy), 1e-30)


def distance_to_boundary_circle(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cx: float = 0.0,
    cy: float = 0.0,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Distance to boundary of circle (centre cx,cy radius a): Θ = max(0, a - r).
    """
    x = np.asarray(grid_x)
    y = np.asarray(grid_y)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.maximum(radius - r, 1e-30)


def _laplacian_2d_dirichlet(
    n_x: int,
    n_y: int,
    dx: float,
    dy: float,
    interior_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete -∇⊥² on a 2D grid with Dirichlet (zero) on boundary.
    Returns (matrix in sparse form or dense, index map). Using dense for simplicity;
    scipy.sparse can be used for large grids.
    """
    n_total = int(interior_mask.sum())
    if n_total == 0:
        return np.zeros((0, 0)), np.zeros(0, dtype=int)
    idx = np.where(interior_mask.ravel())[0]
    inv_idx = np.full(interior_mask.size, -1)
    inv_idx[idx] = np.arange(n_total)
    L = np.zeros((n_total, n_total))
    for i, flat_i in enumerate(idx):
        iy, ix = np.unravel_index(flat_i, (n_y, n_x))
        L[i, i] = 2.0 / (dx**2) + 2.0 / (dy**2)
        for di, dj, coef in [
            (-1, 0, -1.0 / (dx**2)),
            (1, 0, -1.0 / (dx**2)),
            (0, -1, -1.0 / (dy**2)),
            (0, 1, -1.0 / (dy**2)),
        ]:
            ny, nx = iy + dj, ix + di
            if 0 <= ny < n_y and 0 <= nx < n_x:
                j = inv_idx[ny * n_x + nx]
                if j >= 0:
                    L[i, j] = coef
    return L, idx


def hqiv_waveguide_mode_solver(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    omega: float,
    beta_target: float,
    m_phase: int = 1,
    gamma: float = GAMMA,
    Theta_grid: Optional[np.ndarray] = None,
    c: float = C_SI,
    n_modes: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve (∇⊥² + k_c²) E_t = 0 on the cross-section.

    Builds -∇⊥² with Dirichlet BC on boundary (interior = where Theta > threshold).
    Computes k_c²(ω, β, m_phase, δ̇θ′) from Theta_grid (or constant Theta if None).
    Solves eigenvalue problem for the discrete operator; returns eigenvalues (≈ k_c²),
    eigenmodes, and interior mask. For complex k_c² (m_phase ≠ 0) uses dense eigensolver.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
    except ImportError:
        raise ImportError("scipy required for hqiv_waveguide_mode_solver")

    x = np.asarray(grid_x)
    y = np.asarray(grid_y)
    if x.shape != y.shape:
        raise ValueError("grid_x and grid_y must have the same shape")
    n_y, n_x = x.shape
    dx = float(np.diff(x[0, :]).mean()) if x.shape[1] > 1 else 1.0
    dy = float(np.diff(y[:, 0]).mean()) if y.shape[0] > 1 else 1.0
    dx = max(dx, 1e-30)
    dy = max(dy, 1e-30)

    if Theta_grid is None:
        interior_mask = np.ones_like(x, dtype=bool)
        Theta_grid = np.ones_like(x)
    else:
        Theta_grid = np.asarray(Theta_grid)
        interior_mask = Theta_grid > 0.01 * (Theta_grid.max() + 1e-30)

    phi_grid = 2.0 * (c**2) / np.maximum(Theta_grid, 1e-30)
    dot_dtheta = dot_delta_theta_from_phi(phi_grid, gamma=gamma, c=c)
    kc2_grid = kc_squared_hqiv(omega, beta_target, m_phase, dot_dtheta, c=c)

    L, idx = _laplacian_2d_dirichlet(n_x, n_y, dx, dy, interior_mask)
    n_total = L.shape[0]
    if n_total == 0:
        return np.array([]), np.array([]).reshape(0, 0), interior_mask

    kc2_interior = kc2_grid.ravel()[idx]
    K = np.diag(kc2_interior.astype(complex))
    # (∇⊥² + k_c²) E = 0 => discrete ∇² v + K v = 0. Our L = -∇², so (-L + K) v = 0 => (L - K) v = 0.
    A = L.astype(complex) - K  # eigenvalues near 0 correspond to modes

    n_ret = min(n_modes, n_total)
    if np.all(np.isreal(kc2_interior)) and m_phase == 0:
        A_real = A.real
        A_sp = csr_matrix(A_real)
        vals, vecs = eigsh(A_sp, k=n_ret, which="SM")
    else:
        vals, vecs = np.linalg.eig(A)
        order = np.argsort(np.abs(vals))
        vals = vals[order[:n_ret]]
        vecs = vecs[:, order[:n_ret]]

    return vals, vecs, interior_mask
