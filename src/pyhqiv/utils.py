"""Utility functions: grid generation, broadcasting, reproducibility."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def make_grid_3d(
    lo: Union[float, Tuple[float, float, float]],
    hi: Union[float, Tuple[float, float, float]],
    n: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Return (N, 3) array of grid points. lo/hi are bounds per axis; n is count per axis.
    """
    if np.isscalar(lo):
        lo = (lo, lo, lo)
    if np.isscalar(hi):
        hi = (hi, hi, hi)
    if np.isscalar(n):
        n = (n, n, n)
    mg = np.mgrid[
        lo[0] : hi[0] : complex(n[0]),
        lo[1] : hi[1] : complex(n[1]),
        lo[2] : hi[2] : complex(n[2]),
    ]
    return mg.reshape(3, -1).T


def local_theta_from_distance(r: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Θ_local ∝ r (or scale); φ = 2c²/Θ_local. Simple radial model."""
    r = np.asarray(r, dtype=float)
    return np.maximum(np.linalg.norm(r, axis=-1), 1e-30) * scale


def phi_from_theta_local(theta_local: np.ndarray, c: float = 1.0) -> np.ndarray:
    """φ(x) = 2c²/Θ_local(x). c in natural units can be 1."""
    return 2.0 * (c ** 2) / np.maximum(np.asarray(theta_local), 1e-30)


def set_seed(seed: Optional[int] = 42) -> None:
    """Set NumPy and Python RNG seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
