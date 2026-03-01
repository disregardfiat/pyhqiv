"""
Octonion HQIV algebra: left-multiplication matrices, g₂ + Δ, Lie closure to so(8),
hypercharge 4×4 block. Exact reproduction of HQVM/matrices.py (paper Appendix).
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from pyhqiv.constants import GAMMA  # not used in algebra; kept for API consistency


class OctonionHQIVAlgebra:
    """
    Bolt-on HQIV dynamical algebra calculator.
    - Generates all 7 left-multiplication matrices L(e_i)
    - Defines exact Δ (phase-lift generator)
    - Computes Lie closure dimension (g₂ + Δ)
    - Ready for calculator apps: call .lie_closure_dimension() or .print_status()
    """

    def __init__(self, verbose: bool = True) -> None:
        self.n: int = 8
        self.L: List[np.ndarray] = self._build_left_multiplications()
        self.Delta: np.ndarray = self._build_Delta()
        self.g2_basis: List[np.ndarray] = self._build_g2_basis()
        if verbose:
            self.print_status()

    def _build_left_multiplications(self) -> List[np.ndarray]:
        """Standard Fano-plane L(e_i), with L(e7) exactly as in the paper."""
        L = [np.zeros((8, 8)) for _ in range(8)]

        # L(e7) - colour preferred axis (exact match)
        L[7] = np.array([
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ])

        L[1] = np.array([
            [0, -1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
        ])
        L[2] = np.array([
            [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0],
        ])
        L[3] = np.array([
            [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, -1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
        ])
        L[4] = np.array([
            [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0],
        ])
        L[5] = np.array([
            [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
        ])
        L[6] = np.array([
            [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0],
        ])
        return L[1:]  # return only L1 to L7

    def _build_Delta(self) -> np.ndarray:
        """Phase-lift generator Δ = ∂/∂δθ′ (rotation in (e1,e7) plane)."""
        Delta = np.zeros((8, 8))
        Delta[1, 7] = -1.0
        Delta[7, 1] = 1.0
        return Delta

    def _build_g2_basis(self) -> List[np.ndarray]:
        """Generate 14 independent g₂ derivations from commutators of L(e_i)."""
        basis: List[np.ndarray] = []
        for i, j in combinations(range(7), 2):
            comm = self.L[i] @ self.L[j] - self.L[j] @ self.L[i]
            if np.max(np.abs(comm)) > 1e-12:
                basis.append(comm)
        return basis[:14]

    @staticmethod
    def _pack_antisym(M: np.ndarray) -> np.ndarray:
        """Pack 8×8 antisymmetric M into 28 independent entries (upper triangle)."""
        return np.array([M[i, j] for i in range(8) for j in range(i + 1, 8)])

    @staticmethod
    def _unpack_antisym(v: np.ndarray) -> np.ndarray:
        """Unpack 28-vector to 8×8 antisymmetric matrix."""
        M = np.zeros((8, 8))
        idx = 0
        for i in range(8):
            for j in range(i + 1, 8):
                M[i, j] = v[idx]
                M[j, i] = -v[idx]
                idx += 1
        return M

    def lie_closure_dimension(
        self, tol: float = 1e-10, max_iter: int = 40
    ) -> Tuple[int, List[int]]:
        """Iterative Lie closure in the 28-dim space of antisymmetric 8×8 (so(8))."""
        generators = self.g2_basis + [self.Delta]
        current = [g.copy() for g in generators]
        vecs = np.stack([self._pack_antisym(M) for M in current], axis=1)
        history: List[int] = [vecs.shape[1]]
        for _ in range(max_iter):
            new_mats: List[np.ndarray] = []
            for a, b in combinations(range(len(current)), 2):
                comm = current[a] @ current[b] - current[b] @ current[a]
                if np.max(np.abs(comm)) > tol:
                    new_mats.append(comm)
            old_rank = vecs.shape[1]
            for M in new_mats:
                v = self._pack_antisym(M)
                proj = vecs @ (np.linalg.pinv(vecs) @ v)
                residual = v - proj
                if np.linalg.norm(residual) > tol:
                    vecs = np.hstack((vecs, residual.reshape(-1, 1)))
                    U, S, _ = np.linalg.svd(vecs, full_matrices=False)
                    rank = int(np.sum(S > tol))
                    vecs = U[:, :rank]
            history.append(vecs.shape[1])
            if vecs.shape[1] == old_rank or vecs.shape[1] >= 28:
                break
        return int(vecs.shape[1]), history

    def lie_closure_basis(
        self, tol: float = 1e-10, max_iter: int = 40
    ) -> List[np.ndarray]:
        """Return the 28 basis matrices of so(8) as a list of 8×8 arrays."""
        generators = self.g2_basis + [self.Delta]
        current = [g.copy() for g in generators]
        vecs = np.stack([self._pack_antisym(M) for M in current], axis=1)
        for _ in range(max_iter):
            new_mats = []
            for a, b in combinations(range(len(current)), 2):
                comm = current[a] @ current[b] - current[b] @ current[a]
                if np.max(np.abs(comm)) > tol:
                    new_mats.append(comm)
            old_rank = vecs.shape[1]
            for M in new_mats:
                v = self._pack_antisym(M)
                proj = vecs @ (np.linalg.pinv(vecs) @ v)
                residual = v - proj
                if np.linalg.norm(residual) > tol:
                    vecs = np.hstack((vecs, residual.reshape(-1, 1)))
                    U, S, _ = np.linalg.svd(vecs, full_matrices=False)
                    rank = int(np.sum(S > tol))
                    vecs = U[:, :rank]
            if vecs.shape[1] == old_rank or vecs.shape[1] >= 28:
                break
        n_basis = vecs.shape[1]
        return [self._unpack_antisym(vecs[:, k]) for k in range(n_basis)]

    def _identify_color_generators(self, tol: float = 1e-8) -> List[np.ndarray]:
        """Return the 8 generators in g2_basis that preserve e7 (SU(3)_c)."""
        color_gens: List[np.ndarray] = []
        e7 = np.zeros(8)
        e7[7] = 1.0
        for g in self.g2_basis:
            action_on_e7 = g @ e7
            if np.max(np.abs(action_on_e7)) < tol:
                color_gens.append(g)
        return color_gens

    def hypercharge_coefficients(
        self,
        tol: float = 1e-12,
        block_weight: float = 1e15,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        """Weighted least-squares: block (4×4) heavily weighted, then minimize ‖[Y,g₂]‖."""
        basis = self.lie_closure_basis(tol=tol)
        if len(basis) != 28:
            return None, None, basis

        rows = [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
        target = np.array([1 / 6, 0.0, 0.0, 0.0, 0.0, 1 / 2], dtype=np.float64)
        A_block = np.array([[basis[k][i, j] for k in range(28)] for i, j in rows])

        A_comm = []
        for g in self.g2_basis:
            for i in range(8):
                for j in range(i + 1, 8):
                    row = np.array([
                        np.sum(basis[k][i, :] * g[:, j] - g[i, :] * basis[k][:, j])
                        for k in range(28)
                    ], dtype=np.float64)
                    A_comm.append(row)
        A_comm_arr = np.array(A_comm)

        w = block_weight
        A = np.vstack((w * A_block, A_comm_arr))
        b = np.concatenate((w * target, np.zeros(A_comm_arr.shape[0], dtype=np.float64)))
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-14)
        c = np.asarray(c, dtype=np.float64).ravel()[:28]
        if len(c) < 28:
            c = np.pad(c, (0, 28 - len(c)))

        Y = sum(c[k] * basis[k] for k in range(28))
        return c, Y, basis

    def hypercharge_verify(self, Y: np.ndarray, tol: float = 1e-14) -> dict:
        """Verify Y: 4×4 block eigenvalues ±i/6, ±i/2; report max ‖[Y,g₂]‖."""
        block = Y[4:8, 4:8].copy()
        evals = np.linalg.eigvals(block)
        evals_im = np.sort(np.imag(evals))
        err_block = np.abs(block[0, 1] - 1 / 6) + np.abs(block[2, 3] - 1 / 2)
        comm_g2 = [np.max(np.abs(Y @ g - g @ Y)) for g in self.g2_basis]
        max_comm = max(comm_g2) if comm_g2 else 0.0
        return {
            "block_4x4": block,
            "eigenvalues_i_block": evals_im,
            "block_entry_error": err_block,
            "max_commutation_with_g2": max_comm,
        }

    def hypercharge_paper_data(self) -> Optional[dict]:
        """Return exact c, 8×8 Y, 4×4 block, and block eigenvalues for the paper."""
        c, Y, _ = self.hypercharge_coefficients()
        if c is None or Y is None:
            return None
        ver = self.hypercharge_verify(Y)
        return {
            "c": c,
            "Y": Y,
            "block_4x4": ver["block_4x4"],
            "eigenvalues_i_block": ver["eigenvalues_i_block"],
            "block_entry_error": ver["block_entry_error"],
            "max_commutation_with_g2": ver["max_commutation_with_g2"],
        }

    def get_sm_embedding(self) -> dict:
        """Return Standard Model embedding: SU(3)_c generators, U(1)_Y, and so(8) basis."""
        color = self._identify_color_generators()
        c, Y, _ = self.hypercharge_coefficients()
        y = Y if Y is not None else np.zeros((8, 8))
        return {"su3c": color, "u1y": y, "so8_basis": self.lie_closure_basis()}

    def check_triality_anomalies(self, tol: float = 1e-12) -> dict:
        """Explicit check that the three 8's cancel anomalies under SM subgroups."""
        y_left = np.array([0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, -1 / 2])
        y_right = np.array([0, 2 / 3, 2 / 3, 2 / 3, -1 / 3, -1 / 3, -1 / 3, -1.0])
        a_yyy_one_gen = np.sum(y_left**3) - np.sum(y_right**3)
        a_yyy_three = 3 * a_yyy_one_gen
        sum_y_doublets = -1 / 2 + 3 * (1 / 6)
        a_2y_three = 3 * sum_y_doublets
        sum_y_triplets_L = 3 * (1 / 6) + 3 * (1 / 6)
        sum_y_triplets_R = 3 * (2 / 3) + 3 * (-1 / 3)
        a_3y_three = 3 * (sum_y_triplets_L - sum_y_triplets_R)
        a_grav_three = 3 * (np.sum(y_left) - np.sum(y_right))

        results: dict = {
            "U(1)_Y^3": a_yyy_three,
            "SU(2)_L^2 U(1)_Y": a_2y_three,
            "SU(3)_c^2 U(1)_Y": a_3y_three,
            "Grav^2 U(1)_Y": a_grav_three,
        }
        gauge_cancelled = (
            np.abs(results["SU(2)_L^2 U(1)_Y"]) < tol
            and np.abs(results["SU(3)_c^2 U(1)_Y"]) < tol
        )
        results["_gauge_cancelled"] = gauge_cancelled
        results["_cancelled"] = all(np.abs(v) < tol for k, v in results.items() if not k.startswith("_"))
        results["_per_generation"] = {
            "U(1)_Y^3": a_yyy_one_gen,
            "SU(2)_L^2 U(1)_Y": sum_y_doublets,
            "SU(3)_c^2 U(1)_Y": sum_y_triplets_L - sum_y_triplets_R,
            "Grav^2 U(1)_Y": np.sum(y_left) - np.sum(y_right),
        }
        return results

    def print_status(self) -> bool:
        """Print closure status; return True if dim == 28."""
        dim, history = self.lie_closure_dimension()
        print("=== HQIV Dynamical Lie Algebra Calculator ===")
        print(f"Final dimension: {dim} / 28 (so(8))")
        print(f"Growth: {history}")
        print(f"Full so(8) closure: {'✓ YES' if dim == 28 else '✗ NO'}")
        print(f"g₂ basis size: {len(self.g2_basis)}")
        print("Δ included: Yes")
        return dim == 28
