"""Linear reconstructors behind one interface, plus registered tuning rules.

Hyperparameters are selected on validation sources or by a discrepancy rule.
The oracle curve exists, but it is labelled an optimistic ceiling everywhere it
appears and is never the deployable number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import scipy.linalg as sla

Rule = Literal["oracle", "validation", "discrepancy", "gcv"]


@dataclass
class Solution:
    x: np.ndarray
    hyperparameter: float
    rule: Rule
    residual: float


def _svd(B: np.ndarray):
    return sla.svd(B, full_matrices=False)


def tsvd_solve(B: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    U, s, Vt = _svd(B)
    k = int(np.clip(k, 0, s.size))
    if k == 0:
        return np.zeros(B.shape[1])
    return Vt[:k].T @ ((U[:, :k].T @ y) / s[:k])


def ridge_solve(B: np.ndarray, y: np.ndarray, lam: float,
                L: np.ndarray | None = None) -> np.ndarray:
    """min ||Bx - y||^2 + lam ||L x||^2.  L defaults to the identity."""
    d = B.shape[1]
    Reg = np.eye(d) if L is None else L
    lhs = B.T @ B + lam * (Reg.T @ Reg)
    return np.linalg.solve(lhs, B.T @ y)


def second_difference(d: int) -> np.ndarray:
    """Temporal second-difference operator, for smoothness Tikhonov."""
    L = np.zeros((max(d - 2, 0), d))
    for i in range(d - 2):
        L[i, i], L[i, i + 1], L[i, i + 2] = 1.0, -2.0, 1.0
    return L


def select_hyperparameter(
    B: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float],
    rule: Rule,
    solve: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    *,
    truth: np.ndarray | None = None,
    validation: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    noise_level: float | None = None,
) -> float:
    """Pick one hyperparameter under a registered rule.

    ``oracle`` needs the truth and is only ever an upper bound on achievable
    quality; the deployable rules never see ``truth``.
    """
    grid = list(grid)
    if rule == "oracle":
        if truth is None:
            raise ValueError("oracle rule requires the ground truth")
        errs = [np.linalg.norm(solve(B, y, h) - truth) for h in grid]
        return grid[int(np.argmin(errs))]
    if rule == "validation":
        if not validation:
            raise ValueError("validation rule requires held-out (y, x) pairs")
        errs = [float(np.mean([np.linalg.norm(solve(B, yv, h) - xv) for yv, xv in validation]))
                for h in grid]
        return grid[int(np.argmin(errs))]
    if rule == "discrepancy":
        if noise_level is None:
            raise ValueError("discrepancy rule requires the noise level")
        # Morozov: largest regularisation whose residual still matches the
        # noise floor.  Scanning from strong to weak returns the most stable
        # admissible choice rather than the first numerically lucky one.
        for h in sorted(grid, reverse=True):
            if np.linalg.norm(B @ solve(B, y, h) - y) <= noise_level:
                return h
        return min(grid)
    if rule == "gcv":
        U, s, Vt = _svd(B)
        n = B.shape[0]
        best, best_v = grid[0], np.inf
        for h in grid:
            f = s ** 2 / (s ** 2 + h)
            resid = np.linalg.norm(y - U @ (f * (U.T @ y)))
            v = (resid ** 2 / n) / max((1.0 - f.sum() / n) ** 2, 1e-12)
            if v < best_v:
                best, best_v = h, v
        return best
    raise ValueError(f"unknown rule {rule!r}")


def nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.linalg.norm(truth))
    return float(np.linalg.norm(estimate - truth) / max(denom, 1e-300))


def split_visible_null(x: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a source vector into its data-visible and null components."""
    if visible.shape[1] == 0:
        return np.zeros_like(x), x.copy()
    proj = visible @ (visible.T @ x)
    return proj, x - proj
