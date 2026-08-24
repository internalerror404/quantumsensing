"""Order readouts: ideal, partially resolved, unresolved, and the controls.

The order axis is a genuine measurement axis.  ``L_eps`` interpolates from
perfect order separation to a rank-one collapse, which is the bridge between
the ideal theorem and any observation that infers order labels imperfectly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def leakage_matrix(n_orders: int, epsilon: float) -> np.ndarray:
    """L_eps = (1-eps) I + eps * 11^T / n_orders.

    At eps = 0 orders are perfectly resolved; at eps = 1 every pseudo-channel
    is the same average and the order axis has rank one.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    ones = np.ones((n_orders, n_orders)) / n_orders
    return (1.0 - epsilon) * np.eye(n_orders) + epsilon * ones


def two_channel_matrix(n_orders: int) -> np.ndarray:
    """Group n = 0 against n >= 1.  Row 1 averages the indirect orders so the
    two channels carry comparable weight rather than a growing sum."""
    if n_orders < 2:
        raise ValueError("two-channel grouping needs at least two orders")
    L = np.zeros((2, n_orders))
    L[0, 0] = 1.0
    L[1, 1:] = 1.0 / (n_orders - 1)
    return L


def unresolved_matrix(n_orders: int, weights: Sequence[float] | None = None) -> np.ndarray:
    """One-channel collapse  U = sum_n M_n T_n."""
    w = np.ones(n_orders) if weights is None else np.asarray(weights, dtype=float)
    if w.size != n_orders:
        raise ValueError("weight vector length must equal the number of orders")
    return w.reshape(1, n_orders)


def resolved_matrix(n_orders: int) -> np.ndarray:
    return np.eye(n_orders)


def miscalibrate(L: np.ndarray, relative: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb a known mixer by a registered relative amount.

    The perturbation is multiplicative on each entry and the rows are *not*
    renormalised: a miscalibrated instrument does not conveniently preserve
    its own row sums.
    """
    if relative == 0.0:
        return L.copy()
    return L * (1.0 + relative * rng.normal(size=L.shape))


@dataclass(frozen=True)
class OrderMixer:
    """Applies a channel mixer L (n_channels x n_orders) to a list of A_n."""

    L: np.ndarray
    name: str = "custom"

    @property
    def n_channels(self) -> int:
        return int(self.L.shape[0])

    @property
    def n_orders(self) -> int:
        return int(self.L.shape[1])

    def apply(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        """Stack the mixed channels into one dense operator."""
        blocks = list(blocks)
        if len(blocks) != self.n_orders:
            raise ValueError(f"mixer expects {self.n_orders} orders, got {len(blocks)}")
        rows = [sum(self.L[c, n] * blocks[n] for n in range(self.n_orders))
                for c in range(self.n_channels)]
        return np.vstack(rows)

    def condition_number(self) -> float:
        s = np.linalg.svd(self.L, compute_uv=False)
        s = s[s > 0]
        return float(s[0] / s[-1]) if s.size else float("inf")


def resolved(n_orders: int) -> OrderMixer:
    return OrderMixer(resolved_matrix(n_orders), "resolved")


def unresolved(n_orders: int, weights: Sequence[float] | None = None) -> OrderMixer:
    return OrderMixer(unresolved_matrix(n_orders, weights), "unresolved_sum")


def partial(n_orders: int, epsilon: float) -> OrderMixer:
    return OrderMixer(leakage_matrix(n_orders, epsilon), f"partial_leakage_eps{epsilon:g}")


def two_channel(n_orders: int) -> OrderMixer:
    return OrderMixer(two_channel_matrix(n_orders), "two_channel_direct_vs_indirect")
