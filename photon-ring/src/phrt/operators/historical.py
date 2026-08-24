"""Matrix-free historical operators.

Handoff rule 4: never build the large dense operator in a core run.  Rule 5:
the adjoint is written by hand, because an autodiff adjoint tests the
differentiation machinery rather than the transpose that the theory uses.

Two implementations live here:

``StructuredHistoricalOperator``
    the abstract E0/E1 toy, matrix-free over (amplitude, screen sampler,
    per-cell integer delay) triples;

``RayMapHistoricalOperator``
    the physical Kerr operator, matrix-free over cached ray maps and a
    ``SourceBasis``.

Both expose the same interface, so every audit gate runs unchanged against
either one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator


@dataclass(frozen=True)
class OrderKernel:
    """One order's ingredients, kept factored rather than multiplied out."""

    amplitude: float
    screen: np.ndarray          # P_n, shape (n_screen, n_cells)
    delays: np.ndarray          # integer delay per source cell, shape (n_cells,)

    @property
    def n_screen(self) -> int:
        return int(self.screen.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.screen.shape[1])


class StructuredHistoricalOperator(LinearOperator):
    """A = mixer applied to [a_n (P_n kron I_W) D_{Delta_n}]_n, never materialised.

    Source layout is cell-major: ``x[m*H + h]``.
    Data layout is channel-major then screen-major: ``y[(c*K + k)*W + w]``.

    The dimension attributes are spelled out (``n_history``, ``n_window``,
    ``n_screen``, ``n_cells``) rather than H/W/K/M because ``LinearOperator``
    already defines ``.H`` as the Hermitian adjoint.
    """

    def __init__(self, kernels: Sequence[OrderKernel], history_length: int,
                 window: int, mixer: np.ndarray | None = None):
        self.kernels = list(kernels)
        self.n_history = int(history_length)   # H
        self.n_window = int(window)            # W
        k0 = self.kernels[0]
        self.n_screen = k0.n_screen            # K
        self.n_cells = k0.n_cells              # M
        n_orders = len(self.kernels)
        self.L = np.eye(n_orders) if mixer is None else np.asarray(mixer, dtype=np.float64)
        if self.L.shape[1] != n_orders:
            raise ValueError("mixer column count must equal the number of orders")
        for k in self.kernels:
            if k.n_screen != self.n_screen or k.n_cells != self.n_cells:
                raise ValueError("all orders must share screen and cell dimensions")
            if k.delays.size != self.n_cells:
                raise ValueError("delay vector must have one entry per source cell")
            if np.any(k.delays < 0) or np.any(k.delays + self.n_window > self.n_history):
                raise ValueError("delay window runs outside the source history")
        self.n_channels = int(self.L.shape[0])
        super().__init__(dtype=np.float64,
                         shape=(self.n_channels * self.n_screen * self.n_window, self.n_cells * self.n_history))

    # -- per-order primitives ---------------------------------------------
    def _order_forward(self, k: OrderKernel, X: np.ndarray) -> np.ndarray:
        """X is (M, H); returns (K, W)."""
        # window: gather each cell's delayed slice
        Xi = np.empty((self.n_cells, self.n_window))
        for m in range(self.n_cells):
            d = int(k.delays[m])
            Xi[m] = X[m, d:d + self.n_window]
        return k.amplitude * (k.screen @ Xi)

    def _order_adjoint(self, k: OrderKernel, Y: np.ndarray) -> np.ndarray:
        """Y is (K, W); returns (M, H).  Hand-written transpose of the above."""
        Xi = k.amplitude * (k.screen.T @ Y)      # (M, W)
        out = np.zeros((self.n_cells, self.n_history))
        for m in range(self.n_cells):
            d = int(k.delays[m])
            out[m, d:d + self.n_window] += Xi[m]
        return out

    # -- LinearOperator interface -----------------------------------------
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        X = np.asarray(x, dtype=np.float64).reshape(self.n_cells, self.n_history)
        per_order = [self._order_forward(k, X) for k in self.kernels]
        out = np.empty((self.n_channels, self.n_screen, self.n_window))
        for c in range(self.n_channels):
            acc = np.zeros((self.n_screen, self.n_window))
            for n, blk in enumerate(per_order):
                w = self.L[c, n]
                if w != 0.0:
                    acc += w * blk
            out[c] = acc
        return out.reshape(-1)

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        Y = np.asarray(y, dtype=np.float64).reshape(self.n_channels, self.n_screen, self.n_window)
        out = np.zeros((self.n_cells, self.n_history))
        for n, k in enumerate(self.kernels):
            acc = np.zeros((self.n_screen, self.n_window))
            for c in range(self.n_channels):
                w = self.L[c, n]
                if w != 0.0:
                    acc += w * Y[c]
            out += self._order_adjoint(k, acc)
        return out.reshape(-1)

    # -- reference materialisation (smoke sizes only) ----------------------
    def to_dense(self) -> np.ndarray:
        """Explicit matrix, for gate G2 and for smoke-size instances only."""
        cols = [self._matvec(e) for e in np.eye(self.shape[1])]
        return np.column_stack(cols)

    def gram(self, batch: int = 64) -> np.ndarray:
        """Streamed G = A^T A without materialising A.

        For d <= 600 this is cheaper and better conditioned than storing every
        row, and the right singular vectors are recoverable from eigh(G).
        """
        d = self.shape[1]
        G = np.zeros((d, d))
        for start in range(0, d, batch):
            stop = min(start + batch, d)
            E = np.eye(d)[:, start:stop]
            AE = np.column_stack([self._matvec(E[:, j]) for j in range(stop - start)])
            G[:, start:stop] = np.column_stack([self._rmatvec(AE[:, j]) for j in range(stop - start)])
        return 0.5 * (G + G.T)


def structured_operator(spec, seed: int, mixer: np.ndarray | None = None
                        ) -> StructuredHistoricalOperator:
    """Build the matrix-free toy operator from a ``ToySpec``."""
    from phrt.operators.structured import (order_amplitude, order_delays,
                                           spatial_projection, _low_rank_sampler)

    spec.validate()
    shared_P = _low_rank_sampler(spec.n_screen, spec.n_cells)
    kernels = []
    for n in range(spec.max_order + 1):
        rng = np.random.default_rng([seed, n])
        delays = order_delays(n, spec.delay, spec.delay_step, spec.n_cells, rng, spec.max_delay)
        P = shared_P if spec.spatial == "identical" else spatial_projection(
            n, spec.spatial, spec.n_cells, spec.n_screen, rng)
        kernels.append(OrderKernel(order_amplitude(n, spec.attenuation, spec.gamma), P, delays))
    return StructuredHistoricalOperator(kernels, spec.history_length, spec.window, mixer)
