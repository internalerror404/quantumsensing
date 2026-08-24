"""Structured finite-dimensional order operators  A_n = a_n P_n D_{Delta_n}.

This is the abstract stand-in used by E0 and E1.  It is deliberately *not*
built from independent random matrices: the whole point of E1 is to separate
delay diversity from spatial diversity, and iid Gaussian blocks would supply
both at once and answer neither question.  Random operators appear only as the
matched negative control.

Geometry of the toy
-------------------
A source history is a movie ``x`` with ``M`` source-plane cells and ``H``
retarded-time samples, flattened cell-major as ``x[m*H + h]``.

  * ``D_Delta`` samples, per cell, the length-``W`` window beginning at the
    order's integer delay.  With the registered E0 numbers the identity
    ``H = W + N_max*D`` holds exactly (44 = 24 + 5*4), so the deepest order's
    window ends precisely at the end of the history.  Nothing is truncated and
    nothing is padded -- the history length is exactly what the order stack
    needs.  That identity is the reason this reading of the registered symbols
    is used; see docs/E0_SYMBOL_PINNING.md.
  * ``P_n`` maps the ``M`` source cells onto ``K`` screen samples.  Its
    order-to-order variation is the *spatial* diversity knob.
  * ``a_n`` is the attenuation, the knob that decides whether a structurally
    visible mode is also operationally visible.

The full order block is ``A_n = a_n (P_n kron I_W) D_{Delta_n}``, mapping
``R^{M*H} -> R^{K*W}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

DelayStructure = Literal["none", "constant", "perturbed", "cell_dependent"]
SpatialStructure = Literal["identical", "rotation", "rotation_shear", "independent", "gaussian"]
Attenuation = Literal["equalized", "exponential"]


# ---------------------------------------------------------------------------
# delay layer
# ---------------------------------------------------------------------------
def order_delays(order: int, structure: DelayStructure, delay_step: int,
                 n_cells: int, rng: np.random.Generator,
                 max_delay: int) -> np.ndarray:
    """Integer delay per source cell for one order.  Shape (n_cells,)."""
    if structure == "none":
        base = np.zeros(n_cells, dtype=int)
    elif structure == "constant":
        base = np.full(n_cells, order * delay_step, dtype=int)
    elif structure == "perturbed":
        jitter = rng.integers(-1, 2, size=1).repeat(n_cells)
        base = np.full(n_cells, order * delay_step, dtype=int) + jitter
    elif structure == "cell_dependent":
        spread = rng.integers(0, delay_step, size=n_cells)
        base = order * delay_step + spread - spread.mean().astype(int)
    else:
        raise ValueError(f"unknown delay structure {structure!r}")
    return np.clip(base, 0, max_delay).astype(int)


def window_matrix(delays: np.ndarray, history_length: int, window: int) -> np.ndarray:
    """Block-diagonal windowing operator, shape (n_cells*window, n_cells*history)."""
    n_cells = delays.size
    out = np.zeros((n_cells * window, n_cells * history_length))
    for m, d in enumerate(delays):
        for w in range(window):
            out[m * window + w, m * history_length + d + w] = 1.0
    return out


# ---------------------------------------------------------------------------
# spatial layer
# ---------------------------------------------------------------------------
def _rotation(dim: int, angle: float) -> np.ndarray:
    """Rotation acting in the leading 2-plane, identity elsewhere.  Chained over
    consecutive coordinate pairs so it mixes all cells for dim > 2."""
    R = np.eye(dim)
    c, s = np.cos(angle), np.sin(angle)
    for i in range(dim - 1):
        block = np.eye(dim)
        block[i, i] = c
        block[i, i + 1] = -s
        block[i + 1, i] = s
        block[i + 1, i + 1] = c
        R = block @ R
    return R


def _shear(dim: int, amount: float) -> np.ndarray:
    S = np.eye(dim)
    for i in range(dim - 1):
        S[i, i + 1] += amount
    return S


def spatial_projection(order: int, structure: SpatialStructure, n_cells: int,
                       n_screen: int, rng: np.random.Generator,
                       rotation_step: float = 0.4,
                       shear_step: float = 0.25) -> np.ndarray:
    """Screen-sampling matrix P_n, shape (n_screen, n_cells).

    ``identical`` reuses one sampler for every order, so any gain must come
    from delays alone.  ``independent`` is the matched-rank control: an
    order-specific orthogonal remixing of the source cells, which supplies
    order diversity without the rotation/shear geometry.  ``gaussian`` is the
    unstructured negative control.
    """
    base = _low_rank_sampler(n_screen, n_cells)
    if structure == "identical":
        return base
    if structure == "gaussian":
        g = rng.normal(size=(n_screen, n_cells))
        # match the row norms of the structured sampler so the comparison is
        # about structure, not about scale
        g *= (np.linalg.norm(base, axis=1, keepdims=True)
              / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300))
        return g
    if structure == "rotation":
        return base @ _rotation(n_cells, rotation_step * order)
    if structure == "rotation_shear":
        return base @ _shear(n_cells, shear_step * order) @ _rotation(n_cells, rotation_step * order)
    if structure == "independent":
        Q, _ = np.linalg.qr(rng.normal(size=(n_cells, n_cells)))
        return base @ Q
    raise ValueError(f"unknown spatial structure {structure!r}")


def _low_rank_sampler(n_screen: int, n_cells: int) -> np.ndarray:
    """Smooth, fully deterministic screen sampler.

    Deterministic on purpose: every source of order-to-order diversity in this
    module is an explicit, named knob, so a sampler that quietly varied with a
    seed would contaminate the "identical" arm that exists to hold spatial
    structure fixed.

    Rows are shifted raised-cosine kernels over the source cells: a physical
    screen pixel integrates a compact patch of the source plane, it does not
    take an iid random combination of it.
    """
    centres = np.linspace(0.0, n_cells - 1.0, n_screen)
    cells = np.arange(n_cells, dtype=float)
    width = max(n_cells / max(n_screen, 1), 1.0)
    d = (cells[None, :] - centres[:, None]) / width
    P = np.where(np.abs(d) < 1.0, 0.5 * (1.0 + np.cos(np.pi * d)), 0.0)
    # a small structured (not random) asymmetry keeps P from being exactly
    # rank-deficient when n_screen > n_cells
    P = P + 0.05 * np.cos(np.outer(np.arange(n_screen), cells) * 0.7)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    return P / np.maximum(norms, 1e-300)


# ---------------------------------------------------------------------------
# attenuation
# ---------------------------------------------------------------------------
def order_amplitude(order: int, mode: Attenuation, gamma: float) -> float:
    if mode == "equalized":
        return 1.0
    if mode == "exponential":
        return float(np.exp(-gamma * order))
    raise ValueError(f"unknown attenuation {mode!r}")


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToySpec:
    history_length: int = 44          # H
    window: int = 24                  # W
    n_screen: int = 6                 # K
    n_cells: int = 2                  # M
    max_order: int = 5                # N_max
    delay_step: int = 4               # D
    gamma: float = 0.6                # Gamma
    delay: DelayStructure = "constant"
    spatial: SpatialStructure = "identical"
    attenuation: Attenuation = "exponential"

    @property
    def source_dimension(self) -> int:
        return self.n_cells * self.history_length

    @property
    def rows_per_order(self) -> int:
        return self.n_screen * self.window

    @property
    def max_delay(self) -> int:
        return self.history_length - self.window

    def consistent(self) -> bool:
        """True when the deepest order's window ends exactly at the history end."""
        return self.history_length == self.window + self.max_order * self.delay_step

    def validate(self) -> None:
        """Refuse a spec whose delay ladder does not fit inside the history.

        Clipping to make it fit would silently collapse every order onto delay
        zero -- the delay mechanism would vanish while the run still reported
        success, which is precisely the silent fallback the protocol forbids.
        """
        if self.window > self.history_length:
            raise ValueError(
                f"window {self.window} exceeds history length {self.history_length}")
        need = self.window + self.max_order * self.delay_step
        if need > self.history_length:
            raise ValueError(
                f"delay ladder does not fit: window {self.window} + max_order "
                f"{self.max_order} * delay_step {self.delay_step} = {need} > "
                f"history {self.history_length}. Lengthen the history or shorten "
                f"the ladder; delays are never clipped to fit.")


def build_order_blocks(spec: ToySpec, seed: int,
                       orders: Sequence[int] | None = None) -> list[np.ndarray]:
    """One dense A_n per order, in order-index order."""
    spec.validate()
    orders = list(range(spec.max_order + 1)) if orders is None else list(orders)
    blocks: list[np.ndarray] = []
    # One generator per (seed, order) so that adding an order never perturbs
    # the operators of the orders below it.
    shared_P = _low_rank_sampler(spec.n_screen, spec.n_cells)
    for n in orders:
        rng = np.random.default_rng([seed, n])
        delays = order_delays(n, spec.delay, spec.delay_step, spec.n_cells, rng, spec.max_delay)
        Dn = window_matrix(delays, spec.history_length, spec.window)
        if spec.spatial == "identical":
            Pn = shared_P
        else:
            Pn = spatial_projection(n, spec.spatial, spec.n_cells, spec.n_screen, rng)
        an = order_amplitude(n, spec.attenuation, spec.gamma)
        blocks.append(an * np.kron(Pn, np.eye(spec.window)) @ Dn)
    return blocks


def shuffle_delays(spec: ToySpec, seed: int) -> list[np.ndarray]:
    """Negative control: permute the delay assignment across orders, preserving
    the marginal distribution of delays but destroying its order ordering."""
    blocks = build_order_blocks(spec, seed)
    rng = np.random.default_rng([seed, 12345])
    perm = rng.permutation(len(blocks))
    return [blocks[i] for i in perm]


def duplicate_order(spec: ToySpec, seed: int) -> list[np.ndarray]:
    """Negative control: every higher order is a scaled duplicate of order 0."""
    blocks = build_order_blocks(spec, seed)
    a0 = order_amplitude(0, spec.attenuation, spec.gamma)
    out = [blocks[0]]
    for n in range(1, len(blocks)):
        an = order_amplitude(n, spec.attenuation, spec.gamma)
        out.append((an / a0) * blocks[0])
    return out


def zero_amplitude_high_orders(spec: ToySpec, seed: int) -> list[np.ndarray]:
    """Negative control: higher orders carry no signal at all."""
    blocks = build_order_blocks(spec, seed)
    return [blocks[0]] + [np.zeros_like(b) for b in blocks[1:]]
