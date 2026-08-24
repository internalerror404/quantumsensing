"""Declared source classes for the abstract E0/E1 toy.

Every class is returned as an *orthonormal* basis ``Q`` with columns spanning
the class inside the flattened (cell-major) source space.  Restricted
identifiability is always ``B Q``, never ``B`` with a mask applied afterwards:
the two differ whenever the class is not axis-aligned, and the second is wrong.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.fft as sfft

ToyClass = Literal["full", "smooth_separable", "localized_atoms", "orbit_tangent"]


class ClassNotConstructible(ValueError):
    """The requested source class does not exist at these dimensions."""


def _dct_modes(length: int, count: int) -> np.ndarray:
    """First ``count`` orthonormal DCT-II basis vectors, shape (length, count).

    Asking for more modes than the axis has samples is a specification error,
    not a numerical one: ``np.eye(length, count)`` silently pads with zero
    columns and the normalisation then divides by zero, producing NaN columns
    that survive QR as a spuriously smaller basis.  Refuse instead.
    """
    if count > length:
        raise ClassNotConstructible(
            f"cannot build {count} DCT modes on an axis of length {length}")
    M = sfft.idct(np.eye(length, count), axis=0, type=2, norm="ortho")
    return M / np.linalg.norm(M, axis=0, keepdims=True)


def smooth_separable(n_cells: int, history: int, n_spatial: int = 4,
                     n_temporal: int = 6) -> np.ndarray:
    """Separable low-frequency class: n_spatial x n_temporal smooth modes.

    With the registered E0 numbers (4 spatial, 6 temporal) this is exactly the
    24-dimensional smooth restricted model.
    """
    S = _dct_modes(n_cells, n_spatial)          # (M, n_spatial)
    T = _dct_modes(history, n_temporal)         # (H, n_temporal)
    cols = [np.outer(S[:, a], T[:, b]).reshape(-1) for a in range(n_spatial)
            for b in range(n_temporal)]
    Q = np.column_stack(cols)
    return _orthonormalise(Q)


def localized_atoms(n_cells: int, history: int, count: int = 24,
                    width: float = 2.0, seed: int = 0) -> np.ndarray:
    """Compact Gaussian bursts at scattered (cell, time) locations."""
    rng = np.random.default_rng(seed)
    t = np.arange(history, dtype=float)
    cols = []
    for i in range(count):
        m = int(rng.integers(0, n_cells))
        t0 = float(rng.uniform(0, history - 1))
        atom = np.zeros((n_cells, history))
        atom[m] = np.exp(-0.5 * ((t - t0) / width) ** 2)
        cols.append(atom.reshape(-1))
    return _orthonormalise(np.column_stack(cols))


def orbit_tangent(n_cells: int, history: int, count: int = 24,
                  angular_speed: float = 0.35, width: float = 1.2,
                  seed: int = 0) -> np.ndarray:
    """Tangent space of a hotspot orbiting through the source cells.

    Columns are derivatives of the orbit model with respect to its parameters
    (amplitude, phase, radius-proxy, width) at several reference phases, which
    is the class a generative orbit prior would actually make identifiable.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(history, dtype=float)
    cells = np.arange(n_cells, dtype=float)
    cols = []
    phases = np.linspace(0.0, 2 * np.pi, max(count // 4, 1), endpoint=False)
    for ph in phases:
        centre = (n_cells - 1) * 0.5 * (1.0 + np.sin(angular_speed * t + ph))
        d = cells[:, None] - centre[None, :]
        g = np.exp(-0.5 * (d / width) ** 2)
        cols.append(g.reshape(-1))                                  # amplitude
        cols.append((g * (d / width ** 2) * np.cos(angular_speed * t + ph)).reshape(-1))  # phase
        cols.append((g * (d / width ** 2)).reshape(-1))              # radius proxy
        cols.append((g * (d ** 2 / width ** 3)).reshape(-1))         # width
    Q = np.column_stack(cols[:count])
    if Q.shape[1] < count:                                           # pad if needed
        extra = rng.normal(size=(Q.shape[0], count - Q.shape[1])) * 1e-3
        Q = np.column_stack([Q, extra])
    return _orthonormalise(Q)


def _orthonormalise(Q: np.ndarray) -> np.ndarray:
    """QR with rank-deficient columns dropped, so the returned basis really is
    a basis and ``restrict`` cannot be handed a singular Q."""
    Qo, R = np.linalg.qr(Q)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-10 * max(diag.max(), 1e-300)
    return np.ascontiguousarray(Qo[:, keep])


def source_class(name: ToyClass, n_cells: int, history: int,
                 dimension: int = 24, seed: int = 0) -> np.ndarray:
    if name == "full":
        return np.eye(n_cells * history)
    if name == "smooth_separable":
        return smooth_separable(n_cells, history)
    if name == "localized_atoms":
        return localized_atoms(n_cells, history, dimension, seed=seed)
    if name == "orbit_tangent":
        return orbit_tangent(n_cells, history, dimension, seed=seed)
    raise ValueError(f"unknown source class {name!r}")
