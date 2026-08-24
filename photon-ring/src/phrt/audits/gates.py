"""The mechanical correctness gates G2-G6, G9, G13.

Each returns a ``Gate`` carrying the measured value beside its threshold, so a
pass is never asserted without the number that justifies it.  Tolerances come
from the frozen registry and are never widened after a failure is seen.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator

from phrt.audits.rank import numerical_null_basis
from phrt.io.manifests import Gate, gate_from_tolerance


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


# --- G2 ---------------------------------------------------------------------
def gate_dense_parity(op: LinearOperator, dense: np.ndarray, tol: float,
                      name: str = "G2_dense_operator_relative") -> Gate:
    """Matrix-free forward map equals the dense reference."""
    got = np.column_stack([op.matvec(e) for e in np.eye(op.shape[1])])
    denom = max(float(np.abs(dense).max()), 1e-300)
    return gate_from_tolerance(name, float(np.abs(got - dense).max()) / denom, tol)


# --- G3 ---------------------------------------------------------------------
def gate_adjoint(op: LinearOperator, tol: float, n_trials: int = 20,
                 seed: int = 0, name: str = "G3_adjoint_relative") -> Gate:
    """<A x, y> == <x, A* y> over registered random probes.

    The worst trial is reported, not the mean: an adjoint that is right on
    average and wrong on one direction is a broken adjoint.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        x = rng.normal(size=op.shape[1])
        y = rng.normal(size=op.shape[0])
        worst = max(worst, _rel(float(y @ op.matvec(x)), float(x @ op.rmatvec(y))))
    return gate_from_tolerance(name, worst, tol,
                               note=f"worst of {n_trials} probes, seed {seed}")


# --- G4 ---------------------------------------------------------------------
def gate_order_collapse(blocks: Sequence[np.ndarray], mixed_unresolved: np.ndarray,
                        tol: float, weights: Sequence[float] | None = None,
                        name: str = "G4_order_collapse_relative") -> Gate:
    """Mixing the resolved stack down reproduces the unresolved operator."""
    w = np.ones(len(blocks)) if weights is None else np.asarray(weights, dtype=float)
    direct = sum(w[n] * blocks[n] for n in range(len(blocks)))
    denom = max(float(np.abs(direct).max()), 1e-300)
    return gate_from_tolerance(name, float(np.abs(direct - mixed_unresolved).max()) / denom, tol)


# --- G5 ---------------------------------------------------------------------
def gate_kernel_injection(B: np.ndarray, tol: float, seed: int = 0,
                          alpha: float = 1.0,
                          name: str = "G5_kernel_normalized_residual") -> Gate:
    """A numerical null vector must change the source and not the clean data.

    If the operator has no null space the gate is NOT_RUN, not PASS: there is
    nothing to inject and claiming a pass would misreport an untested property.
    """
    V = numerical_null_basis(B)
    if V.shape[1] == 0:
        return Gate(name, "NOT_RUN", note="operator has trivial numerical null space")
    rng = np.random.default_rng(seed)
    c = rng.normal(size=V.shape[1])
    v = V @ (c / np.linalg.norm(c))
    x0 = rng.normal(size=B.shape[1])
    dy = B @ (alpha * v)
    scale = max(float(np.linalg.norm(B @ x0)), 1e-300)
    measured = float(np.linalg.norm(dy)) / scale
    source_change = float(np.linalg.norm(alpha * v) / max(np.linalg.norm(x0), 1e-300))
    g = gate_from_tolerance(name, measured, tol,
                            note=f"null dim {V.shape[1]}, source moved by "
                                 f"{source_change:.3f} relative")
    return g


# --- G6 ---------------------------------------------------------------------
def gate_gram_monotonicity(gram_by_order: Sequence[np.ndarray], tol: float,
                           name: str = "G6_monotonicity_relative_negative_eigenvalue") -> Gate:
    """Cumulative information must never decrease as an order is added.

    Measured value is the most negative eigenvalue of any increment, relative
    to max(1, ||G||_2), reported as a positive number so the tolerance compare
    reads the same way as every other gate.
    """
    worst = 0.0
    detail = []
    for i in range(1, len(gram_by_order)):
        D = gram_by_order[i] - gram_by_order[i - 1]
        D = 0.5 * (D + D.T)
        lam = float(np.min(np.linalg.eigvalsh(D)))
        scale = max(1.0, float(np.linalg.norm(gram_by_order[i], 2)))
        worst = max(worst, max(0.0, -lam) / scale)
        detail.append(round(lam / scale, 18))
    return gate_from_tolerance(name, worst, tol,
                               note=f"min relative increment eigenvalue per order: {detail}")


# --- G9 ---------------------------------------------------------------------
def gate_seed_splits(namespaces, name: str = "G9_source_split_disjoint") -> Gate:
    ov = namespaces.overlaps()
    return Gate(name, "PASS" if not ov else "FAIL", measured=len(ov), threshold=0,
                note=f"overlapping pairs: {ov}" if ov else "all namespaces disjoint")


# --- G13 --------------------------------------------------------------------
def gate_replay(build: Callable[[], np.ndarray], tol: float, n_repeats: int = 2,
                name: str = "G13_replay") -> Gate:
    """A deterministic stage must reproduce bit-identical output on replay."""
    first = build()
    worst = 0.0
    for _ in range(n_repeats - 1):
        again = build()
        denom = max(float(np.abs(first).max()), 1e-300)
        worst = max(worst, float(np.abs(again - first).max()) / denom)
    return gate_from_tolerance(name, worst, tol, note=f"{n_repeats} builds compared")
