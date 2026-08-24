"""Rank, conditioning, and effective-dimension conventions (protocol section 8).

Three distinct notions are kept apart on purpose, because conflating them is
the specific error the paper exists to avoid:

  numerical rank    -- how many singular values are above float noise;
  operational rank  -- how many modes a real measurement could actually see;
  effective rank    -- a soft, spectrum-shape summary, never a count of modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import scipy.linalg as sla

EPS64 = float(np.finfo(np.float64).eps)
REPORT_THRESHOLDS: tuple[float, ...] = (1e-8, 1e-10, 1e-12)


def lapack_rank_threshold(shape: tuple[int, int], sigma_max: float) -> float:
    """Registered primary rule: max(m, d) * eps * sigma_max."""
    return max(shape) * EPS64 * float(sigma_max)


@dataclass
class Spectrum:
    """Singular spectrum of a whitened operator, with every registered summary."""

    singular_values: np.ndarray
    shape: tuple[int, int]
    source_dimension: int
    operational_threshold: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- primary rank rule -------------------------------------------------
    @property
    def sigma_max(self) -> float:
        return float(self.singular_values[0]) if self.singular_values.size else 0.0

    @property
    def primary_threshold(self) -> float:
        return lapack_rank_threshold(self.shape, self.sigma_max)

    @property
    def numerical_rank(self) -> int:
        return int(np.sum(self.singular_values > self.primary_threshold))

    @property
    def nullity(self) -> int:
        return int(self.source_dimension - self.numerical_rank)

    @property
    def sigma_min_positive(self) -> float:
        keep = self.singular_values[self.singular_values > self.primary_threshold]
        return float(keep[-1]) if keep.size else 0.0

    @property
    def kappa_positive(self) -> float:
        """sigma_max / sigma_min^+ on the identifiable support only.

        Never an ordinary condition number after silently dropping null modes:
        the support size travels with the number via ``numerical_rank``.
        """
        s = self.sigma_min_positive
        return float(self.sigma_max / s) if s > 0 else float("inf")

    # -- effective dimensions ---------------------------------------------
    @property
    def stable_rank(self) -> float:
        if self.sigma_max <= 0:
            return 0.0
        return float(np.sum(self.singular_values ** 2) / self.sigma_max ** 2)

    @property
    def effective_rank(self) -> float:
        """Spectral-entropy effective rank exp(-sum p log p), p = sigma_i / sum."""
        s = self.singular_values[self.singular_values > 0]
        if s.size == 0:
            return 0.0
        p = s / s.sum()
        return float(np.exp(-np.sum(p * np.log(p))))

    @property
    def trace_information(self) -> float:
        return float(np.sum(self.singular_values ** 2))

    # -- operational rank --------------------------------------------------
    @property
    def operational_rank(self) -> int:
        """Modes whose unit-amplitude whitened response clears the registered
        detection threshold.  Reported separately from algebraic rank."""
        if self.operational_threshold is None:
            return -1
        return int(np.sum(self.singular_values >= self.operational_threshold))

    @property
    def operational_nullity(self) -> int:
        r = self.operational_rank
        return -1 if r < 0 else int(self.source_dimension - r)

    def threshold_sensitivity(self,
                              thresholds: Sequence[float] = REPORT_THRESHOLDS) -> dict[str, int]:
        """Rank under relative cut-offs, so threshold dependence is never hidden."""
        return {f"rank_rel_{t:.0e}": int(np.sum(self.singular_values > t * self.sigma_max))
                for t in thresholds}

    def summary(self) -> dict[str, Any]:
        d = {
            "sigma_max": self.sigma_max,
            "sigma_min_positive": self.sigma_min_positive,
            "numerical_rank": self.numerical_rank,
            "operational_rank": self.operational_rank,
            "nullity": self.nullity,
            "operational_nullity": self.operational_nullity,
            "kappa_positive": self.kappa_positive,
            "stable_rank": self.stable_rank,
            "effective_rank": self.effective_rank,
            "trace_information": self.trace_information,
            "primary_threshold": self.primary_threshold,
        }
        d.update(self.threshold_sensitivity())
        d.update(self.extra)
        return d


def spectrum_of(B: np.ndarray, source_dimension: int | None = None,
                operational_threshold: float | None = None) -> Spectrum:
    """Dense float64 spectrum.  ``B`` must already be whitened."""
    B = np.asarray(B, dtype=np.float64)
    s = sla.svd(B, compute_uv=False)
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    return Spectrum(s, B.shape, source_dimension or B.shape[1], operational_threshold)


def gram_spectrum(G: np.ndarray, source_dimension: int | None = None,
                  operational_threshold: float | None = None) -> Spectrum:
    """Spectrum recovered from a Gram matrix G = B^T B.

    Eigenvalues are clipped at zero before the square root: a symmetric PSD
    Gram can carry eigenvalues of order -1e-18 from rounding, and sqrt of those
    is NaN, which would silently poison every downstream rank count.
    """
    G = np.asarray(G, dtype=np.float64)
    w = sla.eigh(G, eigvals_only=True)
    w = np.clip(np.sort(np.asarray(w, dtype=np.float64))[::-1], 0.0, None)
    s = np.sqrt(w)
    d = source_dimension or G.shape[0]
    sp = Spectrum(s, (G.shape[0], G.shape[1]), d, operational_threshold)
    sp.extra["min_gram_eigenvalue"] = float(np.min(sla.eigh(G, eigvals_only=True)))
    return sp


def numerical_null_basis(B: np.ndarray, threshold: float | None = None,
                         verify: bool = True) -> np.ndarray:
    """Right singular vectors below the rank threshold, verified by matvec.

    A candidate is only labelled null if ``B v`` is actually small; the SVD
    threshold alone is a claim about the factorisation, not about the operator.
    """
    B = np.asarray(B, dtype=np.float64)
    U, s, Vt = sla.svd(B, full_matrices=True)
    smax = float(s[0]) if s.size else 0.0
    thr = threshold if threshold is not None else lapack_rank_threshold(B.shape, smax)
    keep = [i for i in range(Vt.shape[0]) if i >= s.size or s[i] <= thr]
    V = Vt[keep].T if keep else np.zeros((B.shape[1], 0))
    if verify and V.shape[1]:
        resid = np.linalg.norm(B @ V, axis=0) / max(smax, 1.0)
        V = V[:, resid <= max(thr / max(smax, 1.0), 1e-12) * 1e3 + 1e-12]
    return np.ascontiguousarray(V)


def smallest_visible_modes(B: np.ndarray, count: int = 20,
                           threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The ``count`` smallest *nonzero* right singular vectors and their values."""
    B = np.asarray(B, dtype=np.float64)
    U, s, Vt = sla.svd(B, full_matrices=False)
    smax = float(s[0]) if s.size else 0.0
    thr = threshold if threshold is not None else lapack_rank_threshold(B.shape, smax)
    idx = [i for i in range(s.size) if s[i] > thr]
    idx = idx[-count:] if len(idx) > count else idx
    return s[idx], Vt[idx].T
