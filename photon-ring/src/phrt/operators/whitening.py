"""Noise whitening.  Every reported singular value belongs to B = C^{-1/2} A."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla


@dataclass(frozen=True)
class NoiseModel:
    """Diagonal (row-wise) Gaussian noise.

    ``sigma`` is the per-row standard deviation.  Heteroscedastic arms -- weak
    orders measured with lower effective SNR -- are expressed by handing in a
    per-row sigma rather than by rescaling the operator, so the operator stays
    the physics and the noise stays the instrument.
    """

    sigma: np.ndarray
    name: str = "diagonal"

    @classmethod
    def homoscedastic(cls, n_rows: int, sigma: float = 1.0) -> "NoiseModel":
        return cls(np.full(int(n_rows), float(sigma)), f"homoscedastic_sigma{sigma:g}")

    @classmethod
    def from_snr(cls, A: np.ndarray, snr: float | None,
                 reference_rows: slice | None = None) -> "NoiseModel":
        """Whitened leading-order effective SNR.

        ``snr=None`` means the noise-free arm and is represented by unit sigma,
        which leaves the operator unchanged -- not by a zero sigma, which would
        divide by zero and is not what "noise free" means for a spectrum.
        """
        n_rows = A.shape[0]
        if snr is None:
            return cls(np.ones(n_rows), "noise_free")
        ref = A[reference_rows] if reference_rows is not None else A
        scale = float(np.sqrt(np.mean(np.sum(ref ** 2, axis=1))))
        sigma = np.full(n_rows, max(scale, 1e-300) / float(snr))
        return cls(sigma, f"snr{snr:g}")

    def whiten(self, A: np.ndarray) -> np.ndarray:
        if A.shape[0] != self.sigma.size:
            raise ValueError(f"noise model has {self.sigma.size} rows, operator has {A.shape[0]}")
        return A / self.sigma[:, None]

    def whiten_data(self, y: np.ndarray) -> np.ndarray:
        return y / self.sigma if y.ndim == 1 else y / self.sigma[:, None]

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        out = rng.normal(size=(n, self.sigma.size)) * self.sigma
        return out[0] if n == 1 else out


def gram(B: np.ndarray) -> np.ndarray:
    """Symmetrised B^T B.  The explicit symmetrisation matters: eigh on a
    matrix that is asymmetric at the 1e-17 level still returns real values, but
    the asymmetry shows up as a spurious negative eigenvalue in the
    monotonicity gate."""
    G = B.T @ B
    return 0.5 * (G + G.T)


def restrict(B: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Restriction to a source class with orthonormal basis Q (columns)."""
    if abs(float(np.abs(Q.T @ Q - np.eye(Q.shape[1])).max())) > 1e-10:
        raise ValueError("source-class basis must be orthonormal")
    return B @ Q


def inverse_sqrt_psd(C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """C^{-1/2} for a dense PSD covariance, for the non-diagonal case."""
    w, V = sla.eigh(0.5 * (C + C.T))
    w = np.clip(w, ridge, None)
    if np.any(w <= 0):
        raise ValueError("covariance is singular; supply a ridge")
    return V @ np.diag(w ** -0.5) @ V.T
