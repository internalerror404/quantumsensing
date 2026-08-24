"""Principal angles between visible and null subspaces."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from phrt.audits.rank import lapack_rank_threshold


def orthonormalise(V: np.ndarray) -> np.ndarray:
    if V.size == 0 or V.shape[1] == 0:
        return V.reshape(V.shape[0], 0)
    Q, R = np.linalg.qr(V)
    keep = np.abs(np.diag(R)) > 1e-12 * max(np.abs(np.diag(R)).max(), 1e-300)
    return Q[:, keep]


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Angles in radians, ascending.  Unequal dimensions are allowed; the
    result has min(dim U, dim V) entries and the dimensions are reported by the
    caller, because an angle list alone hides a nullity mismatch."""
    U, V = orthonormalise(U), orthonormalise(V)
    if U.shape[1] == 0 or V.shape[1] == 0:
        return np.zeros(0)
    return np.asarray(sla.subspace_angles(U, V))[::-1]


def visible_subspace(B: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Right singular vectors above the rank threshold (row space of B)."""
    U, s, Vt = sla.svd(np.asarray(B, dtype=np.float64), full_matrices=False)
    thr = threshold if threshold is not None else lapack_rank_threshold(B.shape, float(s[0]))
    return Vt[s > thr].T


def subspace_report(U: np.ndarray, V: np.ndarray) -> dict[str, float | int | list[float]]:
    ang = principal_angles(U, V)
    return {
        "dim_a": int(orthonormalise(U).shape[1]),
        "dim_b": int(orthonormalise(V).shape[1]),
        "n_angles": int(ang.size),
        "max_angle_rad": float(ang[-1]) if ang.size else float("nan"),
        "median_angle_rad": float(np.median(ang)) if ang.size else float("nan"),
        "min_angle_rad": float(ang[0]) if ang.size else float("nan"),
        "angles_rad": [float(a) for a in ang[:20]],
    }
