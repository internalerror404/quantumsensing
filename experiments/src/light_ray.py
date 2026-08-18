r"""Numerical primitives for weak-field Lorentzian light-ray experiments.

The module keeps the paper's convention
    A_gamma h = 1/2 \int h_{mu nu} k^mu k^nu d lambda,
with eta = diag(-1, 1, 1, 1) and k=(1, theta), |theta|=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss

Array = NDArray[np.float64]
ETA = np.diag(np.array([-1.0, 1.0, 1.0, 1.0]))


def normalize(v: Array, eps: float = 1e-15) -> Array:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / n


def null_vector(theta: Array) -> Array:
    theta = normalize(np.asarray(theta, dtype=float))
    return np.concatenate(([1.0], theta))


def fibonacci_sphere(count: int) -> Array:
    if count < 2:
        raise ValueError("count must be at least 2")
    i = np.arange(count, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / (count - 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    az = golden * i
    return np.stack((radius * np.cos(az), y, radius * np.sin(az)), axis=1)


def transverse_frame(theta: Array) -> tuple[Array, Array]:
    theta = normalize(theta)
    anchor = np.array([1.0, 0.0, 0.0])
    if abs(float(theta @ anchor)) > 0.9:
        anchor = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(theta, anchor))
    e2 = normalize(np.cross(theta, e1))
    return e1, e2


@dataclass(frozen=True)
class Ray:
    theta: Array
    offset: Array
    lam_min: float = -2.5
    lam_max: float = 2.5

    def __post_init__(self) -> None:
        theta = normalize(np.asarray(self.theta, dtype=float))
        offset = np.asarray(self.offset, dtype=float)
        if theta.shape != (3,) or offset.shape != (3,):
            raise ValueError("theta and offset must have shape (3,)")
        # Use the t=0 crossing convention offset . theta = 0.
        offset = offset - float(offset @ theta) * theta
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "offset", offset)

    @property
    def k(self) -> Array:
        return null_vector(self.theta)

    def points(self, lam: Array) -> Array:
        lam = np.asarray(lam, dtype=float)
        spatial = self.offset[None, :] + lam[:, None] * self.theta[None, :]
        return np.concatenate((lam[:, None], spatial), axis=1)


def candidate_rays(direction_count: int = 96, offsets_per_direction: int = 4,
                   offset_radius: float = 0.45, lam_extent: float = 2.5) -> list[Ray]:
    rays: list[Ray] = []
    for idx, theta in enumerate(fibonacci_sphere(direction_count)):
        e1, e2 = transverse_frame(theta)
        local_offsets = [np.zeros(3)]
        for j in range(1, offsets_per_direction):
            angle = 2.0 * np.pi * ((idx + j) % max(3, offsets_per_direction - 1)) / max(3, offsets_per_direction - 1)
            local_offsets.append(offset_radius * (np.cos(angle) * e1 + np.sin(angle) * e2))
        for off in local_offsets[:offsets_per_direction]:
            rays.append(Ray(theta=theta, offset=off, lam_min=-lam_extent, lam_max=lam_extent))
    return rays


@lru_cache(maxsize=32)
def gauss_legendre_interval(a: float, b: float, order: int = 512) -> tuple[Array, Array]:
    nodes, weights = leggauss(order)
    lam = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w = 0.5 * (b - a) * weights
    return lam.astype(float), w.astype(float)


def smooth_bump_1d(u: Array) -> tuple[Array, Array]:
    """Return b(u) and db/du for the standard C-infinity bump.

    b(u)=exp(-1/(1-u^2)) for |u|<1, otherwise 0.
    """
    u = np.asarray(u, dtype=float)
    b = np.zeros_like(u)
    db = np.zeros_like(u)
    mask = np.abs(u) < 1.0
    um = u[mask]
    denom = 1.0 - um * um
    bm = np.exp(-1.0 / denom)
    b[mask] = bm
    db[mask] = bm * (-2.0 * um) / (denom * denom)
    return b, db


def product_bump(points: Array, center: Array, radii: Array) -> tuple[Array, Array]:
    """C-infinity product bump and its coordinate gradient in R^4."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape (N,4)")
    u = (points - center[None, :]) / radii[None, :]
    vals = []
    ders = []
    for j in range(4):
        b, db = smooth_bump_1d(u[:, j])
        vals.append(b)
        ders.append(db / radii[j])
    vals_arr = np.stack(vals, axis=1)
    phi = np.prod(vals_arr, axis=1)
    grad = np.zeros_like(points)
    for j in range(4):
        other = np.prod(np.delete(vals_arr, j, axis=1), axis=1)
        grad[:, j] = ders[j] * other
    return phi, grad


@dataclass(frozen=True)
class TensorMode:
    center: Array
    radii: Array
    polarization: Array
    label: str

    def contraction(self, points: Array, k: Array) -> Array:
        phi, _ = product_bump(points, self.center, self.radii)
        scalar = float(k @ self.polarization @ k)
        return phi * scalar


def ray_integral_contraction(ray: Ray, contraction_fn, order: int = 512) -> float:
    lam, weights = gauss_legendre_interval(ray.lam_min, ray.lam_max, order=order)
    points = ray.points(lam)
    vals = np.asarray(contraction_fn(points, ray.k), dtype=float)
    return 0.5 * float(weights @ vals)


def mode_matrix(rays: Iterable[Ray], modes: Iterable[TensorMode], order: int = 256) -> Array:
    ray_list = list(rays)
    mode_list = list(modes)
    out = np.empty((len(ray_list), len(mode_list)), dtype=float)
    for a, ray in enumerate(ray_list):
        lam, weights = gauss_legendre_interval(ray.lam_min, ray.lam_max, order=order)
        points = ray.points(lam)
        k = ray.k
        for i, mode in enumerate(mode_list):
            out[a, i] = 0.5 * float(weights @ mode.contraction(points, k))
    return out


def conformal_contraction(points: Array, k: Array, center: Array, radii: Array) -> Array:
    phi, _ = product_bump(points, center, radii)
    return phi * float(k @ ETA @ k)


def gauge_contraction(points: Array, k: Array, center: Array, radii: Array, covector: Array) -> Array:
    """Contraction of h=2 d^s V with V_nu = covector_nu * bump."""
    _, grad = product_bump(points, center, radii)
    k_dot_grad = grad @ k
    a_dot_k = float(np.asarray(covector, dtype=float) @ k)
    return 2.0 * k_dot_grad * a_dot_k


def endpoint_gauge_contraction(points: Array, k: Array, covector: Array) -> Array:
    """Noncompact endpoint test: V_nu=a_nu f(t), f=.35+.2t+.1t^2-.03t^3."""
    t = points[:, 0]
    fp = 0.2 + 0.2 * t - 0.09 * t * t
    return 2.0 * float(np.asarray(covector) @ k) * fp


def endpoint_gauge_value(t: float, k: Array, covector: Array) -> float:
    f = 0.35 + 0.2 * t + 0.1 * t * t - 0.03 * t**3
    return float(np.asarray(covector) @ k) * f


def make_physical_modes() -> list[TensorMode]:
    centers = [
        np.array([-0.55, -0.45, -0.15, 0.10]),
        np.array([-0.15, 0.35, -0.35, -0.10]),
        np.array([0.25, -0.20, 0.40, 0.20]),
        np.array([0.55, 0.25, 0.05, -0.35]),
    ]
    radii = np.array([0.72, 0.58, 0.58, 0.58])
    modes: list[TensorMode] = []

    # Newtonian-like scalar modes: g_00 and spatial trace change with the same sign.
    p_scalar = np.diag(np.array([-2.0, -2.0, -2.0, -2.0]))
    for j, c in enumerate(centers):
        modes.append(TensorMode(c, radii, p_scalar, f"scalar_{j}"))

    # Gravitomagnetic h_0i modes.
    for j, axis in enumerate(range(1, 4)):
        p = np.zeros((4, 4))
        p[0, axis] = p[axis, 0] = 1.0
        modes.append(TensorMode(centers[j], radii, p, f"frame_drag_{axis}"))

    # Spatial trace-free modes.
    p1 = np.zeros((4, 4)); p1[1, 1] = 1.0; p1[2, 2] = -1.0
    p2 = np.zeros((4, 4)); p2[1, 2] = p2[2, 1] = 1.0
    p3 = np.zeros((4, 4)); p3[1, 1] = 1.0; p3[3, 3] = -1.0
    p4 = np.zeros((4, 4)); p4[1, 3] = p4[3, 1] = 1.0
    for j, p in enumerate((p1, p2, p3, p4)):
        modes.append(TensorMode(centers[j], radii, p, f"anisotropic_{j}"))

    # One additional mixed spatial mode for d=12.
    p5 = np.zeros((4, 4)); p5[2, 3] = p5[3, 2] = 1.0
    modes.append(TensorMode(np.array([0.0, 0.0, 0.0, 0.0]), radii, p5, "anisotropic_4"))
    return modes
