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

# Package-wide Gauss-Legendre order. The compact-support gauge cancellation
# A_gamma(2 d^s V) = 0 is quadrature limited; measured over the full candidate
# pool it is 3.3e-07 at order 256 and 1.2e-12 at order 1024. Every matrix that
# enters a rank or design metric is built at this order so that the pinned
# 1e-10 gate applies to the same numbers the experiments use.
DEFAULT_ORDER = 1024


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


@lru_cache(maxsize=64)
def gauss_legendre_interval(a: float, b: float, order: int = DEFAULT_ORDER) -> tuple[Array, Array]:
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


def ray_integral_contraction(ray: Ray, contraction_fn, order: int = DEFAULT_ORDER) -> float:
    lam, weights = gauss_legendre_interval(ray.lam_min, ray.lam_max, order=order)
    points = ray.points(lam)
    vals = np.asarray(contraction_fn(points, ray.k), dtype=float)
    return 0.5 * float(weights @ vals)


def mode_matrix(rays: Iterable[Ray], modes: Iterable[TensorMode], order: int = DEFAULT_ORDER) -> Array:
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
    """Contraction of h = phi * eta, assembled componentwise.

    The scalar eta(k,k) is deliberately *not* factored out. Building h_{mu nu}
    as a full tensor field and contracting index by index means the gate tests
    the assembled contraction rather than a single float, and the cancellation
    has to happen across sixteen accumulated products.
    """
    phi, _ = product_bump(points, center, radii)
    h = phi[:, None, None] * ETA[None, :, :]
    return np.einsum("nij,i,j->n", h, k, k)


def conformal_contraction_scale(points: Array, k: Array, center: Array, radii: Array) -> Array:
    """Magnitude scale of the gate-1 integrand, for a relative tolerance.

    Without this the 1e-14 gate silently tracks the amplitude of phi.
    """
    phi, _ = product_bump(points, center, radii)
    return np.abs(phi) * float(np.abs(k) @ np.abs(ETA) @ np.abs(k))


def gauge_contraction(points: Array, k: Array, center: Array, radii: Array, covector: Array) -> Array:
    """Contraction of h=2 d^s V with V_nu = covector_nu * bump."""
    _, grad = product_bump(points, center, radii)
    k_dot_grad = grad @ k
    a_dot_k = float(np.asarray(covector, dtype=float) @ k)
    return 2.0 * k_dot_grad * a_dot_k


def _endpoint_profile(t: Array) -> tuple[Array, Array]:
    """Non-polynomial V profile f(t) and its derivative.

    A polynomial profile is integrated *exactly* by Gauss-Legendre, so the gate
    could not fail. This profile is analytic but not polynomial, so the endpoint
    identity is tested against genuine quadrature error.
    """
    t = np.asarray(t, dtype=float)
    f = 0.35 * np.exp(0.4 * np.sin(1.7 * t)) + 0.18 * np.cos(0.9 * t)
    df = (0.35 * 0.4 * 1.7 * np.cos(1.7 * t) * np.exp(0.4 * np.sin(1.7 * t))
          - 0.18 * 0.9 * np.sin(0.9 * t))
    return f, df


def endpoint_gauge_contraction(points: Array, k: Array, covector: Array) -> Array:
    """Noncompact endpoint test: h = 2 d^s V with V_nu = a_nu f(t).

    Assembled as a full symmetric tensor and contracted, so this shares the
    gate-2 code path instead of hand-differentiating the answer.
    """
    t = points[:, 0]
    _, df = _endpoint_profile(t)
    a = np.asarray(covector, dtype=float)
    # dV_nu/dx_mu = a_nu f'(t) delta_mu^0, so (d^s V)_{mu nu} = 1/2 (a_nu f' d_mu0 + a_mu f' d_nu0).
    e0 = np.zeros(4); e0[0] = 1.0
    sym = 0.5 * (np.outer(e0, a) + np.outer(a, e0))
    h = 2.0 * df[:, None, None] * sym[None, :, :]
    return np.einsum("nij,i,j->n", h, k, k)


def endpoint_gauge_value(t: float, k: Array, covector: Array) -> float:
    f, _ = _endpoint_profile(np.array([float(t)]))
    return float(np.asarray(covector) @ k) * float(f[0])


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


def gaussian_packet_frequency_variance(s_width: float, half_extent: float = 14.0,
                                       order: int = 4096) -> float:
    """Var(nu-hat) for the transform-limited Gaussian temporal mode, by quadrature.

    psi_s(t) = (2 pi s^2)^{-1/4} exp(-t^2 / (4 s^2)),  nu-hat = -i d/dt.

    For a real, even, normalized psi:  <nu-hat> = 0  and
    Var(nu-hat) = <psi| -d^2/dt^2 |psi> = integral (dpsi/dt)^2 dt.

    Corollary 3.2 asserts this equals 1/(4 s^2). Computing it rather than
    asserting it is what makes the packet-width gate a test of the physics.
    """
    a = half_extent * s_width
    t, w = gauss_legendre_interval(-a, a, order=order)
    norm = (2.0 * np.pi * s_width ** 2) ** -0.25
    psi = norm * np.exp(-t * t / (4.0 * s_width ** 2))
    dpsi = psi * (-t / (2.0 * s_width ** 2))
    normalization = float(w @ (psi * psi))
    return float(w @ (dpsi * dpsi)) / normalization
