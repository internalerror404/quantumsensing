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


def make_physical_modes(count: int = 12) -> list[TensorMode]:
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

    if count > 12:
        # Extension to d=16 for the registered scaling surface. New centers keep
        # the bumps inside the interior world tube; polarizations repeat earlier
        # families at distinct locations, so independence comes from support.
        extra_centers = [
            np.array([-0.35, 0.50, 0.30, -0.20]),
            np.array([0.45, -0.40, -0.30, 0.15]),
            np.array([0.10, 0.45, 0.25, 0.30]),
            np.array([-0.50, -0.10, 0.45, -0.05]),
        ]
        p6 = np.diag(np.array([-2.0, -2.0, -2.0, -2.0]))
        p7 = np.zeros((4, 4)); p7[0, 1] = p7[1, 0] = 1.0
        p8 = np.zeros((4, 4)); p8[2, 2] = 1.0; p8[3, 3] = -1.0
        p9 = np.zeros((4, 4)); p9[0, 3] = p9[3, 0] = 1.0
        for j, (c, pol, label) in enumerate(zip(
                extra_centers, (p6, p7, p8, p9),
                ("scalar_ext", "frame_drag_ext_1", "anisotropic_ext", "frame_drag_ext_3"))):
            modes.append(TensorMode(c, radii, pol, label))
    if count > len(modes):
        raise ValueError(f"at most {len(modes)} modes available")
    return modes[:count]


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


# ---------------------------------------------------------------------------
# Static endpoint-clock channels
# ---------------------------------------------------------------------------


def static_product_bump(spatial_points: Array, center: Array,
                        radii: Array) -> tuple[Array, Array]:
    """A time-independent C-infinity product bump on R^3 and its gradient.

    The static-redshift section uses spatial fields phi(x) with no time
    dependence.  Keeping this separate from :func:`product_bump` prevents a
    time-localized four-dimensional test field from being mistaken for a
    stationary perturbation.
    """
    spatial_points = np.asarray(spatial_points, dtype=float)
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if spatial_points.ndim != 2 or spatial_points.shape[1] != 3:
        raise ValueError("spatial_points must have shape (N,3)")
    if center.shape != (3,) or radii.shape != (3,):
        raise ValueError("center and radii must have shape (3,)")
    if np.any(radii <= 0.0):
        raise ValueError("radii must be strictly positive")

    u = (spatial_points - center[None, :]) / radii[None, :]
    vals: list[Array] = []
    ders: list[Array] = []
    for j in range(3):
        b, db = smooth_bump_1d(u[:, j])
        vals.append(b)
        ders.append(db / radii[j])
    vals_arr = np.stack(vals, axis=1)
    phi = np.prod(vals_arr, axis=1)
    grad = np.zeros_like(spatial_points)
    for j in range(3):
        other = np.prod(np.delete(vals_arr, j, axis=1), axis=1)
        grad[:, j] = ders[j] * other
    return phi, grad


@dataclass(frozen=True)
class StaticObserver:
    """A static observer labelled by its background spatial position."""

    position: Array
    label: str

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,):
            raise ValueError("StaticObserver.position must have shape (3,)")
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class StaticClockLink:
    """An oriented log-redshift comparison from emitter A to receiver B."""

    emitter: StaticObserver
    receiver: StaticObserver
    label: str = ""

    def __post_init__(self) -> None:
        if self.emitter.label == self.receiver.label and np.allclose(
                self.emitter.position, self.receiver.position):
            raise ValueError("A clock link requires distinct endpoints")
        if not self.label:
            object.__setattr__(
                self, "label", f"{self.emitter.label}->{self.receiver.label}"
            )


@dataclass(frozen=True)
class StaticConformalMode:
    """A stationary conformal perturbation h = phi(x) eta."""

    center: Array
    radii: Array
    label: str
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        if center.shape != (3,) or radii.shape != (3,):
            raise ValueError("center and radii must have shape (3,)")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be strictly positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radii", radii)

    def phi(self, spatial_points: Array) -> Array:
        values, _ = static_product_bump(spatial_points, self.center, self.radii)
        return float(self.amplitude) * values

    def tensor(self, spatial_points: Array) -> Array:
        """Return h_{mu nu}=phi eta, shape (N,4,4)."""
        values = self.phi(spatial_points)
        return values[:, None, None] * ETA[None, :, :]

    def contraction(self, spacetime_points: Array, k: Array) -> Array:
        """Return h(k,k), assembled componentwise along a null ray."""
        spacetime_points = np.asarray(spacetime_points, dtype=float)
        if spacetime_points.ndim != 2 or spacetime_points.shape[1] != 4:
            raise ValueError("spacetime_points must have shape (N,4)")
        h = self.tensor(spacetime_points[:, 1:])
        return np.einsum("nij,i,j->n", h, k, k)


@dataclass(frozen=True)
class StaticGaugeMode:
    r"""A stationary endpoint-fixed pure-gauge perturbation h=2 d^s V.

    The one-form is V_nu(x)=a_nu psi(x), with no t dependence.  If
    ``zero_point`` is supplied, psi=(x_axis-zero_point_axis) phi, so V vanishes
    at that endpoint while its derivative can remain nonzero.  Taking a_0=0
    preserves the static zero-shift ansatz h_{0i}=0.  The assembled tensor may
    then be nonzero at a clock while h_{00}=2 partial_0 V_0=0.
    """

    center: Array
    radii: Array
    covector: Array
    label: str
    zero_point: Array | None = None
    zero_axis: int = 0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        covector = np.asarray(self.covector, dtype=float)
        if center.shape != (3,) or radii.shape != (3,):
            raise ValueError("center and radii must have shape (3,)")
        if covector.shape != (4,):
            raise ValueError("covector must have shape (4,)")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be strictly positive")
        if not 0 <= int(self.zero_axis) < 3:
            raise ValueError("zero_axis must be one of {0,1,2}")
        zero_point = self.zero_point
        if zero_point is not None:
            zero_point = np.asarray(zero_point, dtype=float)
            if zero_point.shape != (3,):
                raise ValueError("zero_point must have shape (3,)")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "covector", covector)
        object.__setattr__(self, "zero_point", zero_point)
        object.__setattr__(self, "zero_axis", int(self.zero_axis))

    def scalar_and_gradient(self, spatial_points: Array) -> tuple[Array, Array]:
        phi, grad3 = static_product_bump(spatial_points, self.center, self.radii)
        if self.zero_point is None:
            return phi, grad3
        axis = self.zero_axis
        factor = np.asarray(spatial_points, dtype=float)[:, axis] - self.zero_point[axis]
        psi = factor * phi
        grad_psi = factor[:, None] * grad3
        grad_psi[:, axis] += phi
        return psi, grad_psi

    def covector_field(self, spatial_points: Array) -> Array:
        psi, _ = self.scalar_and_gradient(spatial_points)
        return psi[:, None] * self.covector[None, :]

    def tensor(self, spatial_points: Array) -> Array:
        """Assemble h_{mu nu}=partial_mu V_nu+partial_nu V_mu."""
        _, grad3 = self.scalar_and_gradient(spatial_points)
        n = grad3.shape[0]
        grad4 = np.zeros((n, 4), dtype=float)
        grad4[:, 1:] = grad3  # partial_0 V_nu = 0 by stationarity.
        d_v = grad4[:, :, None] * self.covector[None, None, :]
        return d_v + np.swapaxes(d_v, 1, 2)


def static_conformal_delay_matrix(rays: Iterable[Ray],
                                  modes: Iterable[StaticConformalMode],
                                  order: int = DEFAULT_ORDER) -> Array:
    """Delay-channel matrix for stationary conformal modes.

    Every entry vanishes analytically because eta(k,k)=0.  The componentwise
    assembly makes this a numerical sign/index test rather than a factored-out
    scalar identity.
    """
    ray_list = list(rays)
    mode_list = list(modes)
    out = np.empty((len(ray_list), len(mode_list)), dtype=float)
    for a, ray in enumerate(ray_list):
        lam, weights = gauss_legendre_interval(
            ray.lam_min, ray.lam_max, order=order
        )
        points = ray.points(lam)
        for j, mode in enumerate(mode_list):
            out[a, j] = 0.5 * float(weights @ mode.contraction(points, ray.k))
    return out


def static_redshift_matrix(links: Iterable[StaticClockLink],
                           modes: Iterable[StaticConformalMode]) -> Array:
    r"""Endpoint log-redshift matrix for static conformal perturbations.

    The generic restricted functional is
        R_AB h = 1/2 [h_00(B)-h_00(A)].
    Since h_00=-phi for h=phi eta with eta_00=-1, this yields
        R_AB(phi eta)=1/2 [phi(A)-phi(B)].
    """
    link_list = list(links)
    mode_list = list(modes)
    out = np.empty((len(link_list), len(mode_list)), dtype=float)
    for ell, link in enumerate(link_list):
        point_a = link.emitter.position[None, :]
        point_b = link.receiver.position[None, :]
        for j, mode in enumerate(mode_list):
            h_a = mode.tensor(point_a)[0]
            h_b = mode.tensor(point_b)[0]
            out[ell, j] = 0.5 * float(h_b[0, 0] - h_a[0, 0])
    return out


def static_redshift_formula_matrix(links: Iterable[StaticClockLink],
                                   modes: Iterable[StaticConformalMode]) -> Array:
    """Independent endpoint-difference expression 1/2[phi(A)-phi(B)]."""
    link_list = list(links)
    mode_list = list(modes)
    out = np.empty((len(link_list), len(mode_list)), dtype=float)
    for ell, link in enumerate(link_list):
        point_a = link.emitter.position[None, :]
        point_b = link.receiver.position[None, :]
        for j, mode in enumerate(mode_list):
            phi_a = float(mode.phi(point_a)[0])
            phi_b = float(mode.phi(point_b)[0])
            out[ell, j] = 0.5 * (phi_a - phi_b)
    return out


def static_gauge_redshift_matrix(links: Iterable[StaticClockLink],
                                 modes: Iterable[StaticGaugeMode]) -> Array:
    """Redshift response of assembled stationary pure-gauge perturbations."""
    link_list = list(links)
    mode_list = list(modes)
    out = np.empty((len(link_list), len(mode_list)), dtype=float)
    for ell, link in enumerate(link_list):
        point_a = link.emitter.position[None, :]
        point_b = link.receiver.position[None, :]
        for j, mode in enumerate(mode_list):
            h_a = mode.tensor(point_a)[0]
            h_b = mode.tensor(point_b)[0]
            out[ell, j] = 0.5 * float(h_b[0, 0] - h_a[0, 0])
    return out


@dataclass(frozen=True)
class StaticTensorMode:
    """A stationary tensor perturbation h = phi(x) * polarization.

    Generic counterpart of :class:`StaticConformalMode`: any constant symmetric
    polarization with a stationary scalar profile. Modes with h_0i = 0 stay
    inside the static-metric ansatz under which the endpoint redshift formula
    R_AB h = 1/2 [h_00(B) - h_00(A)] is derived; the redshift matrix should not
    be applied to polarizations with nonzero h_0i.
    """

    center: Array
    radii: Array
    polarization: Array
    label: str

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        polarization = np.asarray(self.polarization, dtype=float)
        if center.shape != (3,) or radii.shape != (3,):
            raise ValueError("center and radii must have shape (3,)")
        if polarization.shape != (4, 4):
            raise ValueError("polarization must have shape (4,4)")
        if not np.allclose(polarization, polarization.T):
            raise ValueError("polarization must be symmetric")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be strictly positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "polarization", polarization)

    def phi(self, spatial_points: Array) -> Array:
        values, _ = static_product_bump(spatial_points, self.center, self.radii)
        return values

    def tensor(self, spatial_points: Array) -> Array:
        values = self.phi(spatial_points)
        return values[:, None, None] * self.polarization[None, :, :]

    def contraction(self, spacetime_points: Array, k: Array) -> Array:
        spacetime_points = np.asarray(spacetime_points, dtype=float)
        if spacetime_points.ndim != 2 or spacetime_points.shape[1] != 4:
            raise ValueError("spacetime_points must have shape (N,4)")
        h = self.tensor(spacetime_points[:, 1:])
        return np.einsum("nij,i,j->n", h, k, k)


def static_tensor_delay_matrix(rays: Iterable[Ray], modes: Iterable,
                               order: int = DEFAULT_ORDER) -> Array:
    """Delay matrix for any stationary modes exposing .contraction(points, k)."""
    ray_list = list(rays)
    mode_list = list(modes)
    out = np.empty((len(ray_list), len(mode_list)), dtype=float)
    for a, ray in enumerate(ray_list):
        lam, weights = gauss_legendre_interval(ray.lam_min, ray.lam_max, order=order)
        points = ray.points(lam)
        for j, mode in enumerate(mode_list):
            out[a, j] = 0.5 * float(weights @ mode.contraction(points, ray.k))
    return out


def static_tensor_redshift_matrix(links: Iterable[StaticClockLink],
                                  modes: Iterable,
                                  h0i_tolerance: float = 1e-14) -> Array:
    """Endpoint matrix R_AB h = 1/2 [h_00(B) - h_00(A)] for stationary modes.

    Valid only within the static ansatz (stationary h, h_0i = 0), under which
    the paper derives the endpoint formula. Enforced at runtime: a mode whose
    assembled tensor carries h_0i above tolerance at any clock endpoint raises,
    rather than silently producing rows the formula does not cover.
    """
    link_list = list(links)
    mode_list = list(modes)
    endpoints = np.vstack([
        np.vstack((link.emitter.position, link.receiver.position))
        for link in link_list
    ]) if link_list else np.zeros((0, 3))
    for mode in mode_list:
        h = mode.tensor(endpoints)
        h0i = float(np.max(np.abs(h[:, 0, 1:]))) if h.shape[0] else 0.0
        if h0i > h0i_tolerance:
            raise ValueError(
                f"mode {getattr(mode, 'label', mode)!r} has |h_0i| = {h0i:.3e} "
                "at a clock endpoint; the static redshift formula "
                "R_AB h = 1/2 [h_00(B) - h_00(A)] does not apply outside the "
                "h_0i = 0 static ansatz"
            )
    out = np.empty((len(link_list), len(mode_list)), dtype=float)
    for ell, link in enumerate(link_list):
        point_a = link.emitter.position[None, :]
        point_b = link.receiver.position[None, :]
        for j, mode in enumerate(mode_list):
            h_a = mode.tensor(point_a)[0]
            h_b = mode.tensor(point_b)[0]
            out[ell, j] = 0.5 * float(h_b[0, 0] - h_a[0, 0])
    return out
