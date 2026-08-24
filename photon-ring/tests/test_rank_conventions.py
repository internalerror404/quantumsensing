import numpy as np
import pytest

from phrt.audits.rank import Spectrum, gram_spectrum, spectrum_of
from phrt.audits.subspaces import principal_angles, visible_subspace


def test_rank_nullity_and_kappa_on_a_known_operator():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(30, 10))
    B[:, -3:] = 0.0
    sp = spectrum_of(B)
    assert sp.numerical_rank == 7 and sp.nullity == 3
    assert np.isfinite(sp.kappa_positive)


def test_gram_route_agrees_with_svd_route():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(40, 12))
    a, b = spectrum_of(B), gram_spectrum(B.T @ B)
    assert abs(a.numerical_rank - b.numerical_rank) <= 0
    assert np.allclose(a.singular_values, b.singular_values, atol=1e-8)


def test_gram_route_survives_tiny_negative_eigenvalues():
    """A PSD Gram with -1e-18 rounding must not produce NaN singular values."""
    rng = np.random.default_rng(2)
    B = rng.normal(size=(20, 6))
    G = B.T @ B
    G = G - np.eye(6) * 1e-18
    sp = gram_spectrum(G)
    assert np.isfinite(sp.singular_values).all()


def test_operational_rank_is_separate_from_algebraic_rank():
    s = np.array([1.0, 1e-3, 1e-9])
    sp = Spectrum(s, (3, 3), 3, operational_threshold=1e-6)
    assert sp.numerical_rank == 3
    assert sp.operational_rank == 2 and sp.operational_nullity == 1


def test_unset_operational_threshold_reports_minus_one_not_a_fake_count():
    sp = Spectrum(np.array([1.0, 0.5]), (2, 2), 2)
    assert sp.operational_rank == -1


def test_principal_angles_of_a_subspace_with_itself_are_zero():
    rng = np.random.default_rng(3)
    B = rng.normal(size=(30, 8))
    V = visible_subspace(B)
    assert principal_angles(V, V).max() < 1e-8
