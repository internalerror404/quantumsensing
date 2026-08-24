import numpy as np
import pytest

from phrt.audits.gates import gate_kernel_injection
from phrt.audits.rank import numerical_null_basis
from phrt.operators import mixing
from phrt.operators.structured import ToySpec, build_order_blocks
from phrt.operators.whitening import NoiseModel


def _whitened(spec, readout="unresolved"):
    blocks = build_order_blocks(spec, 42)
    n = spec.max_order + 1
    m = mixing.resolved(n) if readout == "resolved" else mixing.unresolved(n)
    A = m.apply(blocks)
    return NoiseModel.homoscedastic(A.shape[0]).whiten(A)


def test_null_vector_changes_source_but_not_data():
    B = _whitened(ToySpec(n_cells=6, n_screen=2, spatial="rotation"))
    g = gate_kernel_injection(B, 1e-8)
    assert g.status == "PASS"


def test_every_returned_null_vector_is_verified_by_matvec():
    B = _whitened(ToySpec(n_cells=6, n_screen=2, spatial="rotation"))
    V = numerical_null_basis(B)
    assert V.shape[1] > 0
    smax = np.linalg.svd(B, compute_uv=False)[0]
    assert (np.linalg.norm(B @ V, axis=0) / smax).max() < 1e-10


def test_trivial_kernel_is_not_run_rather_than_pass():
    """An operator with no null space must not report a passed injection test."""
    rng = np.random.default_rng(0)
    B = rng.normal(size=(40, 10))
    assert gate_kernel_injection(B, 1e-8).status == "NOT_RUN"
