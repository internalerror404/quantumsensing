import numpy as np
import pytest

from phrt.audits.gates import gate_adjoint
from phrt.operators import mixing
from phrt.operators.historical import structured_operator
from phrt.operators.structured import ToySpec


@pytest.mark.parametrize("seed", range(20))
def test_inner_product_adjoint_over_registered_seeds(seed):
    """<Ax, y> == <x, A*y>, the registered gate, over 20 seeds."""
    spec = ToySpec(n_cells=6, n_screen=2, spatial="rotation_shear")
    op = structured_operator(spec, seed)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=op.shape[1])
    y = rng.normal(size=op.shape[0])
    a = float(y @ op.matvec(x))
    b = float(x @ op.rmatvec(y))
    assert abs(a - b) / max(abs(a), abs(b), 1e-300) < 1e-8


def test_adjoint_gate_reports_worst_not_mean():
    spec = ToySpec(n_cells=6, n_screen=2)
    g = gate_adjoint(structured_operator(spec, 42), 1e-8)
    assert g.status == "PASS" and g.measured < 1e-8


def test_adjoint_gate_catches_a_broken_transpose():
    """A deliberately wrong rmatvec must fail the gate, not slip through."""
    spec = ToySpec(n_cells=6, n_screen=2)
    op = structured_operator(spec, 42)
    good = op._rmatvec
    op._rmatvec = lambda y: good(y) * 1.001
    assert gate_adjoint(op, 1e-8).status == "FAIL"
