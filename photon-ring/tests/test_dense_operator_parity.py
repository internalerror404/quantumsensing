import numpy as np
import pytest

from phrt.operators import mixing
from phrt.operators.historical import structured_operator
from phrt.operators.structured import ToySpec, build_order_blocks

SPECS = [
    ToySpec(n_cells=6, n_screen=2, spatial=s, delay=d)
    for s in ("identical", "rotation", "rotation_shear", "independent")
    for d in ("constant", "perturbed", "cell_dependent")
]


@pytest.mark.parametrize("spec", SPECS)
def test_matrix_free_matches_dense(spec):
    op = structured_operator(spec, 42)
    dense = mixing.resolved(spec.max_order + 1).apply(build_order_blocks(spec, 42))
    got = op.to_dense()
    assert got.shape == dense.shape
    denom = max(abs(dense).max(), 1e-300)
    assert abs(got - dense).max() / denom < 1e-10


def test_streamed_gram_matches_dense():
    spec = ToySpec(n_cells=6, n_screen=2, spatial="rotation")
    op = structured_operator(spec, 42)
    dense = op.to_dense()
    ref = dense.T @ dense
    assert abs(op.gram() - ref).max() / abs(ref).max() < 1e-10


def test_mixer_is_applied_matrix_free():
    """A non-identity mixer must give the same answer through either path."""
    spec = ToySpec(n_cells=6, n_screen=2, spatial="rotation")
    n = spec.max_order + 1
    L = mixing.leakage_matrix(n, 0.17)
    op = structured_operator(spec, 42, mixer=L)
    dense = mixing.OrderMixer(L).apply(build_order_blocks(spec, 42))
    assert abs(op.to_dense() - dense).max() / abs(dense).max() < 1e-10


def test_delay_ladder_that_does_not_fit_is_refused_not_clipped():
    """Clamping an over-long ladder to delay zero would silently delete the
    delay mechanism while the run still reported success."""
    spec = ToySpec(n_cells=6, n_screen=2, window=44, delay_step=4, max_order=5)
    with pytest.raises(ValueError, match="delay ladder does not fit"):
        structured_operator(spec, 42)
    with pytest.raises(ValueError, match="delay ladder does not fit"):
        build_order_blocks(spec, 42)


def test_window_longer_than_history_is_refused():
    spec = ToySpec(n_cells=6, n_screen=2, window=64, history_length=44,
                   delay_step=0, max_order=0)
    with pytest.raises(ValueError, match="exceeds history length"):
        build_order_blocks(spec, 42)


def test_registered_e0_spec_fits_exactly():
    spec = ToySpec(n_cells=6, n_screen=2)
    spec.validate()
    assert spec.consistent()
