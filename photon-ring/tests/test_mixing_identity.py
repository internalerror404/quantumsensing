import numpy as np
import pytest

from phrt.operators import mixing
from phrt.operators.structured import ToySpec, build_order_blocks

SPEC = ToySpec(n_cells=6, n_screen=2, spatial="rotation")
BLOCKS = build_order_blocks(SPEC, 42)
N = SPEC.max_order + 1


def test_resolved_to_unresolved_collapse():
    direct = sum(BLOCKS)
    mixed = mixing.unresolved(N).apply(BLOCKS)
    assert abs(direct - mixed).max() / abs(direct).max() < 1e-10


def test_leakage_endpoints():
    assert abs(mixing.leakage_matrix(N, 0.0) - np.eye(N)).max() == 0.0
    L1 = mixing.leakage_matrix(N, 1.0)
    assert np.linalg.matrix_rank(L1) == 1


@pytest.mark.parametrize("eps", [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0])
def test_leakage_rows_sum_to_one(eps):
    L = mixing.leakage_matrix(N, eps)
    assert np.allclose(L.sum(axis=1), 1.0)


def test_leakage_epsilon_is_bounded():
    with pytest.raises(ValueError):
        mixing.leakage_matrix(N, 1.5)


def test_two_channel_separates_direct_from_indirect():
    L = mixing.two_channel_matrix(N)
    assert L[0, 0] == 1.0 and L[0, 1:].sum() == 0.0
    assert np.isclose(L[1, 1:].sum(), 1.0) and L[1, 0] == 0.0


def test_miscalibration_does_not_renormalise_rows():
    rng = np.random.default_rng(0)
    L = mixing.leakage_matrix(N, 0.1)
    Lm = mixing.miscalibrate(L, 0.10, rng)
    assert not np.allclose(Lm.sum(axis=1), 1.0)
