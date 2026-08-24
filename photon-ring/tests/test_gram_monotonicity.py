import numpy as np
import pytest

from phrt.audits.gates import gate_gram_monotonicity
from phrt.operators.structured import ToySpec, build_order_blocks
from phrt.operators.whitening import NoiseModel, gram


def cumulative_grams(spec):
    blocks = build_order_blocks(spec, 42)
    out = []
    for n in range(len(blocks)):
        sub = np.vstack(blocks[: n + 1])
        out.append(gram(NoiseModel.homoscedastic(sub.shape[0]).whiten(sub)))
    return out


@pytest.mark.parametrize("spatial", ["identical", "rotation", "rotation_shear", "independent"])
def test_information_never_decreases_with_an_added_order(spatial):
    g = gate_gram_monotonicity(
        cumulative_grams(ToySpec(n_cells=6, n_screen=2, spatial=spatial)), 1e-10)
    assert g.status == "PASS"


def test_gate_catches_a_decreasing_sequence():
    grams = cumulative_grams(ToySpec(n_cells=6, n_screen=2))
    broken = list(grams)
    broken[-1] = broken[-1] - 0.5 * np.eye(broken[-1].shape[0]) * np.trace(broken[-1]) / broken[-1].shape[0]
    assert gate_gram_monotonicity(broken, 1e-10).status == "FAIL"
