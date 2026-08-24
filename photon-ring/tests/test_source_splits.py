import pytest

from phrt.config import load_registry
from phrt.seeds import SeedNamespaces, rng


def test_namespaces_are_disjoint():
    ns = SeedNamespaces.from_registry(load_registry())
    assert ns.disjoint() and ns.overlaps() == []


def test_seed_index_is_stable_and_bounded():
    ns = SeedNamespaces.from_registry(load_registry())
    assert ns.take("train", 3) == [100000, 100001, 100002]
    with pytest.raises(IndexError):
        ns.seed("null_pair", ns.size("null_pair"))


def test_rng_key_is_process_stable():
    """Python's hash() is salted; the key folder must not depend on it."""
    a = rng("E1", 3, "delay").normal(size=4)
    b = rng("E1", 3, "delay").normal(size=4)
    c = rng("E1", 3, "spatial").normal(size=4)
    assert (a == b).all() and not (a == c).all()
