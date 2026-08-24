"""Deterministic seed management and the registered source-seed namespaces.

Namespaces must never overlap (gate G9).  Seeds are drawn by *index within a
namespace*, never by a global counter, so adding an experiment cannot shift the
sources another experiment already used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phrt.config import Registry

SPLITS = ("train", "validation", "test_id", "test_ood", "null_pair")


@dataclass(frozen=True)
class SeedNamespaces:
    ranges: dict[str, tuple[int, int]]

    @classmethod
    def from_registry(cls, reg: Registry) -> "SeedNamespaces":
        raw = reg.data["source_seed_namespaces"]
        return cls({k: (int(v[0]), int(v[1])) for k, v in raw.items()})

    def size(self, split: str) -> int:
        lo, hi = self.ranges[split]
        return hi - lo + 1

    def seed(self, split: str, index: int) -> int:
        lo, hi = self.ranges[split]
        n = hi - lo + 1
        if not 0 <= index < n:
            raise IndexError(f"index {index} outside {split} namespace of size {n}")
        return lo + index

    def take(self, split: str, count: int, offset: int = 0) -> list[int]:
        return [self.seed(split, offset + i) for i in range(count)]

    def disjoint(self) -> bool:
        seen: set[int] = set()
        for lo, hi in self.ranges.values():
            block = set(range(lo, hi + 1))
            if seen & block:
                return False
            seen |= block
        return True

    def overlaps(self) -> list[tuple[str, str]]:
        out = []
        keys = sorted(self.ranges)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                la, ha = self.ranges[a]
                lb, hb = self.ranges[b]
                if la <= hb and lb <= ha:
                    out.append((a, b))
        return out


def rng(*key: int | str) -> np.random.Generator:
    """Reproducible generator from a structured key.

    Strings are folded in by their bytes so that rng('E1', 3, 'delay') is
    stable across processes -- Python's hash() is salted and must not be used.
    """
    parts: list[int] = []
    for k in key:
        if isinstance(k, int):
            parts.append(k & 0xFFFFFFFF)
        else:
            acc = 0
            for b in k.encode():
                acc = (acc * 131 + b) & 0xFFFFFFFF
            parts.append(acc)
    return np.random.default_rng(np.random.SeedSequence(parts))
