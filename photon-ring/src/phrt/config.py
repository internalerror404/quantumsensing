"""Registry loading and config hashing.

The frozen registry is the scientific contract.  Its SHA-256 goes into every
manifest, so it is read as *bytes* and hashed before parsing -- a YAML
round-trip would silently normalise formatting and change the hash.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REGISTRY_RELATIVE = "configs/paper1_experiment_registry_v0.2.yaml"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Registry:
    path: Path
    sha256: str
    data: dict[str, Any]

    def profile(self, name: str) -> dict[str, Any]:
        profiles = self.data["profiles"]
        if name not in profiles:
            raise KeyError(f"unregistered profile {name!r}; registered: {sorted(profiles)}")
        return profiles[name]

    def geometry_grid(self) -> list[tuple[float, float]]:
        g = self.data["geometry_grid"]
        return [(float(a), float(i)) for a in g["spin"] for i in g["inclination_deg"]]

    def orders(self) -> list[int]:
        return [int(n) for n in self.data["geometry_grid"]["orders"]]

    def gate_threshold(self, key: str) -> Any:
        return self.data["correctness_gates"][key]


def repo_root(start: str | Path | None = None) -> Path:
    here = Path(start or __file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / REGISTRY_RELATIVE).exists():
            return cand
    raise FileNotFoundError("could not locate photon-ring repo root (no configs/ registry above)")


def load_registry(path: str | Path | None = None) -> Registry:
    p = Path(path) if path is not None else repo_root() / REGISTRY_RELATIVE
    raw = p.read_bytes()
    return Registry(path=p, sha256=sha256_bytes(raw), data=yaml.safe_load(raw))


def geometry_id(spin: float, inclination_deg: float) -> str:
    """Canonical geometry key, e.g. a090_i050 for spin 0.9, inclination 50 deg."""
    return f"a{int(round(spin * 100)):03d}_i{int(round(inclination_deg)):03d}"
