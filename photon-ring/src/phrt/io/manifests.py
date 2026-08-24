"""Run manifests and the machine-readable correctness-gate file.

Every manifest validates against schemas/run_manifest_schema_v0.2.json before
it is written.  A manifest that does not validate is a bug, not a warning.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from phrt.config import repo_root, sha256_file
from phrt import provenance

GateStatus = Literal["PASS", "FAIL", "NOT_RUN"]
REQUIRED_TOP = (
    "run_id", "experiment_id", "git_commit", "config_path", "config_sha256",
    "environment_sha256", "hardware", "seeds", "inputs", "outputs", "gate_status",
)


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def peak_rss_mb() -> float:
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return kb / 1024.0 if os.uname().sysname == "Linux" else kb / (1024.0 * 1024.0)
    except Exception:
        return -1.0


@dataclass
class Gate:
    name: str
    status: GateStatus
    measured: Any = None
    threshold: Any = None
    evidence_path: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"status": self.status}
        if self.measured is not None:
            d["measured"] = self.measured
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.evidence_path:
            d["evidence_path"] = self.evidence_path
        if self.note:
            d["note"] = self.note
        return d


def gate_from_tolerance(name: str, measured: float, threshold: float,
                        evidence_path: str | None = None,
                        note: str | None = None) -> Gate:
    ok = bool(measured <= threshold) and measured == measured  # NaN never passes
    return Gate(name, "PASS" if ok else "FAIL", float(measured), float(threshold),
                evidence_path, note)


@dataclass
class RunManifest:
    run_id: str
    experiment_id: str
    seeds: dict[str, Any] = field(default_factory=dict)
    inputs: list[dict[str, str]] = field(default_factory=list)
    outputs: list[dict[str, str]] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    started_at: str = field(default_factory=utcnow)
    extra: dict[str, Any] = field(default_factory=dict)

    def add_input(self, path: str | Path) -> None:
        self.inputs.append({"path": str(path), "sha256": sha256_file(path)})

    def add_output(self, path: str | Path) -> None:
        self.outputs.append({"path": str(path), "sha256": sha256_file(path)})

    def add_gate(self, gate: Gate) -> Gate:
        self.gates.append(gate)
        return gate

    @property
    def failed_gates(self) -> list[str]:
        return [g.name for g in self.gates if g.status == "FAIL"]

    def build(self, config_path: str | Path, config_sha256: str,
              runtime_seconds: float | None = None) -> dict[str, Any]:
        prov = provenance.collect()
        doc: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "git_commit": prov.git_commit,
            "dirty_tree": prov.dirty_tree,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "environment_sha256": prov.environment_sha256,
            "hardware": prov.hardware,
            "seeds": self.seeds,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "gate_status": {g.name: g.to_dict() for g in self.gates},
            "started_at": self.started_at,
            "finished_at": utcnow(),
            "peak_rss_mb": peak_rss_mb(),
            "packages": prov.packages,
            "protocol_deviations": prov.deviations,
        }
        if runtime_seconds is not None:
            doc["runtime_seconds"] = float(runtime_seconds)
        doc.update(self.extra)
        validate_manifest(doc)
        return doc

    def write(self, config_path: str | Path, config_sha256: str,
              runtime_seconds: float | None = None,
              out_dir: str | Path | None = None) -> Path:
        doc = self.build(config_path, config_sha256, runtime_seconds)
        d = Path(out_dir) if out_dir else repo_root() / "artifacts" / "manifests"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.run_id}.json"
        p.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        return p


def validate_manifest(doc: dict[str, Any]) -> None:
    """Structural validation against the frozen schema.

    jsonschema is not a registered dependency, so the required-field and enum
    constraints that actually matter are checked directly.
    """
    missing = [k for k in REQUIRED_TOP if k not in doc]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    hw = doc["hardware"]
    for k in ("platform", "architecture", "memory_bytes"):
        if k not in hw:
            raise ValueError(f"manifest hardware missing {k!r}")
    if not isinstance(hw["memory_bytes"], int):
        raise ValueError("hardware.memory_bytes must be an integer")
    for key in ("inputs", "outputs"):
        for item in doc[key]:
            if "path" not in item or "sha256" not in item:
                raise ValueError(f"{key} entries need path and sha256")
    for name, g in doc["gate_status"].items():
        if g.get("status") not in ("PASS", "FAIL", "NOT_RUN"):
            raise ValueError(f"gate {name} has invalid status {g.get('status')!r}")


def make_run_id(experiment_id: str, config_sha256: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_id}_{stamp}_{config_sha256[:8]}"


def merge_gate_file(gates: Iterable[Gate], run_id: str,
                    path: str | Path | None = None) -> Path:
    """Accumulate gates into artifacts/gates/correctness_gates.json.

    Later runs overwrite earlier entries for the same gate name; the previous
    value is retained under 'superseded' so a gate that used to pass and now
    fails is visible rather than silently replaced.
    """
    p = Path(path) if path else repo_root() / "artifacts" / "gates" / "correctness_gates.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(p.read_text()) if p.exists() else {"gates": {}}
    for g in gates:
        entry = g.to_dict() | {"run_id": run_id, "recorded_at": utcnow()}
        prev = doc["gates"].get(g.name)
        if prev is not None:
            entry["superseded"] = {k: prev[k] for k in prev if k != "superseded"}
        doc["gates"][g.name] = entry
    doc["updated_at"] = utcnow()
    doc["summary"] = {
        "PASS": sum(1 for v in doc["gates"].values() if v["status"] == "PASS"),
        "FAIL": sum(1 for v in doc["gates"].values() if v["status"] == "FAIL"),
        "NOT_RUN": sum(1 for v in doc["gates"].values() if v["status"] == "NOT_RUN"),
    }
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return p
