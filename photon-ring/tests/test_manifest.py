import json

import pytest

from phrt.config import load_registry
from phrt.io.manifests import (Gate, RunManifest, gate_from_tolerance,
                               make_run_id, validate_manifest)


def test_manifest_validates_and_carries_deviations():
    reg = load_registry()
    m = RunManifest(run_id=make_run_id("T", reg.sha256), experiment_id="T", seeds={"s": 1})
    m.add_input(reg.path)
    m.add_gate(gate_from_tolerance("G_x", 1e-15, 1e-10))
    doc = m.build(reg.path, reg.sha256, 0.1)
    validate_manifest(doc)
    assert doc["config_sha256"] == reg.sha256
    assert isinstance(doc["protocol_deviations"], list)


def test_missing_field_is_rejected():
    with pytest.raises(ValueError):
        validate_manifest({"run_id": "x"})


def test_nan_never_passes_a_tolerance_gate():
    assert gate_from_tolerance("g", float("nan"), 1.0).status == "FAIL"


def test_bad_gate_status_is_rejected():
    reg = load_registry()
    m = RunManifest(run_id="r", experiment_id="T")
    m.add_gate(Gate("g", "MAYBE"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        m.build(reg.path, reg.sha256)
