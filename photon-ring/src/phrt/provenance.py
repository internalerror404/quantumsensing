"""Environment, hardware, and deviation capture.

The registered protocol targets one macOS machine with optional Apple-Silicon
MPS.  This session executes on Linux x86_64.  That is a real deviation from the
preregistration and it is recorded as such rather than papered over: nothing in
this module ever reports a platform it is not running on.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Packages whose versions are pinned into the environment hash.  Anything that
# can change a float64 result belongs here.
TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "pyarrow", "h5py", "matplotlib",
    "scikit-learn", "scikit-image", "pyyaml", "torch", "aart", "imageio",
)

REGISTERED_TARGET = {
    "execution_target": "macos_native",
    "accelerator": "apple_silicon_mps_optional",
    "physics_precision": "float64_cpu",
    "ml_precision": "float32_mps_or_cpu",
}


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "ABSENT"
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return "UNKNOWN"


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


# Top-level-relative pathspecs (":/" prefix) so the answer does not depend on
# which subdirectory git happens to be invoked from.
SOURCE_PATHSPECS = (":/photon-ring/src", ":/photon-ring/scripts",
                    ":/photon-ring/configs", ":/photon-ring/tests")


def git_dirty(paths: tuple[str, ...] = SOURCE_PATHSPECS) -> bool:
    """Dirty status scoped to *source*, not to generated artifacts.

    Hashing the whole tree makes every emitter after the first report a dirty
    run purely because an earlier emitter wrote its own output.  Untracked
    source files count as dirty; untracked artifacts do not.
    """
    if _git("rev-parse", "--show-toplevel") == "UNKNOWN":
        return True
    return bool(_git("status", "--porcelain", "--", *paths).strip())


def torch_device_report() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch": "ABSENT"}
    mps = getattr(torch.backends, "mps", None)
    return {
        "torch": torch.__version__,
        "mps_built": bool(mps and mps.is_built()),
        "mps_available": bool(mps and mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": select_device_name(),
    }


def select_device_name() -> str:
    """Registered policy is MPS-or-CPU.  CUDA is *not* silently substituted for
    MPS in scientific paths; it is only reported."""
    try:
        import torch
    except Exception:
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if (mps and mps.is_available()) else "cpu"


def hardware() -> dict[str, Any]:
    try:
        import psutil

        mem = int(psutil.virtual_memory().total)
    except Exception:
        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            mem = -1
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    mps = torch_device_report().get("mps_available", False)
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "memory_bytes": mem,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "mps_available": mps,
    }


def deviations() -> list[dict[str, str]]:
    """Registered-versus-actual differences.  Empty list means full compliance."""
    out: list[dict[str, str]] = []
    hw = hardware()
    if hw["platform"] != "Darwin":
        out.append({
            "id": "D1_platform",
            "registered": "macos_native (execution_target in registry)",
            "actual": f"{hw['platform']} {hw['architecture']}",
            "effect": (
                "No macOS-specific or Apple-Silicon-specific result can be claimed. "
                "All float64 CPU numerics are platform-portable and unaffected; "
                "runtime and peak-RSS rows describe this Linux host, not a Mac."
            ),
        })
    if not hw["mps_available"]:
        out.append({
            "id": "D2_no_mps",
            "registered": "optional Apple-Silicon MPS for compact neural models",
            "actual": "no MPS device present",
            "effect": (
                "Gate G11 (CPU/MPS inference parity) cannot be executed and is "
                "recorded NOT_RUN. Neural training runs on CPU float32. CUDA is "
                "not substituted for MPS in any registered gate."
            ),
        })
    return out


@dataclass
class Provenance:
    git_commit: str
    dirty_tree: bool
    python: str
    packages: dict[str, str]
    hardware: dict[str, Any]
    torch: dict[str, Any]
    registered_target: dict[str, str] = field(default_factory=lambda: dict(REGISTERED_TARGET))
    deviations: list[dict[str, str]] = field(default_factory=list)

    @property
    def environment_sha256(self) -> str:
        payload = json.dumps(
            {"python": self.python, "packages": self.packages,
             "architecture": self.hardware["architecture"],
             "platform": self.hardware["platform"]},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["environment_sha256"] = self.environment_sha256
        return d


def collect() -> Provenance:
    return Provenance(
        git_commit=git_commit(),
        dirty_tree=git_dirty(),
        python=sys.version.split()[0],
        packages=_package_versions(),
        hardware=hardware(),
        torch=torch_device_report(),
        deviations=deviations(),
    )
