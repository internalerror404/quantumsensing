from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # scripted figure generation; the macOS backend wants the main thread
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from light_ray import candidate_rays

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Figure 1: finite null-channel geometry schematic in a spatial projection.
rays = candidate_rays(direction_count=24, offsets_per_direction=2, offset_radius=0.35, lam_extent=1.5)
fig = plt.figure(figsize=(7.0, 5.2))
ax = fig.add_subplot(111, projection="3d")
for ray in rays[::3]:
    lam = np.linspace(ray.lam_min, ray.lam_max, 80)
    pts = ray.points(lam)
    ax.plot(pts[:, 1], pts[:, 2], pts[:, 3], linewidth=0.8, alpha=0.55)
# Compact world tube projected to a spatial cube.
r = [-0.72, 0.72]
for z in r:
    for y in r:
        ax.plot(r, [y, y], [z, z], linewidth=1.0)
for z in r:
    for x in r:
        ax.plot([x, x], r, [z, z], linewidth=1.0)
for y in r:
    for x in r:
        ax.plot([x, x], [y, y], r, linewidth=1.0)
ax.set_xlabel("$x^1$")
ax.set_ylabel("$x^2$")
ax.set_zlabel("$x^3$")
ax.set_title("Finite null-channel sampling of an interior metric perturbation")
ax.view_init(elev=22, azim=35)
fig.tight_layout()
fig.savefig(FIG / "ray_network.pdf", bbox_inches="tight")
fig.savefig(FIG / "ray_network.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Figure 2: one-instance QFIM eigenvalue spectrum (clearly labeled as sanity check).
report = json.loads((ROOT / "results" / "design_demo.json").read_text())
# Recompute spectra is unnecessary for the paper figure; plot the scalar design summaries.
names = [d["name"].replace("_", " ") for d in report["designs"]]
mins = [d["min_eigenvalue"] for d in report["designs"]]
conds = [d["condition_number"] for d in report["designs"]]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = np.arange(len(names))
ax.bar(x, mins)
ax.set_yscale("log")
ax.set_xticks(x, names, rotation=22, ha="right")
ax.set_ylabel("Minimum eigenvalue of whitened QFIM")
ax.set_title("Single synthetic task: design changes the worst visible direction")
ax.grid(axis="y", which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "design_sanity.pdf", bbox_inches="tight")
fig.savefig(FIG / "design_sanity.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Figure 3: ML experiment pipeline.
fig, ax = plt.subplots(figsize=(10.5, 3.6))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.6); ax.axis("off")
boxes = [
    (0.2, 1.15, 1.7, 1.25, "Metric-family task\n$V, R, $ masks"),
    (2.2, 1.15, 1.7, 1.25, "Candidate null rays\n$\\{\\gamma_i\\}_{i=1}^M$"),
    (4.2, 1.15, 1.8, 1.25, "Sensitivity + QFI\n$a_i, q_i=1/s_i^2$"),
    (6.35, 1.15, 1.8, 1.25, "DeepSets selector\n$K$ rays / allocation"),
    (8.45, 1.15, 1.8, 1.25, "MAP reconstruction\nrank, RMSE, coverage"),
]
for x0, y0, w, h, text in boxes:
    patch = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.03,rounding_size=0.05", fill=False, linewidth=1.4)
    ax.add_patch(patch)
    ax.text(x0+w/2, y0+h/2, text, ha="center", va="center", fontsize=10)
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    y = boxes[i][1] + boxes[i][3]/2
    ax.add_patch(FancyArrowPatch((x1+0.05, y), (x2-0.05, y), arrowstyle="->", mutation_scale=12, linewidth=1.2))
ax.text(5.25, 3.05, "Amortized quantum-statistical null-channel design", ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(5.25, 0.45, "Training objective: maximize $\\lambda_{\\min}$ or $\\log\\det$ of the whitened QFIM; evaluation keeps the inverse solver linear and transparent.", ha="center", va="center", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "ml_pipeline.pdf", bbox_inches="tight")
fig.savefig(FIG / "ml_pipeline.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(FIG)
