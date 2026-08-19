"""Three-panel central figure for Experiment D+R-1.

Panel A: whitened QFIM spectra for arms A, D, C -- four numerical-zero
directions in the delay-only arms, a fully positive spectrum with clocks.
Panel B: conformal nullity against the resource added.
Panel C: conformal posterior std against relative clock precision rho.

Reads results/joint_delay_redshift.json; writes paper/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Categorical slots in fixed order (validated: CVD deltaE 9.2, normal 27.6).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a1a", "#6b6b6b"
D_MAIN = "12"
ZERO_FLOOR = 1e-20  # display floor; true values are < 1e-30

report = json.loads((ROOT / "results" / "joint_delay_redshift.json").read_text())
block = report["per_dp"][D_MAIN]
arms = block["arms"]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
plt.rcParams["font.size"] = 10

# --- Panel A: visibility spectrum -------------------------------------------
ax = axes[0]
# Arms A and D have near-identical spectra (that IS the negative control), so
# a small horizontal dodge keeps both visible instead of overplotting.
series = [
    ("A_delay_only", "delay only", BLUE, "o", -0.14),
    ("D_plus_4_more_rays", "+4 delay rays", ORANGE, "s", 0.14),
    ("C_plus_4_links", "+4 clock links", AQUA, "D", 0.0),
]
for key, label, color, marker, dodge in series:
    vals = np.asarray(arms[key]["whitened_eigenvalues"], dtype=float)
    clipped = np.maximum(vals, ZERO_FLOOR)
    idx = np.arange(1, len(vals) + 1) + dodge
    ax.plot(idx, clipped, marker=marker, ms=5, lw=1.4, color=color,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8,
            label=label, clip_on=False)
ax.axhspan(ZERO_FLOOR / 30, ZERO_FLOOR * 3, color="#eeeeee", zorder=0)
ax.text(1.0, ZERO_FLOOR * 5, "numerical zero (< 1e-30), clipped for display",
        fontsize=7.5, color=MUTED)
ax.set_yscale("log")
ax.set_ylim(ZERO_FLOOR / 30, 2e1)
ax.set_xlabel("eigenvalue index (whitened QFIM)")
ax.set_ylabel("eigenvalue")
ax.set_title("A  Visibility spectrum ($d_p=12$)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8.5, loc="center right")
ax.grid(axis="y", alpha=0.2, lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# --- Panel B: rank restoration ----------------------------------------------
ax = axes[1]
resources = ["none", "+4 delay\nrays", "+3 clock\nlinks", "+4 clock\nlinks", "+10 clock\nlinks"]
nullity = [arms["A_delay_only"]["nullity"], arms["D_plus_4_more_rays"]["nullity"],
           arms["B_plus_3_links"]["nullity"], arms["C_plus_4_links"]["nullity"],
           arms["E_plus_all_links"]["nullity"]]
colors = [BLUE, ORANGE, AQUA, AQUA, AQUA]
x = np.arange(len(resources))
bars = ax.bar(x, nullity, width=0.62, color=colors, edgecolor="white", linewidth=1.5)
for xi, ni in zip(x, nullity):
    ax.text(xi, ni + 0.12, str(ni), ha="center", fontsize=9.5, color=INK)
ax.set_xticks(x, resources, fontsize=8.2)
ax.set_ylabel("conformal nullity")
ax.set_ylim(0, 4.8)
ax.set_yticks(range(5))
ax.set_title("B  Rank restoration (all $d_p$)", loc="left", fontsize=10)
ax.grid(axis="y", alpha=0.2, lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# --- Panel C: precision conversion ------------------------------------------
ax = axes[2]
for key, label, color, ls, ms_mark in (("C_plus_4_links", "4 selected links", AQUA, "-", 6),
                                       ("E_plus_all_links", "all 10 links", BLUE, "--", 4)):
    sweep = arms[key]["clock_precision_sweep"]
    rho = np.asarray(sweep["rho"]); u = np.asarray(sweep["u_conf"])
    ax.plot(rho, u, lw=1.6, ls=ls, color=color, label=label)
    for mark in sweep["marked_points_rho"]:
        u_mark = float(np.interp(np.log(mark), np.log(rho), u))
        ax.plot(mark, u_mark, "o", ms=ms_mark, color=color,
                markerfacecolor="white", markeredgewidth=1.2)
        if key == "C_plus_4_links":
            ax.annotate(r"$\rho=%g$" % mark, (mark, u_mark),
                        textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=MUTED)
# The high-rho floor is a physical-nuisance floor, not residual non-identifiability:
# finite delay precision in theta propagates through R_p into the inferred alpha.
ax.annotate("floor = physical-nuisance\nleakage through $R_p$,\nnot a surviving kernel",
            xy=(60, 0.315), xytext=(5.5, 0.09), fontsize=7.5, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
ax.text(0.11, 1.015, "prior-limited", fontsize=7.5, color=MUTED)
ax.set_xscale("log")
ax.set_xlabel(r"relative clock precision  $\rho=\sigma_D/\sigma_R$")
ax.set_ylabel("mean conformal posterior std (whitened)")
ax.set_ylim(0, 1.1)
ax.set_title("C  Precision conversion ($d_p=12$)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.grid(axis="y", alpha=0.2, lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.suptitle("Clock-assisted conformal rank restoration: a distinct observable\u2014not additional delay samples\u2014lifts the kernel",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "joint_rank_restoration.pdf", bbox_inches="tight")
fig.savefig(FIG / "joint_rank_restoration.png", dpi=220, bbox_inches="tight")
print(FIG / "joint_rank_restoration.png")
