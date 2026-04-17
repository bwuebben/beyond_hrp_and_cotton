"""
fig09_gamma_scatter.py
======================

Figure fig09:gamma_scatter: predicted vs empirical gamma* across the
108-cell expanded validation grid of Experiment 2, colored by regime.

Usage:
    python figures/code/fig09_gamma_scatter.py

Reads:
    results/sec09_adaptive/exp2_validation.csv

Writes:
    figures/output/fig09_gamma_scatter.pdf
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "results", "sec09_adaptive", "exp2_validation.csv")
OUT = os.path.join(ROOT, "figures", "output", "fig09_gamma_scatter.pdf")


REGIME_STYLE = {
    "factor":   dict(color="#1f77b4", marker="o", label="factor"),
    "block":    dict(color="#2ca02c", marker="s", label="block"),
    "spiked":   dict(color="#d62728", marker="^", label="spiked"),
    "equicorr": dict(color="#9467bd", marker="D", label="equicorr"),
}


def main():
    rows = list(csv.DictReader(open(DATA)))
    by_regime = {k: {"emp": [], "pred": []} for k in REGIME_STYLE}
    for r in rows:
        reg = r["regime"]
        if reg not in by_regime:
            continue
        by_regime[reg]["emp"].append(float(r["gamma_emp"]))
        by_regime[reg]["pred"].append(float(r["gamma_pred"]))

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for reg, style in REGIME_STYLE.items():
        bucket = by_regime[reg]
        if not bucket["emp"]:
            continue
        ax.scatter(bucket["pred"], bucket["emp"],
                   c=style["color"], marker=style["marker"], s=48,
                   edgecolors="black", linewidths=0.5,
                   label=style["label"], alpha=0.85)

    ax.plot([0, 1], [0, 1], color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Predicted $\gamma^{\star}$ (adaptive rule)")
    ax.set_ylabel(r"Empirical $\gamma^{\star}$ (MC argmax of OOS Sharpe)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Report R^2 and MAE on the plot.
    all_emp = np.array([float(r["gamma_emp"]) for r in rows])
    all_pred = np.array([float(r["gamma_pred"]) for r in rows])
    mae = float(np.mean(np.abs(all_emp - all_pred)))
    if np.var(all_emp) > 0:
        r2 = 1.0 - np.sum((all_emp - all_pred) ** 2) / np.sum(
            (all_emp - all_emp.mean()) ** 2
        )
    else:
        r2 = float("nan")
    ax.text(0.02, 0.98,
            f"$R^2 = {r2:.3f}$\nMAE $= {mae:.3f}$\n$n = {len(rows)}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="black",
                      alpha=0.85))
    ax.set_title(r"Adaptive $\gamma^{\star}$: predicted vs empirical "
                 "(108 synthetic cells)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig09scatter] wrote {OUT}")


if __name__ == "__main__":
    main()
