"""
fig10_direction_err_timeseries.py
=================================

Figure fig10:direction_err_timeseries: in-sample direction error
against the direct Markowitz solution for CRISP (gamma=0.5) and A3
(gamma=0.5) over the backtest period, with the condition number
kappa(corr(Sigma_hat)) overlaid on a secondary y-axis.

Usage:
    python figures/code/fig10_direction_err_timeseries.py

Reads:
    results/sec10/direction_err.npz

Writes:
    figures/output/fig10_direction_err_timeseries.pdf
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "results", "sec10", "direction_err.npz")
OUT  = os.path.join(ROOT, "figures", "output",
                    "fig10_direction_err_timeseries.pdf")


def main():
    data = np.load(DATA, allow_pickle=True)
    rebal_months = data["rebal_months"]
    kappa = data["kappa"]
    methods = list(data["methods"])
    de = data["dir_err"]
    years = 1995.0 + rebal_months / 12.0

    fig, ax1 = plt.subplots(figsize=(6.8, 4.0))

    colors = {
        "B g=0.5 (100sw)": ("#1f77b4", r"CRISP $\gamma{=}0.5$"),
        "A3 g=0.5":        ("#2ca02c", r"HRP-$\mu$ $\gamma{=}0.5$"),
    }
    for i, name in enumerate(methods):
        color, label = colors.get(name, ("k", name))
        ax1.plot(years, de[i], color=color, lw=1.6, label=label)

    ax1.set_xlabel("Year")
    ax1.set_ylabel(r"In-sample direction error $1 - |\cos(\hat w, \hat\Sigma^{-1}\hat\mu)|$")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax2 = ax1.twinx()
    ax2.plot(years, kappa, color="#888888", lw=1.2, linestyle="--",
             alpha=0.85, label=r"$\kappa(\mathrm{corr}(\hat\Sigma))$")
    ax2.set_ylabel(r"$\kappa(\mathrm{corr}(\hat\Sigma))$ (dashed, grey)",
                   color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.set_yscale("log")

    ax1.set_title("In-sample direction error vs. Markowitz, composite "
                  "signal", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig10] wrote {OUT}")


if __name__ == "__main__":
    main()
