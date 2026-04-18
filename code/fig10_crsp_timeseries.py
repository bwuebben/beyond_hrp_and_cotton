"""
fig10_crsp_timeseries.py
========================

Figure fig10:crsp_timeseries: cumulative equity curves for the main
allocation methods on the composite (momentum + value) signal over the
simulated Russell-1000-analog backtest produced by
figures/code/compute10_crsp_backtest.py.

Usage:
    python figures/code/fig10_crsp_timeseries.py

Reads:
    results/sec10/equity_curves_named.npz

Writes:
    figures/output/fig10_crsp_timeseries.pdf
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "results", "sec10", "equity_curves_named.npz")
OUT  = os.path.join(ROOT, "figures", "output", "fig10_crsp_timeseries.pdf")


STYLE = {
    "B g=0.5 (100sw)": dict(color="#1f77b4", lw=2.4,
                             label=r"CRISP $\gamma{=}0.5$"),
    "A3 g=0.5":        dict(color="#2ca02c", lw=2.0,
                             label=r"HRP-$\mu$ $\gamma{=}0.5$"),
    "LW-Markowitz":    dict(color="#9467bd", lw=1.8, label="Ledoit-Wolf Markowitz"),
    "Direct":          dict(color="#d62728", lw=1.6, label="Direct Markowitz"),
    "HRP":             dict(color="#7f7f7f", lw=1.5, label="HRP"),
    "Cotton g=0.7":    dict(color="#ff7f0e", lw=1.4, linestyle="--",
                             label=r"Cotton $\gamma{=}0.7$"),
    "1/N":             dict(color="#17becf", lw=1.3, linestyle=":",
                             label=r"$1/N$"),
}


def main():
    data = np.load(DATA, allow_pickle=True)
    rebal_months = data["rebal_months"]
    names = list(data["method_names"])
    curves = data["curves"]

    # Convert month-index to approximate calendar year.
    # Month 0 of R corresponds to Jan 1995; rebalance at month t means
    # OOS return on month t.
    years = 1995.0 + rebal_months / 12.0

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, name in enumerate(names):
        if name not in STYLE:
            continue
        style = STYLE[name]
        ax.plot(years, curves[i], **style)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative growth of \\$1")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title("Simulated Russell-1000-analog backtest, composite signal "
                 "(long-only, 10 bps/side)",
                 fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig10] wrote {OUT}")


if __name__ == "__main__":
    main()
