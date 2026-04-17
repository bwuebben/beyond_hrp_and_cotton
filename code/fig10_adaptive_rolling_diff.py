"""
fig10_adaptive_rolling_diff.py
==============================

Figure fig10:adaptive_rolling_diff: rolling 12-month mean return
difference between adaptive-gamma Method B and fixed gamma=0.5 Method
B on the simulated CRSP-analog panel. Positive = adaptive outperforms
fixed.

Usage:
    python figures/code/fig10_adaptive_rolling_diff.py

Reads:
    results/sec10/adaptive_gamma_ts.npz

Writes:
    figures/output/fig10_adaptive_rolling_diff.pdf
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "results", "sec10", "adaptive_gamma_ts.npz")
OUT = os.path.join(ROOT, "figures", "output",
                    "fig10_adaptive_rolling_diff.pdf")


def main():
    d = np.load(DATA, allow_pickle=True)
    rebal = np.asarray(d["rebal_months"])
    years = 1995.0 + rebal / 12.0
    diffs = np.asarray(d["ret_adaptive"]) - np.asarray(d["ret_fixed_05"])
    roll = 12
    kernel = np.ones(roll) / roll
    rolling_mean = np.convolve(diffs, kernel, mode="same")
    cum_diff = np.cumsum(diffs)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    ax1.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    ax1.plot(years, rolling_mean, color="#2ca02c", lw=1.7,
             label="12-mo rolling mean return diff")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Adaptive $-$ fixed $\\gamma=0.5$ monthly return")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(years, cum_diff, color="#1f77b4", lw=1.4, alpha=0.85,
             label="Cumulative return diff")
    ax2.set_ylabel("Cumulative return diff", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left",
               fontsize=8, framealpha=0.9)
    ax1.set_title("Adaptive vs fixed $\\gamma=0.5$ rolling return "
                  "difference (composite signal)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig10diff] wrote {OUT}")


if __name__ == "__main__":
    main()
