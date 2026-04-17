"""
fig10_gamma_t_series.py
=======================

Figure fig10:gamma_t_series: time series of the adaptive gamma*_t on
the simulated CRSP-analog panel, with kappa(hat_C_t) overlaid on a
secondary axis.

Usage:
    python figures/code/fig10_gamma_t_series.py

Reads:
    results/sec10/adaptive_gamma_ts.npz

Writes:
    figures/output/fig10_gamma_t_series.pdf
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
OUT = os.path.join(ROOT, "figures", "output", "fig10_gamma_t_series.pdf")


def main():
    d = np.load(DATA, allow_pickle=True)
    rebal = np.asarray(d["rebal_months"])
    years = 1995.0 + rebal / 12.0
    g = np.asarray(d["gamma_star_ts"])
    kC = np.asarray(d["kappa_ts"])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ln1 = ax1.plot(years, g, color="#1f77b4", lw=1.8,
                   label=r"Adaptive $\gamma^{\star}_t$")
    ax1.set_xlabel("Year")
    ax1.set_ylabel(r"$\gamma^{\star}_t$", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.5, color="#1f77b4", ls="--", lw=1.0, alpha=0.6,
                label=r"fixed $\gamma = 0.5$")

    ax2 = ax1.twinx()
    ln2 = ax2.plot(years, kC, color="#d62728", lw=1.3, alpha=0.75,
                   label=r"$\kappa(\widehat C_t)$")
    ax2.set_ylabel(r"$\kappa(\widehat C_t)$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_yscale("log")

    # Shade canonical crisis windows to anchor interpretation.
    for (a, b, label) in [
        (2000.25, 2002.5, "2000--02"),
        (2007.75, 2009.5, "2008"),
        (2020.0, 2020.5, "2020"),
    ]:
        ax1.axvspan(a, b, color="gray", alpha=0.12)

    # Merge legends.
    lines = ln1 + ln2
    labs = [ln.get_label() for ln in lines]
    ax1.legend(lines + [ax1.get_children()[3]],
               labs + [r"fixed $\gamma = 0.5$"],
               loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_title(r"Adaptive $\gamma^{\star}_t$ on the simulated "
                  "Russell-1000-analog panel (trailing 12-mo IC)",
                  fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig10gamma_t] wrote {OUT}")


if __name__ == "__main__":
    main()
