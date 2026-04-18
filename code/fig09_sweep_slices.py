"""
fig09_sweep_slices.py
=====================

Primary figure for the three-channels §9 sweep-regularization
experiment. Plots OOS Sharpe vs sweep count p at fixed gamma slices
gamma in {0.3, 0.5, 0.7, 1.0}, one panel per covariance regime at
T/N = 1.0 with noisy mu.

Reads results/sec09_sweep/sweep_grid.npz.
Writes figures/output/fig09_sweep_slices.pdf.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "results", "sec09_sweep", "sweep_grid.npz")
OUT = os.path.join(ROOT, "figures", "output", "fig09_sweep_slices.pdf")

SLICE_GAMMAS = [0.3, 0.5, 0.7, 1.0]
COLORS = {0.3: "#2ca02c", 0.5: "#1f77b4", 0.7: "#ff7f0e", 1.0: "#d62728"}


def main():
    d = np.load(DATA, allow_pickle=True)
    sharpe = d["sharpe"]               # (reg, tn, sig, gamma, p)
    regime_names = list(d["regime_names"])
    tn_grid = list(d["tn_grid"])
    sig_kinds = list(d["sig_kinds"])
    gamma_grid = d["gamma_grid"]
    p_grid = d["p_grid"]

    ti = tn_grid.index(1.0)
    si = sig_kinds.index("noisy")

    n_reg = len(regime_names)
    fig, axes = plt.subplots(1, n_reg, figsize=(3.2 * n_reg, 3.6),
                             sharey=False)
    if n_reg == 1:
        axes = [axes]

    for ri, reg in enumerate(regime_names):
        ax = axes[ri]
        for g in SLICE_GAMMAS:
            gi = int(np.argmin(np.abs(gamma_grid - g)))
            y = sharpe[ri, ti, si, gi]
            ax.plot(p_grid, y, marker="o", ms=4.5, lw=1.6,
                    color=COLORS[g], label=rf"$\gamma={g:.1f}$")
        ax.set_xscale("log")
        ax.set_xlabel("Sweep count $p$ (log scale)")
        if ri == 0:
            ax.set_ylabel("Mean OOS Sharpe")
        ax.set_title(reg)
        ax.grid(True, alpha=0.3)
        if ri == n_reg - 1:
            ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle(r"OOS Sharpe vs sweep count $p$ at four $\gamma$ slices "
                 r"(noisy $\mu$, $T/N = 1.0$)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig09sweep] wrote {OUT}")


if __name__ == "__main__":
    main()
