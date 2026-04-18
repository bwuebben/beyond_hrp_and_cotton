"""
fig09_sweep_heatmap.py
======================

Secondary figure for the three-channels §9 sweep-regularization
experiment. Contour/heatmap of mean OOS Sharpe over the (gamma, p)
plane, one panel per covariance regime at T/N = 1.0 with noisy mu.

Reads results/sec09_sweep/sweep_grid.npz.
Writes figures/output/fig09_sweep_heatmap.pdf.
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
OUT = os.path.join(ROOT, "figures", "output", "fig09_sweep_heatmap.pdf")


def main():
    d = np.load(DATA, allow_pickle=True)
    sharpe = d["sharpe"]
    regime_names = list(d["regime_names"])
    tn_grid = list(d["tn_grid"])
    sig_kinds = list(d["sig_kinds"])
    gamma_grid = d["gamma_grid"]
    p_grid = d["p_grid"]

    ti = tn_grid.index(1.0)
    si = sig_kinds.index("noisy")

    n_reg = len(regime_names)
    fig, axes = plt.subplots(1, n_reg, figsize=(3.3 * n_reg, 3.5),
                             sharey=True)
    if n_reg == 1:
        axes = [axes]

    # Shared color scale.
    panels = [sharpe[ri, ti, si] for ri in range(n_reg)]
    vmin = float(min(np.nanmin(z) for z in panels))
    vmax = float(max(np.nanmax(z) for z in panels))

    for ri, reg in enumerate(regime_names):
        ax = axes[ri]
        Z = sharpe[ri, ti, si]  # (n_gamma, n_p)
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=[np.log10(p_grid[0]), np.log10(p_grid[-1]),
                               float(gamma_grid[0]), float(gamma_grid[-1])],
                       vmin=vmin, vmax=vmax, cmap="viridis")
        # Mark the argmax.
        k = int(np.nanargmax(Z))
        gi_star, pi_star = np.unravel_index(k, Z.shape)
        ax.plot([np.log10(p_grid[pi_star])],
                [gamma_grid[gi_star]],
                marker="*", color="white", ms=10, mec="black", mew=0.7)
        ax.set_title(reg, fontsize=10)
        ax.set_xlabel(r"$\log_{10} p$")
        if ri == 0:
            ax.set_ylabel(r"$\gamma$")

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Mean OOS Sharpe")
    fig.suptitle(r"$(\gamma, p)$ OOS Sharpe surface "
                 r"(noisy $\mu$, $T/N = 1.0$)",
                 fontsize=11, y=1.05)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig09heatmap] wrote {OUT}")


if __name__ == "__main__":
    main()
