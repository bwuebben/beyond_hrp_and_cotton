"""
fig06_bias_variance_curves.py
=============================

Figure fig06:bias_variance_curves: OOS Sharpe shortfall L(gamma) for
three representative (T/N, IC, kappa(C)) parameter combinations, with
the adaptive-rule predicted gamma* marked. Consumes the lgamma_curves
artifact written by compute09_adaptive_gamma.py.

Usage:
    python figures/code/fig06_bias_variance_curves.py

Reads:
    results/sec09_adaptive/lgamma_curves.npz

Writes:
    figures/output/fig06_bias_variance_curves.pdf
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "results", "sec09_adaptive", "lgamma_curves.npz")
OUT = os.path.join(ROOT, "figures", "output", "fig06_bias_variance_curves.pdf")


def load_cells(path: str) -> list[dict]:
    npz = np.load(path, allow_pickle=True)
    # The writer produced keys of the form "<tag>__gamma", "<tag>__means", ...
    tags = set()
    for k in npz.files:
        tag, _, _ = k.rpartition("__")
        if tag:
            tags.add(tag)
    cells = []
    for tag in sorted(tags):
        cells.append(dict(
            tag=tag,
            gamma=np.asarray(npz[f"{tag}__gamma"]),
            means=np.asarray(npz[f"{tag}__means"]),
            kappa_C=float(npz[f"{tag}__kappa_C"]),
            tn=float(npz[f"{tag}__tn"]),
            ic=float(npz[f"{tag}__ic"]),
            gamma_pred=float(npz[f"{tag}__gamma_pred"]),
            regime=str(npz[f"{tag}__regime"]),
        ))
    # Sort ascending by NSR = kappa^2 * N/T / ic^2 so the three cells
    # come out in low -> mid -> high NSR order.
    def nsr(c):
        return (c["kappa_C"] ** 2) / (c["ic"] ** 2) / c["tn"]
    cells.sort(key=nsr)
    return cells


def loss_from_means(means: np.ndarray) -> np.ndarray:
    """Convert mean OOS Sharpe to relative Sharpe shortfall."""
    m = np.array(means, dtype=float)
    peak = np.nanmax(m)
    return peak - m


def main():
    cells = load_cells(DATA)
    if len(cells) < 3:
        raise RuntimeError(f"expected >=3 representative cells, got "
                           f"{len(cells)}")
    # Use the first (lowest-NSR), middle, and last (highest-NSR).
    lo, mid, hi = cells[0], cells[len(cells) // 2], cells[-1]
    chosen = [lo, mid, hi]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    labels = ["low NSR", "mid NSR", "high NSR"]
    for c, col, lab in zip(chosen, colors, labels):
        L = loss_from_means(c["means"])
        ax.plot(c["gamma"], L, color=col, lw=2.1,
                label=f"{lab}: $T/N={c['tn']:.1f}$, "
                      f"$\\mathrm{{IC}}={c['ic']:.2f}$, "
                      f"$\\kappa(C)={c['kappa_C']:.0f}$")
        # Mark predicted gamma*
        ax.axvline(c["gamma_pred"], color=col, ls="--", lw=1.1, alpha=0.85)
        ax.plot([c["gamma_pred"]], [np.interp(c["gamma_pred"],
                                              c["gamma"], L)],
                marker="o", color=col, ms=7, mec="black", mew=0.7)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Sharpe shortfall  $L(\gamma) = \max_\gamma \mathrm{SR}(\gamma) - \mathrm{SR}(\gamma)$")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title("Bias--variance curves $L(\\gamma)$ with predicted "
                 "$\\gamma^{\\star}$ from the adaptive rule",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig06bv] wrote {OUT}")


if __name__ == "__main__":
    main()
