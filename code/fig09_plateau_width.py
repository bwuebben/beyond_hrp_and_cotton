"""
fig09_plateau_width.py
======================

Replacement for fig09_gamma_scatter: shows the width of the
gamma-insensitive Sharpe plateau for each calibration cell.

For each cell, the plateau is the range [gamma_lo, gamma_hi] where
the mean OOS Sharpe is within 2% of the peak Sharpe (relative).
The figure displays these as horizontal bars grouped by regime,
with a vertical line at gamma = 0.5 and the empirical argmax marked.

Usage:
    python figures/code/fig09_plateau_width.py

Reads:
    results/sec09_adaptive/exp1_calibration.csv
    (plus re-loads the L(gamma) curves from exp1 rows via the npz,
     or can use the CSV means column if available)

Writes:
    figures/output/fig09_plateau_width.pdf
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESDIR = os.path.join(ROOT, "results", "sec09_adaptive")
OUT = os.path.join(ROOT, "figures", "output", "fig09_plateau_width.pdf")

REGIME_STYLE = {
    "factor":   dict(color="#1f77b4", label="factor"),
    "block":    dict(color="#2ca02c", label="block"),
    "spiked":   dict(color="#d62728", label="spiked"),
    "equicorr": dict(color="#9467bd", label="equicorr"),
}

# Try to load plateau data from the npz saved by compute09.
# The npz stores all exp1 L(gamma) curves.
PLATEAU_NPZ = os.path.join(RESDIR, "exp1_plateau.npz")


def compute_plateau(gamma_grid, means, threshold=0.02):
    """Return (gamma_lo, gamma_hi, gamma_star) for the plateau where
    mean Sharpe is within `threshold` (relative) of the peak."""
    v = np.array(means, dtype=float)
    peak = np.nanmax(v)
    if peak <= 0 or not np.isfinite(peak):
        return 0.0, 1.0, 0.5
    cutoff = peak * (1.0 - threshold)
    inside = gamma_grid[v >= cutoff]
    if len(inside) == 0:
        return 0.0, 1.0, 0.5
    gamma_star = gamma_grid[np.nanargmax(v)]
    return float(inside[0]), float(inside[-1]), float(gamma_star)


def main():
    # Load the exp1 calibration CSV.  The CSV doesn't store the full
    # L(gamma) curve, so we need the npz.  If the npz isn't available,
    # try loading the lgamma_curves.npz (which only has 3 cells).
    # As a fallback, re-derive from the validation CSV means column.

    # Try loading all exp1 curves from an npz saved by an updated compute09.
    plateau_path = os.path.join(RESDIR, "exp1_curves.npz")
    if os.path.exists(plateau_path):
        d = np.load(plateau_path, allow_pickle=True)
        gamma_grid = d["gamma_grid"]
        rows = []
        for key in d["cell_keys"]:
            key_str = str(key)
            means = d[f"{key_str}__means"]
            regime = str(d[f"{key_str}__regime"])
            tn = float(d[f"{key_str}__tn"])
            ic = float(d[f"{key_str}__ic"])
            lo, hi, gs = compute_plateau(gamma_grid, means)
            rows.append(dict(regime=regime, tn=tn, ic=ic,
                             gamma_lo=lo, gamma_hi=hi, gamma_star=gs))
    else:
        # Fallback: re-run a lightweight version to get the curves.
        # This requires numpy and the covariance builders.
        print(f"[fig09plateau] {plateau_path} not found, "
              "computing plateau widths from scratch (200 MC reps)...")
        rows = _compute_fresh()

    if not rows:
        print("[fig09plateau] no data, aborting")
        return

    # Sort: group by regime, then by (T/N, IC) within regime.
    regime_order = ["factor", "block", "spiked", "equicorr"]
    rows.sort(key=lambda r: (regime_order.index(r["regime"]),
                              r["tn"], r["ic"]))

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    y_labels = []
    y_pos = []
    y = 0
    prev_regime = None
    for r in rows:
        if r["regime"] != prev_regime:
            if prev_regime is not None:
                y += 0.5  # gap between regimes
            prev_regime = r["regime"]
        style = REGIME_STYLE[r["regime"]]
        lo, hi, gs = r["gamma_lo"], r["gamma_hi"], r["gamma_star"]
        ax.barh(y, hi - lo, left=lo, height=0.7,
                color=style["color"], alpha=0.5, edgecolor=style["color"],
                linewidth=0.8)
        ax.plot(gs, y, "k|", markersize=8, markeredgewidth=1.5)
        y_labels.append(f"{r['regime'][:3]} T/N={r['tn']:.1f} IC={r['ic']:.2f}")
        y_pos.append(y)
        y += 1

    ax.axvline(0.5, color="black", ls="--", lw=1.2, alpha=0.7,
               label=r"$\gamma = 0.5$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=5.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_title(r"Width of the $\gamma$-insensitive Sharpe plateau "
                 r"(within 2\% of peak)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Add regime labels as a legend.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=s["color"], alpha=0.5, label=s["label"])
               for s in REGIME_STYLE.values()]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", lw=1.2,
                              label=r"$\gamma = 0.5$"))
    handles.append(plt.Line2D([0], [0], color="black", marker="|",
                              linestyle="None", markersize=8,
                              label=r"$\gamma^{\star}_{\mathrm{emp}}$"))
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig09plateau] wrote {OUT}")


def _compute_fresh():
    """Lightweight compute of plateau widths for 48 calibration cells."""
    import sys
    code_dir = os.path.join(ROOT, "figures", "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from compute09_adaptive_gamma import (
        regime_factory, kappa_corr, run_cell, smooth_argmax,
        REGIMES as REGIME_LIST,
    )

    N = 100
    seed = 42
    n_mc = 200
    gamma_grid = np.linspace(0.0, 1.0, 51)
    TN_grid = [0.6, 1.0, 2.0, 5.0]
    IC_grid = [0.02, 0.05, 0.10]

    rows = []
    for r_idx, regime in enumerate(REGIME_LIST):
        cov = regime_factory(regime)(N, seed)
        rng_mu = np.random.RandomState(seed + 1000 + r_idx)
        mu_true = rng_mu.randn(N) * 0.02
        for tn_idx, tn in enumerate(TN_grid):
            T = max(int(round(tn * N)), 10)
            for ic_idx, ic in enumerate(IC_grid):
                cell_seed = seed + r_idx * 10000 + tn_idx * 100 + ic_idx
                means = run_cell(cov, mu_true, T, ic, gamma_grid,
                                 n_mc=n_mc, ridge=1e-4, seed=cell_seed)
                lo, hi, gs = compute_plateau(gamma_grid, means)
                rows.append(dict(regime=regime, tn=tn, ic=ic,
                                 gamma_lo=lo, gamma_hi=hi, gamma_star=gs))
                print(f"  {regime:8s} T/N={tn:.1f} IC={ic:.2f} "
                      f"plateau=[{lo:.2f}, {hi:.2f}] "
                      f"gamma*={gs:.2f}")
    return rows


if __name__ == "__main__":
    main()
