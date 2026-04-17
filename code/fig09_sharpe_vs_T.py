"""
fig09_sharpe_vs_T.py

Line plot of mean OOS Sharpe as a function of sample size T, for each method
on the structural sector-tilt signal (oracle mu), read from
results/02_walkforward_sensitivity_and_structural.txt.

Referenced in Section 9 of the Cotton-Schur paper as fig09:sharpe_vs_T.

Usage:
    python figures/code/fig09_sharpe_vs_T.py
Writes:
    figures/output/fig09_sharpe_vs_T.pdf
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "figures", "output", "fig09_sharpe_vs_T.pdf")

# Hardcoded numbers from results/02_walkforward_sensitivity_and_structural.txt
# (EXPERIMENT 2, structural mu, oracle mu estimator).  T=500 is not in
# results/02 so we omit it for this panel; the table retains the sensitivity
# sweep which does include T=500 in experiment 1.
T_values = [60, 120, 240]

mean_sharpe = {
    "CRISP $\\gamma{=}0.7$": [0.497, 0.540, 0.571],
    "CRISP $\\gamma{=}0.5$": [0.485, 0.516, 0.536],
    r"HRP-$\Sigma\mu$ $\gamma{=}1.0$": [0.503, 0.515, 0.517],
    r"HRP-$\Sigma\mu$ $\gamma{=}0.5$": [0.424, 0.433, 0.435],
    r"HRP-$\mu$ $\gamma{=}0.5$":       [0.417, 0.427, 0.428],
    "Direct Markowitz":          [0.376, 0.294, 0.497],
    "HRP":                       [-0.001, 0.001, 0.001],
    "$1/N$":                     [0.000, 0.000, 0.000],
}

ORACLE = 0.645  # from EXPERIMENT 2 header in results/02

styles = {
    "CRISP $\\gamma{=}0.7$": dict(color="#1f77b4", marker="o", lw=2.2),
    "CRISP $\\gamma{=}0.5$": dict(color="#17becf", marker="s", lw=2.0,
                                     linestyle="--"),
    r"HRP-$\Sigma\mu$ $\gamma{=}1.0$": dict(color="#ff7f0e", marker="P",
                                             lw=2.2),
    r"HRP-$\Sigma\mu$ $\gamma{=}0.5$": dict(color="#e377c2", marker="p",
                                             lw=1.8, linestyle="--"),
    r"HRP-$\mu$ $\gamma{=}0.5$":       dict(color="#2ca02c", marker="^", lw=2.0),
    "Direct Markowitz":          dict(color="#d62728", marker="D", lw=1.8),
    "HRP":                       dict(color="#7f7f7f", marker="v", lw=1.4,
                                     linestyle=":"),
    "$1/N$":                     dict(color="#bcbd22", marker="x", lw=1.4,
                                     linestyle=":"),
}


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for label, series in mean_sharpe.items():
        ax.plot(T_values, series, label=label, **styles[label])

    ax.axhline(ORACLE, color="black", linestyle="-.", lw=1.0,
               label=f"Oracle Sharpe = {ORACLE:.3f}")

    ax.set_xlabel("Sample size $T$ (months)")
    ax.set_ylabel("Mean OOS Sharpe")
    ax.set_title("OOS Sharpe vs $T$: structural $\\mu$ (sector tilts), "
                 "oracle $\\mu$")
    ax.set_xticks(T_values)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 0.72)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
