"""
figC_a1_cosine_histogram.py

Histogram of Method A1's cosine with the oracle Markowitz portfolio,
under estimation noise, for the structured sector-tilt signal. Mirrors
the noise-panel loop of experiment_3_a1_deep_dive in walkforward.py but
collects the per-trial cosines and plots their distribution at three
sample sizes T in {60, 240, 1000}.

Illustrates the Appendix C point: the cosine distribution is centered
near zero with std ~0.8, independently of T, so A1's direction relative
to Markowitz is essentially random.

Usage:
    python figures/code/figC_a1_cosine_histogram.py
Writes:
    figures/output/figC_a1_cosine_histogram.pdf
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "figures", "output", "figC_a1_cosine_histogram.pdf")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from study import (  # noqa: E402
    make_structured_cov,
    build_hrp_tree,
    method_a1_weights,
)


def sector_tilt_mu(N: int, n_sectors: int = 5,
                   tilts=(0.04, -0.04, 0.02, -0.02, 0.0)) -> np.ndarray:
    mu = np.zeros(N)
    per = N // n_sectors
    for s in range(n_sectors):
        mu[s * per:(s + 1) * per] = tilts[s]
    return mu


def collect_cosines(N: int, T: int, n_mc: int, gamma: float = 0.5) -> np.ndarray:
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    mu_true = sector_tilt_mu(N)
    w_or = np.linalg.solve(cov_true, mu_true)
    rng = np.random.RandomState(5)
    cos_vals = []
    for _ in range(n_mc):
        samples = rng.multivariate_normal(mu_true, cov_true, T)
        cov_hat = np.cov(samples.T) + 1e-4 * np.eye(N)
        tree = build_hrp_tree(cov_hat)
        w = method_a1_weights(cov_hat, mu_true, tree, gamma)
        denom = np.linalg.norm(w) * np.linalg.norm(w_or) + 1e-30
        cos_vals.append(float(w @ w_or) / denom)
    return np.asarray(cos_vals)


def main() -> None:
    N = 100
    n_mc = 400  # larger than walkforward.py's 60 so the histogram is smooth
    Ts = [60, 240, 1000]
    colors = {60: "#d62728", 240: "#1f77b4", 1000: "#2ca02c"}

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))
    bins = np.linspace(-1.0, 1.0, 41)
    for T in Ts:
        cos_vals = collect_cosines(N, T, n_mc)
        mean = float(np.mean(cos_vals))
        std = float(np.std(cos_vals))
        frac_neg = float((cos_vals < 0).mean())
        label = (f"$T={T}$: mean ${mean:+.2f}$, std ${std:.2f}$, "
                 f"frac$<$0 ${frac_neg:.2f}$")
        ax.hist(cos_vals, bins=bins, histtype="stepfilled",
                alpha=0.35, color=colors[T], label=label,
                density=True, edgecolor=colors[T], linewidth=1.6)
        print(f"T={T}: N={len(cos_vals)}, mean={mean:+.3f}, "
              f"std={std:.3f}, frac_neg={frac_neg:.2f}")

    ax.axvline(0.0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel(r"$\cos(w_{\mathrm{A1}},\,w^\star)$")
    ax.set_ylabel("density")
    ax.set_title(r"Method A1 direction cosine with oracle Markowitz, "
                 r"structured $\mu$, $\gamma=0.5$")
    ax.set_xlim(-1.02, 1.02)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
