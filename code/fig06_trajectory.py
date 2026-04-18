"""fig06_trajectory.py

Figure 6.1: the shrinkage trajectory across three difficulty levels.

Three-panel figure showing direction error vs. gamma at different
correlation-matrix condition numbers kappa(C). Each panel shows the
exact P_gamma^{-1} mu curve (monotone) and CRISP at three sweep
budgets. The panels demonstrate that:
  - Easy problems (low kappa): CRISP converges fast, no interior optimum
  - Moderate problems (base case): mild interior optimum at finite sweeps
  - Hard problems (worst case): strong interior optimum, many sweeps needed

Visualizes Propositions 6.3 (monotone trajectory) and 6.5 (interior
gamma* optimum at finite sweep budgets).
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study import (make_structured_cov, case_9_worst_case_highcond,
                   method_b_solve, dir_err, _kappa_corr)


def compute_trajectory(cov, mu, gammas, sweep_counts):
    """Compute direction error curves for exact and finite-sweep CRISP."""
    w_star = np.linalg.solve(cov, mu)
    diag = np.diag(cov)

    err_exact = np.zeros_like(gammas)
    err_sweeps = {p: np.zeros_like(gammas) for p in sweep_counts}

    for k, g in enumerate(gammas):
        P = (1.0 - g) * np.diag(diag) + g * cov
        w_exact = np.linalg.solve(P, mu)
        err_exact[k] = dir_err(w_exact, w_star)
        for p in sweep_counts:
            w = method_b_solve(cov, mu, g, max_sweeps=p, tol=0.0)
            err_sweeps[p][k] = dir_err(w, w_star)

    return err_exact, err_sweeps


def main() -> None:
    rng = np.random.RandomState(42)
    gammas = np.linspace(0.0, 1.0, 41)

    # Three difficulty levels
    # Easy: low correlation, kappa(C) ~ 10
    cov_easy, _ = make_structured_cov(200, 5, 0.1, 0.02, seed=42)
    mu_easy = rng.randn(200) * 0.02

    # Moderate: paper's base case, kappa(C) ~ 121
    cov_mod, _ = make_structured_cov(200, 5, 0.6, 0.15, seed=42)
    mu_mod = rng.randn(200) * 0.02

    # Hard: worst-case adversarial, kappa(C) ~ 3800
    cov_hard, mu_hard, _ = case_9_worst_case_highcond(N=200, seed=42)

    panels = [
        ("Easy", cov_easy, mu_easy, [50, 200, 1000]),
        ("Moderate (base case)", cov_mod, mu_mod, [100, 500, 2000]),
        ("Hard (worst case)", cov_hard, mu_hard, [200, 2000, 10000]),
    ]

    sweep_colors = ["#d62728", "#2ca02c", "#ff7f0e"]
    sweep_markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)

    for ax, (title, cov, mu, sweep_counts) in zip(axes, panels):
        kc = _kappa_corr(cov)
        err_exact, err_sweeps = compute_trajectory(
            cov, mu, gammas, sweep_counts)

        ax.plot(gammas, err_exact, color="#1f77b4", lw=2.5,
                label=r"exact $P_\gamma^{-1}\mu$")

        for p, col, mkr in zip(sweep_counts, sweep_colors, sweep_markers):
            err = err_sweeps[p]
            k_min = int(np.argmin(err))
            ax.plot(gammas, err, color=col, lw=1.8, marker=mkr, ms=2.5,
                    label=r"CRISP $p{=}" + f"{p:,}" + r"$")
            # Mark interior optimum
            if 0 < k_min < len(gammas) - 1:
                ax.axvline(gammas[k_min], color=col, ls="--", lw=0.8,
                           alpha=0.5)

        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_title(rf"{title}, $\kappa(C)\approx{kc:.0f}$", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        ax.set_ylim(1e-8, 1.5)

    axes[0].set_ylabel(
        r"direction error $\mathrm{dir}(\hat w,\,\Sigma^{-1}\mu)$")

    fig.suptitle(
        "Shrinkage trajectory: exact (monotone) vs finite-sweep CRISP "
        "(interior optimum at finite $p$)",
        fontsize=11, y=1.01)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig06_trajectory.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(out_path)}")

    # Print summary
    for title, cov, mu, sweep_counts in panels:
        kc = _kappa_corr(cov)
        err_exact, err_sweeps = compute_trajectory(
            cov, mu, gammas, sweep_counts)
        print(f"\n{title} (kappa(C) = {kc:.1f}):")
        print(f"  exact: dir(0)={err_exact[0]:.3e}, dir(1)={err_exact[-1]:.3e}")
        for p in sweep_counts:
            err = err_sweeps[p]
            k_min = int(np.argmin(err))
            print(f"  p={p:>6,}: gamma*={gammas[k_min]:.2f}, "
                  f"dir*={err[k_min]:.3e}")


if __name__ == "__main__":
    main()
