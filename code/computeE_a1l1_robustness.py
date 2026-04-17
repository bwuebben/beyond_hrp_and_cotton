"""
computeE_a1l1_robustness.py
============================

Appendix E robustness panel for A1-L1 (and A3/HRP-mu for comparison).

Tests sensitivity of A1-L1 to:
  1. Ridge regularization: lambda in {1e-8, 1e-6, 1e-4, 1e-2}
  2. Linkage method: Ward, single, complete, average
  3. Gamma sweep: gamma in {0.0, 0.3, 0.5, 0.7, 1.0}

All experiments use the base-case covariance (N=100, 5 sectors,
rho_w=0.6, rho_c=0.15) with structural sector-tilt mu and T=120.

Usage:
    python figures/code/computeE_a1l1_robustness.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from study import (make_structured_cov, build_hrp_tree, hrp_flat_weights,
                   method_a1_l1_weights, method_a3_weights,
                   method_b_solve, TreeNode, build_tree_from_linkage)
from walkforward import make_sector_tilt_mu


def build_tree_with_linkage(cov, method='ward'):
    """Build HRP tree with a specified linkage method."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method=method)
    return build_tree_from_linkage(Z, cov.shape[0])


def run_mc(cov_true, mu_true, n_mc, T, ridge, linkage_method,
           gammas, seed=999):
    """Run Monte Carlo and return mean OOS Sharpe for each (method, gamma)."""
    N = cov_true.shape[0]
    rng = np.random.RandomState(seed)
    results = {}
    for g in gammas:
        for tag in ['A1L1', 'A3', 'B']:
            results[(tag, g)] = []

    for _ in range(n_mc):
        samples = rng.multivariate_normal(mu_true, cov_true, T)
        cov_hat = np.cov(samples.T) + ridge * np.eye(N)
        try:
            tree = build_tree_with_linkage(cov_hat, linkage_method)
        except Exception:
            continue
        for g in gammas:
            for tag, fn in [
                ('A1L1', lambda c, m, t, g=g: method_a1_l1_weights(c, m, t, g)),
                ('A3', lambda c, m, t, g=g: method_a3_weights(c, m, t, g)),
                ('B', lambda c, m, t, g=g: method_b_solve(c, m, g, max_sweeps=100)),
            ]:
                try:
                    w = fn(cov_hat, mu_true, tree)
                    if not np.all(np.isfinite(w)):
                        results[(tag, g)].append(np.nan)
                        continue
                    var = float(w @ cov_true @ w)
                    sr = float(w @ mu_true) / np.sqrt(var) if var > 1e-15 else 0.0
                    results[(tag, g)].append(sr)
                except Exception:
                    results[(tag, g)].append(np.nan)
    return results


def print_table(results, gammas, title):
    print(f"\n{title}")
    print(f"  {'gamma':>6s}  {'A1-L1':>10s}  {'A3/HRP-mu':>10s}  "
          f"{'CRISP':>10s}  {'A1L1/A3':>8s}")
    print("  " + "-" * 52)
    for g in gammas:
        vals = {}
        for tag in ['A1L1', 'A3', 'B']:
            v = np.array(results[(tag, g)], dtype=float)
            valid = v[np.isfinite(v)]
            vals[tag] = float(np.mean(valid)) if len(valid) > 0 else np.nan
        ratio = vals['A1L1'] / vals['A3'] if abs(vals['A3']) > 1e-10 else np.nan
        print(f"  {g:6.1f}  {vals['A1L1']:10.3f}  {vals['A3']:10.3f}  "
              f"{vals['B']:10.3f}  {ratio:8.2f}x")


def main():
    N = 100
    n_mc = 60
    T = 120
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    mu_true = make_sector_tilt_mu(N)
    gammas = [0.0, 0.3, 0.5, 0.7, 1.0]

    w_or = np.linalg.solve(cov_true, mu_true)
    oracle_sh = (w_or @ mu_true) / np.sqrt(w_or @ cov_true @ w_or)
    print("=" * 70)
    print(f"Appendix E: A1-L1 Robustness Panel (N={N}, T={T}, n_mc={n_mc})")
    print(f"Oracle Sharpe = {oracle_sh:.3f}")
    print("=" * 70)

    # --- Experiment 1: Ridge sensitivity ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Ridge Regularization Sensitivity")
    print("=" * 70)
    for ridge in [1e-8, 1e-6, 1e-4, 1e-2]:
        results = run_mc(cov_true, mu_true, n_mc, T, ridge, 'ward',
                         gammas, seed=999)
        print_table(results, gammas, f"ridge = {ridge:.0e}")

    # --- Experiment 2: Linkage sensitivity ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Linkage Method Sensitivity")
    print("=" * 70)
    for lmethod in ['ward', 'single', 'complete', 'average']:
        results = run_mc(cov_true, mu_true, n_mc, T, 1e-4, lmethod,
                         gammas, seed=999)
        print_table(results, gammas, f"linkage = {lmethod}")

    # --- Experiment 3: Sample size sensitivity ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sample Size Sensitivity (ridge=1e-4, ward)")
    print("=" * 70)
    for T_test in [60, 120, 240, 500]:
        results = run_mc(cov_true, mu_true, n_mc, T_test, 1e-4, 'ward',
                         gammas, seed=999)
        print_table(results, gammas, f"T = {T_test} (N/T = {N/T_test:.2f})")


if __name__ == "__main__":
    main()
