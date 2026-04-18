"""
computeC_a1_pathology.py

Reproduces the headline numbers of Appendix C (Method A1 and the
recursive-normalization sign pathology). Imports only primitives from
study.py and reruns the NOISELESS panel of experiment_3_a1_deep_dive
from walkforward.py:

  1. Build the block-structured covariance used throughout Section 9.
  2. Construct the structured sector-tilt mu and the random-mu baseline.
  3. For each (gamma, mu) pair, run Method A1 on (cov_true, mu) with the
     HRP tree built from cov_true, and report:
       - OOS Sharpe on the true problem,
       - cosine with the oracle portfolio w_or = cov_true^{-1} mu,
       - gross leverage ||w||_1.
  4. Print 1/N, HRP, Direct, A1 g in {0, 0.5, 1.0} exactly as they appear
     in results/03_a1_deep_dive.txt under the "All methods on TRUE cov"
     block, so a reader can verify the numbers cited in Table C.1.

The output reproduces the numbers quoted in Appendix C.

Usage:
    python figures/code/computeC_a1_pathology.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from study import (  # noqa: E402
    make_structured_cov,
    build_hrp_tree,
    hrp_flat_weights,
    method_a1_weights,
    method_a1_l1_weights,
    method_a3_weights,
)


def sector_tilt_mu(N: int, n_sectors: int = 5,
                   tilts=(0.04, -0.04, 0.02, -0.02, 0.0)) -> np.ndarray:
    """Block-constant sector tilt exactly as in walkforward.py."""
    mu = np.zeros(N)
    per = N // n_sectors
    for s in range(n_sectors):
        mu[s * per:(s + 1) * per] = tilts[s]
    return mu


def sharpe_and_cos(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                   w_or: np.ndarray) -> tuple[float, float, float]:
    var = float(w @ cov @ w)
    sh = float(w @ mu) / np.sqrt(var) if var > 1e-15 else 0.0
    denom = np.linalg.norm(w) * np.linalg.norm(w_or) + 1e-30
    cs = float(w @ w_or) / denom
    gross = float(np.abs(w).sum())
    return sh, cs, gross


def evaluate(label: str, mu: np.ndarray, cov_true: np.ndarray) -> None:
    N = cov_true.shape[0]
    w_or = np.linalg.solve(cov_true, mu)
    oracle_var = float(w_or @ cov_true @ w_or)
    oracle_sh = float(w_or @ mu) / np.sqrt(oracle_var)
    tree = build_hrp_tree(cov_true)

    print(f"\n  {label}, Oracle Sharpe = {oracle_sh:.3f}")
    print(f"  {'method':<12} {'Sharpe':>10} {'cos(w,oracle)':>14} "
          f"{'|w|1':>8}")
    print("  " + "-" * 50)

    # Baselines
    w_eq = np.ones(N) / N
    sh, cs, g = sharpe_and_cos(w_eq, mu, cov_true, w_or)
    print(f"  {'1/N':<12} {sh:10.3f} {cs:14.3f} {g:8.2f}")

    w_hrp = hrp_flat_weights(cov_true, tree)
    sh, cs, g = sharpe_and_cos(w_hrp, mu, cov_true, w_or)
    print(f"  {'HRP':<12} {sh:10.3f} {cs:14.3f} {g:8.2f}")

    sh, cs, g = sharpe_and_cos(w_or, mu, cov_true, w_or)
    print(f"  {'Direct':<12} {sh:10.3f} {cs:14.3f} {g:8.2f}")

    for gamma in (0.0, 0.5, 1.0):
        w = method_a1_weights(cov_true, mu, tree, gamma)
        sh, cs, g = sharpe_and_cos(w, mu, cov_true, w_or)
        name = f"A1  g={gamma:.1f}"
        print(f"  {name:<12} {sh:10.3f} {cs:14.3f} {g:8.2f}")

    for gamma in (0.0, 0.5, 1.0):
        w = method_a1_l1_weights(cov_true, mu, tree, gamma)
        sh, cs, g = sharpe_and_cos(w, mu, cov_true, w_or)
        name = f"A1L1 g={gamma:.1f}"
        print(f"  {name:<12} {sh:10.3f} {cs:14.3f} {g:8.2f}")

    for gamma in (0.0, 0.5, 1.0):
        w = method_a3_weights(cov_true, mu, tree, gamma)
        sh, cs, g = sharpe_and_cos(w, mu, cov_true, w_or)
        name = f"A3  g={gamma:.1f}"
        print(f"  {name:<12} {sh:10.3f} {cs:14.3f} {g:8.2f}")


def main() -> None:
    N = 100
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)

    print("=" * 78)
    print("Appendix C: A1 on the NOISELESS problem (Sigma = Sigma_true)")
    print("=" * 78)

    mu_struct = sector_tilt_mu(N)
    evaluate("structured sector-tilt mu", mu_struct, cov_true)

    rng = np.random.RandomState(7)
    mu_rand = rng.randn(N) * 0.02
    evaluate("random mu (seed 7)", mu_rand, cov_true)


if __name__ == "__main__":
    main()
