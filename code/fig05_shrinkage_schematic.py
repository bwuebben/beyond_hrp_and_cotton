"""
fig05_shrinkage_schematic.py
============================

Generates the Method B shrinkage-operator / Gauss-Seidel diagnostic figure
for Section 5 of "Beyond HRP and Schur: Hierarchical and Iterative Methods
for General Mean-Variance Portfolios".

The figure consists of two panels on a single row:

    (left)  Preconditioned condition number kappa(D^{-1} P_gamma) as a
            function of gamma, interpolating monotonically from 1 at
            gamma = 0 to kappa(C) at gamma = 1.  Computed from the
            closed-form expression ((1-g) + g lambda_1) / ((1-g) + g lambda_N).

    (right) Empirical Gauss-Seidel sweep count to reach relative residual
            epsilon = 1e-6 on the linear system P_gamma w = mu, using the
            same sample covariance and a random signal.  The sweep-count
            curve should track the condition-number curve monotonically up
            to discretization, illustrating the rate theorem of Section 5.4.

The covariance is a 60-asset block-structured model with four clusters
and an idiosyncratic volatility spread, chosen so that the correlation
matrix C has a moderate spread of eigenvalues (condition number in the
low tens rather than pathological) so that both panels are visible on a
linear axis.

Run (from project root):

    python3 figures/code/fig05_shrinkage_schematic.py

Output:

    figures/output/fig05_shrinkage_schematic.pdf
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_block_cov(N: int, n_blocks: int, rho_within: float, rho_cross: float,
                   vol_lo: float, vol_hi: float, seed: int) -> np.ndarray:
    """Block-correlation matrix with heterogeneous volatilities.

    Returns an SPD covariance Sigma = diag(vol) @ C @ diag(vol).
    """
    rng = np.random.RandomState(seed)
    n_per = N // n_blocks
    corr = np.full((N, N), rho_cross)
    for b in range(n_blocks):
        i0 = b * n_per
        i1 = (b + 1) * n_per if b < n_blocks - 1 else N
        corr[i0:i1, i0:i1] = rho_within
    np.fill_diagonal(corr, 1.0)
    # force SPD after manual fill
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vols = rng.uniform(vol_lo, vol_hi, N)
    sigma = np.outer(vols, vols) * corr
    sigma = 0.5 * (sigma + sigma.T)
    return sigma


def p_gamma(sigma: np.ndarray, gamma: float) -> np.ndarray:
    D = np.diag(np.diag(sigma))
    return (1.0 - gamma) * D + gamma * sigma


def gauss_seidel_sweeps(sigma: np.ndarray, mu: np.ndarray, gamma: float,
                        tol: float, max_sweeps: int) -> int:
    """Scalar Gauss-Seidel on P_gamma w = mu.

    Returns the number of sweeps to reach the relative residual tolerance
    ||P_gamma w - mu|| / ||mu|| <= tol. If max_sweeps is reached, returns
    max_sweeps.
    """
    N = sigma.shape[0]
    diag = np.diag(sigma)
    # initial guess: diagonal solution (the gamma = 0 exact solution)
    w = mu / diag
    mu_norm = np.linalg.norm(mu)
    if mu_norm == 0.0:
        return 0
    for sweep in range(1, max_sweeps + 1):
        # one Gauss-Seidel sweep in-place
        for i in range(N):
            off = sigma[i, :] @ w - sigma[i, i] * w[i]
            w[i] = (mu[i] - gamma * off) / diag[i]
        # residual of P_gamma w - mu
        pgw = p_gamma(sigma, gamma) @ w
        res = np.linalg.norm(pgw - mu) / mu_norm
        if res <= tol:
            return sweep
    return max_sweeps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig05_shrinkage_schematic.pdf")

    # ----- covariance model -----
    N = 60
    sigma = make_block_cov(
        N=N,
        n_blocks=4,
        rho_within=0.65,
        rho_cross=0.20,
        vol_lo=0.15,
        vol_hi=0.40,
        seed=20260411,
    )
    D = np.diag(np.diag(sigma))
    Dinv_half = np.diag(1.0 / np.sqrt(np.diag(sigma)))
    C = Dinv_half @ sigma @ Dinv_half
    C = 0.5 * (C + C.T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    lam_max, lam_min = lam[0], lam[-1]
    kappa_C = lam_max / lam_min

    # signal: random but fixed seed
    rng = np.random.RandomState(7)
    mu = rng.randn(N) * 0.02

    # ----- panel (a): kappa(D^{-1} P_gamma) closed form -----
    gammas_dense = np.linspace(0.0, 1.0, 401)
    kappa_curve = ((1.0 - gammas_dense) + gammas_dense * lam_max) / \
                  ((1.0 - gammas_dense) + gammas_dense * lam_min)

    # ----- panel (b): empirical sweep count on a coarser grid -----
    gammas_emp = np.array(
        [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    )
    tol = 1e-6
    max_sweeps = 2000
    sweeps = np.array([
        gauss_seidel_sweeps(sigma, mu, float(g), tol, max_sweeps)
        for g in gammas_emp
    ])

    # ----- plotting -----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    ax.plot(gammas_dense, kappa_curve, color="#1f77b4", lw=2.0,
            label=r"$\kappa(D^{-1} P_\gamma)$")
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
    ax.axhline(kappa_C, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\kappa(D^{-1} P_\gamma)$")
    ax.set_title("Preconditioned condition number")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.9, kappa_C * 1.08)
    ax.text(0.03, 1.0 + 0.02 * kappa_C, r"$\kappa = 1$",
            fontsize=9, color="gray", va="bottom")
    ax.text(0.55, kappa_C * 1.01, rf"$\kappa(C) \approx {kappa_C:.1f}$",
            fontsize=9, color="gray", va="bottom")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    ax = axes[1]
    ax.plot(gammas_emp, sweeps, marker="o", color="#d62728", lw=1.8,
            markersize=5.5, label=r"sweeps to $\|r\|/\|\mu\|\leq 10^{-6}$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Gauss--Seidel sweeps")
    ax.set_title("Empirical sweep count")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(sweeps) * 1.15 + 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.suptitle(
        r"Method B: $P_\gamma = (1-\gamma)\,D + \gamma\,\Sigma$. "
        r"Correlation conditioning drives both condition number and sweep count.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # sanity print
    print(f"kappa(C) = {kappa_C:.4f}")
    print(f"lambda_max = {lam_max:.4f}, lambda_min = {lam_min:.4f}")
    print("gamma, sweeps:")
    for g, s in zip(gammas_emp, sweeps):
        print(f"  gamma={g:4.2f}  sweeps={s:4d}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
