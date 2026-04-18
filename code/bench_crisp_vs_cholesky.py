# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "numba>=0.59",
# ]
# ///
"""
Empirical wall-clock benchmark: CRISP vs LAPACK Cholesky across N.

Purpose: measure the per-solve wall-clock ratio between direct Cholesky
and CRISP (Gauss-Seidel on P_gamma w = mu) to validate the "5-10x
faster at N=500" and "crossover at N~3000-5000" claims in the paper.

Run with uv (recommended, no venv changes needed):
    uv run bench_crisp_vs_cholesky.py

uv will read the inline script metadata above, create an ephemeral
Python 3.13 environment with numpy/scipy/numba, and run the script.
Nothing is added to pyproject.toml or uv.lock.

Methodology:
  - Direct pipeline: scipy.linalg.cho_factor + cho_solve on Sigma.
  - CRISP pipeline: gamma=0.5, p=100 sweeps of Jacobi-preconditioned GS.
  - Three CRISP variants timed:
      (a) pure-Python reference (slow; upper bound on CRISP cost)
      (b) numba JIT-compiled (fair comparison to LAPACK)
      (c) numpy-vectorized Jacobi-like lower bound (optional)
  - 5 warm-up trials, then N_TRIALS timed trials, report median.
  - Single-solve and backtest-amortized (360-date) timings.

Outputs:
  - stdout: full table with ratios
  - results/11_bench_crisp_vs_cholesky.txt: raw numbers for the paper
"""

import io
import os
import platform
import sys
import time

import numpy as np
import scipy
import scipy.linalg as sla

try:
    import numba
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# CRISP implementations
# ---------------------------------------------------------------------------

def crisp_python(cov, mu, gamma=0.5, max_sweeps=100):
    """Pure-Python reference CRISP. Slow. Use as upper bound."""
    N = len(mu)
    diag = np.diag(cov).copy()
    if gamma < 1e-14:
        return mu / diag
    w = mu / diag  # warm start
    for _ in range(max_sweeps):
        for i in range(N):
            off = cov[i, :] @ w - cov[i, i] * w[i]
            w[i] = (mu[i] - gamma * off) / diag[i]
    return w


if HAVE_NUMBA:

    @njit(cache=True, fastmath=True)
    def crisp_numba(cov, mu, gamma, max_sweeps):
        """Numba-JIT CRISP. Fair comparison to LAPACK."""
        N = mu.shape[0]
        diag = np.empty(N)
        for i in range(N):
            diag[i] = cov[i, i]
        w = mu / diag
        if gamma < 1e-14:
            return w
        for _sweep in range(max_sweeps):
            for i in range(N):
                off = 0.0
                for j in range(N):
                    if j != i:
                        off += cov[i, j] * w[j]
                w[i] = (mu[i] - gamma * off) / diag[i]
        return w


# ---------------------------------------------------------------------------
# Direct Cholesky pipeline
# ---------------------------------------------------------------------------

def cholesky_solve(cov, mu):
    """scipy.linalg Cholesky factorization + triangular solve."""
    c, low = sla.cho_factor(cov, lower=True, check_finite=False)
    return sla.cho_solve((c, low), mu, check_finite=False)


# ---------------------------------------------------------------------------
# Covariance builder (matches study.py / paper's block structure)
# ---------------------------------------------------------------------------

def make_block_cov(N, n_sectors=5, rho_within=0.6, rho_cross=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n_per = N // n_sectors
    corr = np.full((N, N), rho_cross)
    for s in range(n_sectors):
        idx = slice(s * n_per, (s + 1) * n_per)
        corr[idx, idx] = rho_within
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(0.15, 0.40, N)
    cov = np.outer(vols, vols) * corr
    # Ensure SPD (block-correlation is SPD for these parameters, but symmetrize).
    cov = 0.5 * (cov + cov.T)
    return cov


def build_pgamma(cov, gamma=0.5):
    D = np.diag(np.diag(cov))
    return (1.0 - gamma) * D + gamma * cov


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def timed(fn, n_warm=3, n_trials=7):
    """Return (median, min, max) wall-clock seconds across n_trials."""
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(ts.min()), float(ts.max())


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("CRISP vs LAPACK Cholesky wall-clock benchmark")
    out("=" * 78)
    out(f"Platform:  {platform.platform()}")
    out(f"Processor: {platform.processor()}")
    out(f"Python:    {sys.version.split()[0]}")
    out(f"NumPy:     {np.__version__}")
    out(f"SciPy:     {scipy.__version__}")
    out(f"Numba:     {numba.__version__ if HAVE_NUMBA else 'NOT INSTALLED'}")
    out()
    out("NumPy BLAS/LAPACK configuration:")
    out("-" * 78)
    # np.show_config() prints directly; capture it by redirecting stdout.
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    np.show_config()
    sys.stdout = old_stdout
    out(buf.getvalue().rstrip())
    out("-" * 78)
    out()

    if not HAVE_NUMBA:
        out("WARNING: Numba not available. CRISP will time ~100x slower")
        out("         than on a compiled implementation. The paper's claims")
        out("         are about compiled CRISP (Remark rem:wall_clock).")
        out("         Install with: pip install numba")
        out()

    # Benchmark grid. Smaller N uses more trials.
    grid = [
        (100,   20),
        (200,   15),
        (500,   10),
        (1000,   7),
        (2000,   5),
        (3000,   3),
        (5000,   3),
    ]

    gamma = 0.5
    p = 100

    out(f"CRISP parameters: gamma={gamma}, p={p} sweeps")
    out(f"Direct pipeline:  scipy.linalg Cholesky (cho_factor + cho_solve)")
    out()
    out("=" * 78)
    out("SINGLE-SOLVE WALL-CLOCK (median over trials, milliseconds)")
    out("=" * 78)
    header = (f"{'N':>6s} {'Cholesky':>12s} {'CRISP-numba':>14s} "
              f"{'CRISP-py':>12s} {'ratio(py/chol)':>16s} "
              f"{'ratio(numba/chol)':>18s}")
    out(header)
    out("-" * len(header))

    results_table = []

    for N, n_trials in grid:
        cov = make_block_cov(N)
        rng = np.random.RandomState(42 + N)
        mu = rng.randn(N) * 0.02
        P = build_pgamma(cov, gamma=gamma)

        # Cholesky on Sigma
        t_chol_med, t_chol_min, _ = timed(
            lambda: cholesky_solve(cov, mu),
            n_warm=3, n_trials=n_trials)

        # CRISP numba on P_gamma
        if HAVE_NUMBA:
            # First call is JIT compile; that's excluded via warmup.
            t_nb_med, t_nb_min, _ = timed(
                lambda: crisp_numba(cov, mu, gamma, p),
                n_warm=3, n_trials=n_trials)
        else:
            t_nb_med = float('nan')
            t_nb_min = float('nan')

        # CRISP pure Python. Skip for large N where it becomes prohibitive.
        if N <= 1000:
            t_py_med, t_py_min, _ = timed(
                lambda: crisp_python(cov, mu, gamma=gamma, max_sweeps=p),
                n_warm=1, n_trials=max(3, n_trials // 2))
        else:
            t_py_med = float('nan')
            t_py_min = float('nan')

        ratio_py = t_py_med / t_chol_med if not np.isnan(t_py_med) else float('nan')
        ratio_nb = t_nb_med / t_chol_med if not np.isnan(t_nb_med) else float('nan')

        row = (f"{N:>6d} {t_chol_med*1e3:>12.3f} "
               f"{t_nb_med*1e3:>14.3f} {t_py_med*1e3:>12.3f} "
               f"{ratio_py:>15.1f}x {ratio_nb:>17.2f}x")
        out(row)
        results_table.append({
            'N': N,
            't_chol_ms': t_chol_med * 1e3,
            't_numba_ms': t_nb_med * 1e3,
            't_py_ms': t_py_med * 1e3,
            'ratio_py_over_chol': ratio_py,
            'ratio_numba_over_chol': ratio_nb,
        })

    out()
    out("=" * 78)
    out("INTERPRETATION")
    out("=" * 78)
    out("- ratio(numba/chol) > 1 means Cholesky is faster by that factor.")
    out("- ratio(numba/chol) < 1 means CRISP is faster by 1/ratio.")
    out("- Crossover = N at which ratio(numba/chol) crosses 1.0.")
    out()
    if HAVE_NUMBA:
        out("Paper claim (Remark rem:wall_clock):")
        out("  - At N=500, Cholesky 5-10x faster.")
        out("  - Single-solve crossover at N~3000-5000.")
        out()
        out("Measured values:")
        for r in results_table:
            verdict = ""
            if r['N'] == 500:
                lo, hi = 5.0, 10.0
                val = r['ratio_numba_over_chol']
                if lo <= val <= hi:
                    verdict = f"  -> CONSISTENT with paper's 5-10x at N=500"
                elif val > hi:
                    verdict = f"  -> Cholesky faster than paper claims ({val:.1f}x vs 5-10x)"
                elif val < lo:
                    verdict = f"  -> Cholesky less dominant than paper claims ({val:.1f}x < 5x)"
                out(f"  N=500: ratio={val:.2f}x{verdict}")
            if r['N'] in (3000, 5000):
                val = r['ratio_numba_over_chol']
                if val < 1.0:
                    verdict = f"  -> CRISP dominates (paper's crossover confirmed)"
                else:
                    verdict = f"  -> Cholesky still {val:.2f}x faster"
                out(f"  N={r['N']}: ratio={val:.2f}x{verdict}")

    out()
    out("=" * 78)
    out("BACKTEST-AMORTIZED (360 rebalance dates, median per-solve)")
    out("=" * 78)
    out("Each date: re-estimate Sigma from fresh data, re-solve from scratch.")
    out("(Cholesky cannot reuse its factorization across dates.)")
    out()

    bt_dates = 360
    # Only run backtest timing at representative N to keep runtime reasonable.
    bt_grid = [500, 1000, 2000] + ([3000] if HAVE_NUMBA else [])

    header = f"{'N':>6s} {'Cholesky':>14s} {'CRISP-numba':>14s} {'ratio':>10s}"
    out(header)
    out("-" * len(header))

    for N in bt_grid:
        # Pre-generate 360 different covariances to simulate rolling estimation.
        covs = [make_block_cov(N, seed=42 + d) for d in range(bt_dates)]
        mus = [np.random.RandomState(42 + d).randn(N) * 0.02
               for d in range(bt_dates)]

        # Cholesky total
        for cov, mu in zip(covs[:3], mus[:3]):  # warmup
            cholesky_solve(cov, mu)
        t0 = time.perf_counter()
        for cov, mu in zip(covs, mus):
            cholesky_solve(cov, mu)
        t_chol_total = time.perf_counter() - t0

        # CRISP numba total
        if HAVE_NUMBA:
            for cov, mu in zip(covs[:3], mus[:3]):  # warmup
                crisp_numba(cov, mu, gamma, p)
            t0 = time.perf_counter()
            for cov, mu in zip(covs, mus):
                crisp_numba(cov, mu, gamma, p)
            t_nb_total = time.perf_counter() - t0
            ratio = t_nb_total / t_chol_total
        else:
            t_nb_total = float('nan')
            ratio = float('nan')

        per_chol = t_chol_total / bt_dates * 1e3
        per_nb = (t_nb_total / bt_dates * 1e3) if HAVE_NUMBA else float('nan')
        out(f"{N:>6d} {per_chol:>13.3f}ms {per_nb:>13.3f}ms {ratio:>9.2f}x")

    out()
    out("=" * 78)
    out("NOTES")
    out("=" * 78)
    out("1. Pure-Python CRISP is ~100x slower than Numba-compiled; its timing")
    out("   is a reference ceiling, not the paper's claim.")
    out("2. The paper's claims in Remark rem:wall_clock assume compiled CRISP")
    out("   (Numba/Cython/C) against LAPACK Cholesky with BLAS-3 throughput.")
    out("3. Crossover depends on hardware: BLAS-3 / BLAS-1 GFLOPS ratio varies")
    out("   from ~3x to ~20x across systems.")
    out("4. CRISP here runs at gamma=0.5 with p=100 sweeps (paper's default")
    out("   recommendation). Cholesky solves the raw Sigma system Sigma w = mu.")
    out("   These solve different linear systems and produce different portfolios;")
    out("   the comparison is between two end-to-end solver pipelines, which is")
    out("   what a practitioner would actually do.")

    # Write results file
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "11_bench_crisp_vs_cholesky.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
