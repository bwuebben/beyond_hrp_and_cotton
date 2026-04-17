"""
compute09_adaptive_gamma.py
===========================

Numerical validation of the adaptive gamma rule derived in Section 6.5:

    gamma*  approx  1 / (1 + c * kappa(C)^2 * (N/T) / IC^2)

Runs three experiments and writes artifacts consumed by Section 9.5 and
Section 6's Figure fig06:bias_variance_curves.

Experiments
-----------
Exp 1 (calibration):
    4 covariance regimes x 4 T/N ratios x 3 IC levels = 48 cells.
    For each cell: draw 80 Monte Carlo replications of (hat_Sigma, hat_mu),
    run Method B on a grid of 11 gamma values with 100 sweeps, score each
    by closed-form OOS Sharpe under the true (Sigma, mu).  gamma*_emp is
    the argmax of mean OOS Sharpe.  Solve for c in each cell from the
    formula; report bar{c}, sigma_c, c_max/c_min.

Exp 2 (validation):
    Expand the grid to 4 x 9 x 3 = 108 cells.  Compare gamma*_emp to
    gamma*_pred from the formula with bar{c} from Exp 1.  Scatter plot
    and summary statistics.  Dump three representative L(gamma) curves
    for the Section 6 figure.

Exp 3 (spectral sensitivity):
    Under a spiked covariance with 1, 5, 20 spikes plus a Marchenko-
    Pastur decay sub-regime, hold kappa(C) fixed and fit c per sub-regime.
    Tests whether kappa(C) alone explains the calibration or whether
    spectral shape matters.

Artifacts (under results/sec09_adaptive/)
-----------------------------------------
    exp1_calibration.csv      one row per (regime, T/N, IC) cell
    exp2_validation.csv       one row per cell in the 108-cell grid
    exp3_spectral.csv         one row per spike-configuration cell
    lgamma_curves.npz         L(gamma) arrays for the three representative
                              cells used by fig06_bias_variance_curves.py
    summary.txt               human-readable summary

Usage
-----
    python figures/code/compute09_adaptive_gamma.py

Design notes
------------
  - Monte Carlo noise is kept under control by using mean Sharpe over
    n_mc replications rather than a single draw.  n_mc = 80 is a
    compromise between Monte Carlo error and wall-clock time.
  - Method B is run at 100 sweeps everywhere to match the paper's
    recommended sweep budget.
  - OOS Sharpe is evaluated in closed form as (w^T mu_true) /
    sqrt(w^T Sigma_true w).  There is no held-out sample simulation;
    the closed form eliminates OOS estimation noise entirely, which is
    what Experiment 2's `T_oos=5000` simulation in the theory note was
    approximating.
  - Seed 42 throughout for reproducibility.
"""

from __future__ import annotations

import os
import csv
import time
from collections import defaultdict
from typing import Callable

import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESDIR = os.path.join(ROOT, "results", "sec09_adaptive")
os.makedirs(RESDIR, exist_ok=True)

from study import (build_hrp_tree, method_a1_l1_weights,  # noqa: E402
                   method_a3_weights, method_b_solve)

# --- We validate the *exact* Method B fixed point (the infinite-sweep
# limit). The adaptive rule derived in Section 6.5 is a statement about
# the bias-variance tradeoff of w(gamma) = P_gamma^{-1} hat_mu; the
# convergence-slack contribution analysed in Proposition 6.5 is a
# separate effect that Method B at 100 sweeps controls down to
# negligible levels on the cells in this calibration grid.
def method_b_solve_exact(cov_hat: np.ndarray, mu_hat: np.ndarray,
                         gamma: float) -> np.ndarray:
    N = cov_hat.shape[0]
    d = np.diag(cov_hat)
    if gamma < 1e-14:
        return mu_hat / d
    P = (1.0 - gamma) * np.diag(d) + gamma * cov_hat
    return np.linalg.solve(P, mu_hat)


# ================================================================
# Covariance regimes
# ================================================================

def make_factor(N: int, K: int, seed: int) -> np.ndarray:
    """K-factor model covariance with Gaussian loadings."""
    rng = np.random.RandomState(seed)
    B = rng.randn(N, K) * 0.25
    sigma_f = np.diag(rng.uniform(0.10, 0.30, K) ** 2)
    idio = np.diag(rng.uniform(0.10, 0.30, N) ** 2)
    cov = B @ sigma_f @ B.T + idio
    return cov


def make_block(N: int, n_sectors: int, rho_w: float, rho_c: float,
               seed: int) -> np.ndarray:
    """Block covariance with within/cross sector correlations and
    uniform [0.15, 0.40] volatilities."""
    rng = np.random.RandomState(seed)
    n_per = N // n_sectors
    corr = np.full((N, N), rho_c)
    for s in range(n_sectors):
        lo = s * n_per
        hi = (s + 1) * n_per if s < n_sectors - 1 else N
        corr[lo:hi, lo:hi] = rho_w
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(0.15, 0.40, N)
    return np.outer(vols, vols) * corr


def make_spiked(N: int, n_spikes: int, spike_mag: float,
                seed: int, mp_shape: bool = False) -> np.ndarray:
    """Spiked correlation: identity plus n_spikes rank-1 perturbations
    with magnitude spike_mag, scaled to unit vols.  If mp_shape, use a
    continuous Marchenko-Pastur-like decay instead of discrete spikes."""
    rng = np.random.RandomState(seed)
    vols = rng.uniform(0.15, 0.40, N)
    if mp_shape:
        # Put eigenvalues on a decaying geometric sequence.
        eigs = np.linspace(1.0, spike_mag, N)[::-1]
        # random orthogonal basis
        A = rng.randn(N, N)
        Q, _ = np.linalg.qr(A)
        corr = Q @ np.diag(eigs) @ Q.T
        # normalize back to unit diagonal
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    else:
        corr = np.eye(N)
        for _ in range(n_spikes):
            v = rng.randn(N)
            v /= np.linalg.norm(v)
            corr = corr + spike_mag * np.outer(v, v)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return np.outer(vols, vols) * corr


def make_equicorr(N: int, rho: float, seed: int) -> np.ndarray:
    """Equi-correlation with rho, uniform [0.15, 0.40] volatilities."""
    rng = np.random.RandomState(seed)
    corr = np.full((N, N), rho)
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(0.15, 0.40, N)
    return np.outer(vols, vols) * corr


def kappa_corr(cov: np.ndarray) -> float:
    """Condition number of the correlation matrix derived from cov."""
    d = np.sqrt(np.diag(cov))
    C = cov / np.outer(d, d)
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 1e-12, None)
    return float(w.max() / w.min())


# ================================================================
# Regime registry used by Experiments 1 and 2
# ================================================================

def regime_factory(name: str) -> Callable[[int, int], np.ndarray]:
    """Return a covariance builder that takes (N, seed) and returns Sigma."""
    if name == "factor":
        return lambda N, seed: make_factor(N, K=3, seed=seed)
    if name == "block":
        return lambda N, seed: make_block(N, n_sectors=5, rho_w=0.6,
                                          rho_c=0.15, seed=seed)
    if name == "spiked":
        return lambda N, seed: make_spiked(N, n_spikes=5, spike_mag=3.0,
                                           seed=seed)
    if name == "equicorr":
        return lambda N, seed: make_equicorr(N, rho=0.6, seed=seed)
    raise ValueError(f"unknown regime {name}")


REGIMES = ["factor", "block", "spiked", "equicorr"]


# ================================================================
# Monte Carlo core
# ================================================================

def inject_ic_noise(mu_true: np.ndarray, ic: float,
                    rng: np.random.RandomState) -> np.ndarray:
    """Return hat_mu = mu_true + eps with Pearson correlation ~ ic.
    For mu_true with cross-sectional std sigma_mu, eps is Gaussian with
    variance sigma_mu^2 * (1/ic^2 - 1)."""
    sigma_mu = float(np.std(mu_true))
    if sigma_mu < 1e-12 or ic >= 1.0:
        return mu_true.copy()
    sigma_eps2 = sigma_mu * sigma_mu * (1.0 / (ic * ic) - 1.0)
    eps = rng.randn(len(mu_true)) * np.sqrt(sigma_eps2)
    return mu_true + eps


def oos_sharpe(w: np.ndarray, mu_true: np.ndarray,
               cov_true: np.ndarray) -> float:
    """Closed-form OOS Sharpe = (w'mu)/sqrt(w'Sigma w), under the true DGP."""
    num = float(w @ mu_true)
    var = float(w @ cov_true @ w)
    if var < 1e-18:
        return 0.0
    return num / np.sqrt(var)


def run_cell_tree(cov_true: np.ndarray, mu_true: np.ndarray, T: int,
                  ic: float, gamma_grid: np.ndarray, n_mc: int,
                  ridge: float, seed: int) -> dict[str, np.ndarray]:
    """
    Monte Carlo for tree-based methods (A1-L1 and A3/HRP-mu) across gamma.
    Returns dict with keys 'A1L1' and 'A3', each an array of shape
    (len(gamma_grid),) with mean OOS Sharpe per gamma.
    """
    N = cov_true.shape[0]
    rng = np.random.RandomState(seed)
    sums = {m: np.zeros(len(gamma_grid)) for m in ('A1L1', 'A3')}
    counts = {m: np.zeros(len(gamma_grid), dtype=int) for m in ('A1L1', 'A3')}
    for _ in range(n_mc):
        samples = rng.multivariate_normal(mu_true, cov_true, T)
        cov_hat = np.cov(samples.T, bias=False) + ridge * np.eye(N)
        hat_mu = inject_ic_noise(mu_true, ic, rng)
        try:
            tree = build_hrp_tree(cov_hat)
        except Exception:
            continue
        for k, g in enumerate(gamma_grid):
            for mname, mfn in [('A1L1', method_a1_l1_weights),
                               ('A3', method_a3_weights)]:
                try:
                    w = mfn(cov_hat, hat_mu, tree, float(g))
                    if not np.all(np.isfinite(w)):
                        continue
                    sums[mname][k] += oos_sharpe(w, mu_true, cov_true)
                    counts[mname][k] += 1
                except Exception:
                    continue
    return {m: np.where(counts[m] > 0, sums[m] / np.maximum(counts[m], 1),
                        np.nan)
            for m in ('A1L1', 'A3')}


def run_cell(cov_true: np.ndarray, mu_true: np.ndarray, T: int, ic: float,
             gamma_grid: np.ndarray, n_mc: int, ridge: float,
             seed: int, sweeps: int = 100) -> np.ndarray:
    """
    Monte Carlo over (hat_Sigma, hat_mu) ~ sample distribution.
    Returns an array of shape (len(gamma_grid),) with mean OOS Sharpe
    per gamma over the n_mc replications.
    """
    N = cov_true.shape[0]
    rng = np.random.RandomState(seed)
    sums = np.zeros(len(gamma_grid))
    counts = np.zeros(len(gamma_grid), dtype=int)
    for _ in range(n_mc):
        # Draw T samples from N(mu_true, Sigma_true) to build hat_Sigma.
        samples = rng.multivariate_normal(mu_true, cov_true, T)
        cov_hat = np.cov(samples.T, bias=False) + ridge * np.eye(N)
        # Inject IC-calibrated signal noise on top of the true mu.
        hat_mu = inject_ic_noise(mu_true, ic, rng)
        for k, g in enumerate(gamma_grid):
            try:
                w = method_b_solve_exact(cov_hat, hat_mu, float(g))
                if not np.all(np.isfinite(w)):
                    continue
                sums[k] += oos_sharpe(w, mu_true, cov_true)
                counts[k] += 1
            except Exception:
                continue
    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return means


def oracle_sharpe(mu_true: np.ndarray, cov_true: np.ndarray) -> float:
    """Square root of mu' Sigma^{-1} mu."""
    w = np.linalg.solve(cov_true, mu_true)
    return float(np.sqrt(max(mu_true @ w, 0.0)))


def smooth_argmax(xs: np.ndarray, ys: np.ndarray) -> float:
    """Quadratic interpolation around the grid argmax for finer
    resolution than the grid itself. Returns a value in [xs[0], xs[-1]]."""
    k = int(np.nanargmax(ys))
    if k == 0 or k == len(xs) - 1:
        return float(xs[k])
    x0, x1, x2 = xs[k - 1], xs[k], xs[k + 1]
    y0, y1, y2 = ys[k - 1], ys[k], ys[k + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(x1)
    delta = 0.5 * (y0 - y2) / denom
    return float(np.clip(x1 + delta * (x1 - x0), x0, x2))


def curve_informativeness(gammas: np.ndarray, means: np.ndarray) -> float:
    """Relative range of L(gamma) around its maximum. Cells whose mean-
    Sharpe curve is nearly flat in gamma are MC-noise-dominated and
    should get low weight in the global calibration."""
    v = np.array(means, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0.0
    vmax = v.max()
    vmin = v.min()
    scale = max(abs(vmax), 1e-9)
    return float((vmax - vmin) / scale)


def fit_c_global(rows: dict,
                 min_info: float = 0.05) -> tuple[float, float, float]:
    """
    Fit the two-parameter family

        gamma_pred = 1 / (1 + c * NSR ** alpha)

    by non-linear least squares in the linear gamma-space on cells whose
    L(gamma) curve is informative and whose empirical argmax is interior.
    The theoretical derivation in Section 6.5 corresponds to alpha = 1;
    the fitted alpha reports how closely the data supports that.

    Weighted least squares with weights = curve informativeness, so
    flat (MC-noise-dominated) cells contribute little.

    Returns (c_bar, alpha, rmse) where rmse is the un-weighted RMSE on
    the included cells.
    """
    xs, ys, ws = [], [], []
    for r in rows.values():
        g = r["gamma_emp"]
        info = curve_informativeness(r["gamma_grid"], r["means"])
        if info < min_info:
            continue
        if not (0.02 < g < 0.98):
            continue
        xs.append(r["nsr"])
        ys.append(g)
        ws.append(info)
    if not xs:
        return float("nan"), float("nan"), float("nan")
    x = np.array(xs); y = np.array(ys); w = np.array(ws)
    log_nsr = np.log(x)

    def obj(log_c: float, alpha: float) -> float:
        c = np.exp(log_c)
        yhat = 1.0 / (1.0 + c * np.exp(alpha * log_nsr))
        return float(np.sum(w * (y - yhat) ** 2))

    # Two-parameter grid search then polish.
    log_c_grid = np.linspace(-24.0, 2.0, 131)
    alpha_grid = np.linspace(0.00, 1.5, 61)
    best = (np.inf, 0.0, 1.0)
    for alpha in alpha_grid:
        for log_c in log_c_grid:
            v = obj(log_c, alpha)
            if v < best[0]:
                best = (v, log_c, alpha)
    _, log_c_best, alpha_best = best
    # One round of 1-D refinement on log_c holding alpha fixed.
    from math import sqrt
    a = log_c_best - 0.5
    b = log_c_best + 0.5
    phi = (sqrt(5) - 1) / 2
    for _ in range(50):
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        if obj(x1, alpha_best) < obj(x2, alpha_best):
            b = x2
        else:
            a = x1
    log_c_best = 0.5 * (a + b)
    c_bar = float(np.exp(log_c_best))
    yhat = 1.0 / (1.0 + c_bar * np.exp(alpha_best * log_nsr))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return c_bar, float(alpha_best), rmse


# ================================================================
# Experiment 1: calibrate c on a base grid
# ================================================================

def cell_key(regime: str, tn: float, ic: float) -> str:
    return f"{regime}|tn={tn}|ic={ic}"


def experiment_1(N: int, n_mc: int, seed: int) -> dict:
    """
    Base calibration grid: 4 regimes x 4 T/N x 3 IC = 48 cells.
    Returns a dict keyed by cell_key with all measured quantities.
    """
    TN_grid = [0.6, 1.0, 2.0, 5.0]
    IC_grid = [0.02, 0.05, 0.10]
    gamma_grid = np.linspace(0.0, 1.0, 51)

    rows = {}
    print(f"[Exp1] {len(REGIMES)} regimes x {len(TN_grid)} T/N x "
          f"{len(IC_grid)} IC = {len(REGIMES)*len(TN_grid)*len(IC_grid)} "
          f"cells, n_mc={n_mc}")
    t_start = time.time()
    for r_idx, regime in enumerate(REGIMES):
        cov = regime_factory(regime)(N, seed)
        kc = kappa_corr(cov)
        rng_mu = np.random.RandomState(seed + 1000 + r_idx)
        mu_true = rng_mu.randn(N) * 0.02
        sr_oracle = oracle_sharpe(mu_true, cov)
        for tn_idx, tn in enumerate(TN_grid):
            T = max(int(round(tn * N)), 10)
            for ic_idx, ic in enumerate(IC_grid):
                t0 = time.time()
                # Deterministic seed per cell (avoids hash() randomisation).
                cell_seed = seed + r_idx * 10000 + tn_idx * 100 + ic_idx
                means = run_cell(cov, mu_true, T, ic, gamma_grid,
                                 n_mc=n_mc, ridge=1e-4,
                                 seed=cell_seed)
                gamma_emp = smooth_argmax(gamma_grid, means)
                sr_emp = float(np.nanmax(means))
                nsr = (kc ** 2) * (N / T) / (ic * ic)
                c_cell = (1.0 / max(gamma_emp, 1e-6) - 1.0) / nsr if gamma_emp > 1e-3 else np.nan
                # Tree-based methods
                tree_means = run_cell_tree(cov, mu_true, T, ic,
                                           gamma_grid, n_mc=n_mc,
                                           ridge=1e-4, seed=cell_seed)
                gamma_a1l1 = smooth_argmax(gamma_grid, tree_means['A1L1'])
                gamma_a3 = smooth_argmax(gamma_grid, tree_means['A3'])
                sr_a1l1 = float(np.nanmax(tree_means['A1L1']))
                sr_a3 = float(np.nanmax(tree_means['A3']))
                row = dict(regime=regime, N=N, T=T, tn=tn, ic=ic,
                           kappa_C=kc, sr_oracle=sr_oracle,
                           gamma_emp=gamma_emp, sr_emp=sr_emp,
                           nsr=nsr, c_cell=c_cell,
                           gamma_a1l1=gamma_a1l1, sr_a1l1=sr_a1l1,
                           gamma_a3=gamma_a3, sr_a3=sr_a3,
                           means=means.tolist(),
                           means_a1l1=tree_means['A1L1'].tolist(),
                           means_a3=tree_means['A3'].tolist(),
                           gamma_grid=gamma_grid.tolist())
                rows[cell_key(regime, tn, ic)] = row
                dt = time.time() - t0
                print(f"  {regime:8s}  T/N={tn:<4}  IC={ic:<5}  "
                      f"kappa(C)={kc:9.2f}  "
                      f"g*B={gamma_emp:.2f} g*A1L1={gamma_a1l1:.2f} "
                      f"g*A3={gamma_a3:.2f}  "
                      f"SR: B={sr_emp:.3f} A1L1={sr_a1l1:.3f} A3={sr_a3:.3f}  "
                      f"[{dt:4.1f}s]")
    print(f"[Exp1] done in {time.time()-t_start:.1f}s")
    return rows


# ================================================================
# Experiment 2: validate on a finer grid
# ================================================================

def experiment_2(N: int, n_mc: int, seed: int, c_bar: float,
                 alpha: float = 1.0) -> dict:
    """
    Expanded grid for out-of-sample validation of the adaptive formula.
    4 regimes x 9 T/N x 3 IC = 108 cells.
    """
    TN_grid = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    IC_grid = [0.02, 0.05, 0.10]
    gamma_grid = np.linspace(0.0, 1.0, 51)
    rows = {}
    n_cells = len(REGIMES) * len(TN_grid) * len(IC_grid)
    print(f"[Exp2] {n_cells} cells, n_mc={n_mc}")
    t_start = time.time()
    i = 0
    for r_idx, regime in enumerate(REGIMES):
        cov = regime_factory(regime)(N, seed)
        kc = kappa_corr(cov)
        rng_mu = np.random.RandomState(seed + 1000 + r_idx)
        mu_true = rng_mu.randn(N) * 0.02
        for tn_idx, tn in enumerate(TN_grid):
            T = max(int(round(tn * N)), 10)
            for ic_idx, ic in enumerate(IC_grid):
                i += 1
                t0 = time.time()
                cell_seed = seed + 50000 + r_idx * 10000 + tn_idx * 100 + ic_idx
                means = run_cell(cov, mu_true, T, ic, gamma_grid,
                                 n_mc=n_mc, ridge=1e-4,
                                 seed=cell_seed)
                gamma_emp = smooth_argmax(gamma_grid, means)
                sr_emp = float(np.nanmax(means))
                nsr = (kc ** 2) * (N / T) / (ic * ic)
                gamma_pred = 1.0 / (1.0 + c_bar * (nsr ** alpha))
                tree_means = run_cell_tree(cov, mu_true, T, ic,
                                           gamma_grid, n_mc=n_mc,
                                           ridge=1e-4, seed=cell_seed)
                gamma_a1l1 = smooth_argmax(gamma_grid, tree_means['A1L1'])
                gamma_a3 = smooth_argmax(gamma_grid, tree_means['A3'])
                sr_a1l1 = float(np.nanmax(tree_means['A1L1']))
                sr_a3 = float(np.nanmax(tree_means['A3']))
                row = dict(regime=regime, N=N, T=T, tn=tn, ic=ic,
                           kappa_C=kc,
                           gamma_emp=gamma_emp, gamma_pred=gamma_pred,
                           sr_emp=sr_emp, nsr=nsr,
                           gamma_a1l1=gamma_a1l1, sr_a1l1=sr_a1l1,
                           gamma_a3=gamma_a3, sr_a3=sr_a3,
                           means=means.tolist(),
                           means_a1l1=tree_means['A1L1'].tolist(),
                           means_a3=tree_means['A3'].tolist(),
                           gamma_grid=gamma_grid.tolist())
                rows[cell_key(regime, tn, ic)] = row
                dt = time.time() - t0
                if i % 10 == 0 or i == 1 or i == n_cells:
                    print(f"  [{i:3d}/{n_cells}] {regime:8s}  T/N={tn:<4}  "
                          f"IC={ic:<5}  g*B={gamma_emp:.2f}  "
                          f"g*pred={gamma_pred:.2f}  "
                          f"SR: B={sr_emp:.3f} A1L1={sr_a1l1:.3f}  "
                          f"[{dt:4.1f}s]")
    print(f"[Exp2] done in {time.time()-t_start:.1f}s")
    return rows


# ================================================================
# Experiment 3: spectral-shape sensitivity
# ================================================================

def experiment_3(N: int, n_mc: int, seed: int) -> dict:
    """Hold kappa(C) approximately constant and vary the number of spikes.
    Returns rows keyed by sub-regime name."""
    print(f"[Exp3] 4 spike configurations, n_mc={n_mc}")
    tn = 2.0
    ic = 0.05
    T = int(round(tn * N))
    gamma_grid = np.linspace(0.0, 1.0, 51)
    target_kappa = 30.0
    configs = [
        dict(name="1 spike",    n_spikes=1,  spike_mag=target_kappa - 1),
        dict(name="5 spikes",   n_spikes=5,  spike_mag=(target_kappa - 1) / 5),
        dict(name="20 spikes",  n_spikes=20, spike_mag=(target_kappa - 1) / 20),
        dict(name="MP decay",   n_spikes=0,  spike_mag=1.0 / target_kappa,
             mp_shape=True),
    ]
    rows = {}
    rng_mu = np.random.RandomState(seed + 9999)
    mu_true = rng_mu.randn(N) * 0.02
    for cfg_idx, cfg in enumerate(configs):
        mp_shape = cfg.get("mp_shape", False)
        cov = make_spiked(N, n_spikes=cfg["n_spikes"],
                          spike_mag=cfg["spike_mag"],
                          seed=seed, mp_shape=mp_shape)
        kc = kappa_corr(cov)
        t0 = time.time()
        cell_seed = seed + 90000 + cfg_idx
        means = run_cell(cov, mu_true, T, ic, gamma_grid,
                         n_mc=n_mc, ridge=1e-4,
                         seed=cell_seed)
        gamma_emp = smooth_argmax(gamma_grid, means)
        nsr = (kc ** 2) * (N / T) / (ic * ic)
        c_cell = (1.0 / max(gamma_emp, 1e-6) - 1.0) / nsr if gamma_emp > 1e-3 else np.nan
        # Simple spectral summary: effective rank of the correlation matrix.
        d = np.sqrt(np.diag(cov))
        C = cov / np.outer(d, d)
        lam = np.linalg.eigvalsh(C)
        lam = np.clip(lam, 1e-12, None)
        p = lam / lam.sum()
        erank = float(np.exp(-(p * np.log(p)).sum()))
        rows[cfg["name"]] = dict(name=cfg["name"], kappa_C=kc, erank=erank,
                                 gamma_emp=gamma_emp, c_cell=c_cell,
                                 means=means.tolist())
        dt = time.time() - t0
        print(f"  {cfg['name']:12s}  kappa(C)={kc:7.2f}  erank={erank:6.1f}  "
              f"gamma*_emp={gamma_emp:.2f}  c={c_cell:6.2f}  [{dt:4.1f}s]")
    return rows


# ================================================================
# Dumping / reporting
# ================================================================

def write_exp1_csv(rows: dict) -> None:
    path = os.path.join(RESDIR, "exp1_calibration.csv")
    cols = ["regime", "N", "T", "tn", "ic", "kappa_C", "sr_oracle",
            "gamma_emp", "sr_emp", "nsr", "c_cell",
            "gamma_a1l1", "sr_a1l1", "gamma_a3", "sr_a3"]
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key in sorted(rows.keys()):
            row = rows[key]
            w.writerow({k: row[k] for k in cols})
    print(f"[write] {path}")


def write_exp2_csv(rows: dict) -> None:
    path = os.path.join(RESDIR, "exp2_validation.csv")
    cols = ["regime", "N", "T", "tn", "ic", "kappa_C", "gamma_emp",
            "gamma_pred", "sr_emp", "nsr",
            "gamma_a1l1", "sr_a1l1", "gamma_a3", "sr_a3"]
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key in sorted(rows.keys()):
            row = rows[key]
            w.writerow({k: row[k] for k in cols})
    print(f"[write] {path}")


def write_exp3_csv(rows: dict) -> None:
    path = os.path.join(RESDIR, "exp3_spectral.csv")
    cols = ["name", "kappa_C", "erank", "gamma_emp", "c_cell"]
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in rows:
            row = rows[k]
            w.writerow({c: row[c] for c in cols})
    print(f"[write] {path}")


def write_lgamma_curves(exp2_rows: dict, cells: list[str]) -> None:
    """Save the three representative L(gamma) arrays for fig06."""
    out = {}
    for cell in cells:
        row = exp2_rows[cell]
        tag = cell.replace("|", "__")
        out[f"{tag}__gamma"] = np.array(row["gamma_grid"])
        out[f"{tag}__means"] = np.array(row["means"])
        out[f"{tag}__kappa_C"] = float(row["kappa_C"])
        out[f"{tag}__tn"] = float(row["tn"])
        out[f"{tag}__ic"] = float(row["ic"])
        out[f"{tag}__gamma_pred"] = float(row["gamma_pred"])
        out[f"{tag}__regime"] = row["regime"]
    path = os.path.join(RESDIR, "lgamma_curves.npz")
    np.savez(path, **out)
    print(f"[write] {path}")


def summary_stats(rows1: dict, rows2: dict, rows3: dict,
                  c_bar: float, alpha: float, rmse_cal: float) -> str:
    c1 = np.array([r["c_cell"] for r in rows1.values()
                   if np.isfinite(r["c_cell"])])
    rmse = np.sqrt(np.mean([(r["gamma_pred"] - r["gamma_emp"]) ** 2
                            for r in rows2.values()]))
    mae = float(np.mean([abs(r["gamma_pred"] - r["gamma_emp"])
                         for r in rows2.values()]))
    y = np.array([r["gamma_emp"] for r in rows2.values()])
    yhat = np.array([r["gamma_pred"] for r in rows2.values()])
    if np.var(y) > 0:
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")
    pearson = float(np.corrcoef(y, yhat)[0, 1]) if np.var(y) > 0 else float("nan")
    c3 = np.array([r["c_cell"] for r in rows3.values()
                   if np.isfinite(r["c_cell"])])
    spread3 = (c3.max() / c3.min()) if len(c3) > 0 else float("nan")
    lines = [
        "Adaptive gamma* rule: numerical validation summary",
        "=" * 60,
        "",
        f"Calibrated two-parameter fit: gamma* = 1 / (1 + c * NSR^alpha)",
        f"  c_bar    = {c_bar:.3e}",
        f"  alpha    = {alpha:.3f}",
        f"  RMSE(cal)= {rmse_cal:.3f}",
        "",
        f"Exp 1 -- calibration ({len(rows1)} cells)",
        f"  min(c_cell) / max(c_cell) = {c1.min():.3e} / {c1.max():.3e}",
        "",
        f"Exp 2 -- validation  ({len(rows2)} cells)",
        f"  Pearson r (emp vs pred) = {pearson:.3f}",
        f"  R^2       (emp vs pred) = {r2:.3f}",
        f"  MAE       (emp vs pred) = {mae:.3f}",
        f"  RMSE      (emp vs pred) = {rmse:.3f}",
        "",
        f"Exp 3 -- spectral sensitivity",
        f"  c spread across spike configs = {spread3:.2f}x",
        f"  c_cells: {[f'{c:.3e}' for c in c3.tolist()]}",
        "",
    ]
    out = "\n".join(lines)
    with open(os.path.join(RESDIR, "summary.txt"), "w") as fh:
        fh.write(out)
    return out


# ================================================================
# Main
# ================================================================

def main(N: int = 100, n_mc_exp1: int = 500, n_mc_exp2: int = 300,
         n_mc_exp3: int = 500, seed: int = 42) -> None:
    print(f"compute09_adaptive_gamma: N={N} seed={seed}")
    t_all = time.time()

    rows1 = experiment_1(N, n_mc=n_mc_exp1, seed=seed)
    c_bar, alpha, rmse_cal = fit_c_global(rows1)
    print(f"[calib] 2-param fit: c_bar = {c_bar:.3e}, "
          f"alpha = {alpha:.3f}, RMSE (calibration) = {rmse_cal:.3f}")
    write_exp1_csv(rows1)

    rows2 = experiment_2(N, n_mc=n_mc_exp2, seed=seed, c_bar=c_bar,
                         alpha=alpha)
    write_exp2_csv(rows2)

    rows3 = experiment_3(N, n_mc=n_mc_exp3, seed=seed)
    write_exp3_csv(rows3)

    # Pick three representative cells for fig06: low, medium, high NSR.
    pool = [(k, r["nsr"]) for k, r in rows2.items()]
    pool.sort(key=lambda kv: kv[1])
    low_cell = pool[len(pool) // 10][0]       # 10th percentile
    mid_cell = pool[len(pool) // 2][0]        # median
    high_cell = pool[int(len(pool) * 0.9)][0] # 90th percentile
    write_lgamma_curves(rows2, [low_cell, mid_cell, high_cell])

    summary = summary_stats(rows1, rows2, rows3, c_bar, alpha, rmse_cal)
    print()
    print(summary)
    print(f"[main] total elapsed {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
