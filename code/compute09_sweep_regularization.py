"""
compute09_sweep_regularization.py
=================================

OOS Experiment 7b (three_channels.md Phase A):

Sweep BOTH gamma and the sweep count p over a (gamma, p) grid on the
same synthetic Monte Carlo design used in Section 9, isolating the
role of early stopping as a statistical regularizer separately from
operator shrinkage.

For each cell (regime, T/N, signal-type), draw n_mc Monte Carlo
replications of (hat_Sigma, hat_mu). For each gamma in the grid run
one Gauss-Seidel chain on P_gamma w = hat_mu initialised at
w^(0) = hat_mu / diag(hat_Sigma), checkpointing the iterate at each
p in the checkpoint grid. At each (gamma, p) checkpoint evaluate the
closed-form OOS Sharpe under the population (Sigma, mu) and record
the mean across the MC replications.

The artifacts produced drive the §9 sweep-regularization experiment
(OOS Sharpe vs p at fixed gamma slices) and the §9 (gamma, p)
heatmap. They also provide the numbers against which the 8
predictions in three_channels.md §A.4 are checked.

Artifacts (under results/sec09_sweep/)
--------------------------------------
    sweep_grid.npz           (n_regimes, n_tn, n_sig, n_gamma, n_p)
                             mean OOS Sharpe surface + auxiliaries
    convergence_table.csv    mean sweeps to residual < 1e-6 per gamma
    predictions.json         which of the 8 predictions held

Usage
-----
    python figures/code/compute09_sweep_regularization.py
"""

from __future__ import annotations

import os
import json
import csv
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESDIR = os.path.join(ROOT, "results", "sec09_sweep")
os.makedirs(RESDIR, exist_ok=True)


# ================================================================
# Regimes (reuse from compute09_adaptive_gamma.py)
# ================================================================

from compute09_adaptive_gamma import (   # noqa: E402
    make_factor, make_block, make_spiked, make_equicorr,
    kappa_corr, inject_ic_noise, oos_sharpe, oracle_sharpe,
)


REGIMES = {
    "factor":   lambda N, seed: make_factor(N, K=3, seed=seed),
    "block":    lambda N, seed: make_block(N, n_sectors=5, rho_w=0.6,
                                           rho_c=0.15, seed=seed),
    "spiked":   lambda N, seed: make_spiked(N, n_spikes=5, spike_mag=3.0,
                                            seed=seed),
    "equicorr": lambda N, seed: make_equicorr(N, rho=0.6, seed=seed),
}


# ================================================================
# Method B with checkpointing
# ================================================================

def method_b_checkpoints(cov: np.ndarray, mu: np.ndarray, gamma: float,
                         p_grid: list[int],
                         resid_grid: np.ndarray | None = None
                         ) -> tuple[np.ndarray, int]:
    """
    Run scalar Gauss-Seidel on P_gamma w = mu and checkpoint w at every
    p in p_grid. Initial guess is the diagonal solution mu / diag(cov).

    Parameters
    ----------
    cov       : (N, N) SPD covariance
    mu        : (N,)
    gamma     : shrinkage level
    p_grid    : list of monotone non-negative ints; 0 means "initial"
    resid_grid: (len(p_grid),) output array for residual at each checkpoint,
                or None to skip

    Returns
    -------
    W : (len(p_grid), N) array of weight vectors at each checkpoint
    n_conv : number of sweeps at which ||r||/||mu|| dropped below 1e-10
             (or p_grid[-1] + 1 if never reached)
    """
    N = len(mu)
    d = np.diag(cov)
    p_grid = list(p_grid)
    max_p = max(p_grid) if p_grid else 0
    W = np.zeros((len(p_grid), N))

    if gamma < 1e-14:
        w = mu / d
        for k in range(len(p_grid)):
            W[k] = w
        return W, 1

    w = mu / d
    n_conv = max_p + 1
    mu_norm = max(np.linalg.norm(mu), 1e-12)

    # Write the p=0 checkpoint if requested.
    for k, pk in enumerate(p_grid):
        if pk == 0:
            W[k] = w

    for sweep in range(1, max_p + 1):
        for i in range(N):
            off = cov[i, :] @ w - cov[i, i] * w[i]
            w[i] = (mu[i] - gamma * off) / d[i]
        # Residual on P_gamma w = mu.
        if n_conv > max_p:
            Pw = (1.0 - gamma) * d * w + gamma * (cov @ w)
            if np.linalg.norm(Pw - mu) / mu_norm < 1e-10:
                n_conv = sweep
        for k, pk in enumerate(p_grid):
            if pk == sweep:
                W[k] = w.copy()
    return W, n_conv


# ================================================================
# Experiment driver
# ================================================================

P_GRID = [1, 2, 5, 10, 25, 50, 100, 200, 500]
GAMMA_GRID = np.linspace(0.0, 1.0, 11)
TN_GRID = [0.6, 1.0, 2.0, 5.0]
SIGNAL_KINDS = ["oracle", "noisy"]
N_DEFAULT = 100
N_MC_DEFAULT = 200


def run_cell(cov_true: np.ndarray, mu_true: np.ndarray, T: int,
             signal_kind: str, n_mc: int, seed: int,
             ridge: float = 1e-4,
             ic_for_noisy: float = 0.05
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo over (hat_Sigma, hat_mu) at this cell's parameters.

    Returns
    -------
    sharpe    : (n_gamma, n_p) mean OOS Sharpe across MC reps
    direrr    : (n_gamma, n_p) mean in-sample direction error vs Sigma^-1 mu
    n_conv_mean : (n_gamma,) mean convergence sweep count per gamma
    """
    N = cov_true.shape[0]
    rng = np.random.RandomState(seed)
    n_gamma = len(GAMMA_GRID)
    n_p = len(P_GRID)
    sharpe = np.zeros((n_gamma, n_p))
    direrr = np.zeros((n_gamma, n_p))
    counts = np.zeros((n_gamma, n_p), dtype=int)
    n_conv_accum = np.zeros(n_gamma)

    for _ in range(n_mc):
        samples = rng.multivariate_normal(mu_true, cov_true, T)
        cov_hat = np.cov(samples.T, bias=False) + ridge * np.eye(N)
        if signal_kind == "oracle":
            hat_mu = mu_true.copy()
        else:
            hat_mu = inject_ic_noise(mu_true, ic_for_noisy, rng)

        # In-sample reference for direction error.
        try:
            ref = np.linalg.solve(cov_hat, hat_mu)
            ref_norm = np.linalg.norm(ref) + 1e-20
        except Exception:
            ref = hat_mu / np.diag(cov_hat)
            ref_norm = np.linalg.norm(ref) + 1e-20

        for gi, g in enumerate(GAMMA_GRID):
            W, n_conv = method_b_checkpoints(cov_hat, hat_mu, float(g),
                                             P_GRID)
            n_conv_accum[gi] += min(n_conv, P_GRID[-1])
            for pk in range(n_p):
                w = W[pk]
                if not np.all(np.isfinite(w)):
                    continue
                sharpe[gi, pk] += oos_sharpe(w, mu_true, cov_true)
                wn = w / (np.linalg.norm(w) + 1e-20)
                rn = ref / ref_norm
                cos = float(wn @ rn)
                direrr[gi, pk] += 1.0 - abs(cos)
                counts[gi, pk] += 1

    mean_sharpe = np.where(counts > 0, sharpe / np.maximum(counts, 1), np.nan)
    mean_direrr = np.where(counts > 0, direrr / np.maximum(counts, 1), np.nan)
    n_conv_mean = n_conv_accum / max(n_mc, 1)
    return mean_sharpe, mean_direrr, n_conv_mean


def run_all(N: int = N_DEFAULT, n_mc: int = N_MC_DEFAULT,
            seed: int = 42) -> dict:
    print(f"compute09_sweep_regularization: N={N} n_mc={n_mc}")
    regime_names = list(REGIMES.keys())

    n_reg = len(regime_names)
    n_tn = len(TN_GRID)
    n_sig = len(SIGNAL_KINDS)
    n_g = len(GAMMA_GRID)
    n_p = len(P_GRID)

    sharpe_tensor = np.full((n_reg, n_tn, n_sig, n_g, n_p), np.nan)
    dir_tensor = np.full((n_reg, n_tn, n_sig, n_g, n_p), np.nan)
    conv_tensor = np.full((n_reg, n_tn, n_sig, n_g), np.nan)
    kappa_C = np.zeros(n_reg)

    t_start = time.time()
    for ri, reg in enumerate(regime_names):
        cov = REGIMES[reg](N, seed)
        kappa_C[ri] = kappa_corr(cov)
        rng_mu = np.random.RandomState(seed + 1000 + ri)
        mu_true = rng_mu.randn(N) * 0.02
        for ti, tn in enumerate(TN_GRID):
            T = max(int(round(tn * N)), 10)
            for si, sig in enumerate(SIGNAL_KINDS):
                t0 = time.time()
                sh, de, nc = run_cell(
                    cov, mu_true, T, sig, n_mc=n_mc,
                    seed=seed + hash((reg, tn, sig)) % 100000
                )
                sharpe_tensor[ri, ti, si] = sh
                dir_tensor[ri, ti, si] = de
                conv_tensor[ri, ti, si] = nc
                dt = time.time() - t0
                best_sh = float(np.nanmax(sh))
                best_ij = np.unravel_index(np.nanargmax(sh), sh.shape)
                print(f"  {reg:8s} T/N={tn:<4} {sig:6s}  "
                      f"max Sharpe={best_sh:+.3f} at "
                      f"(gamma={GAMMA_GRID[best_ij[0]]:.1f}, "
                      f"p={P_GRID[best_ij[1]]})  [{dt:5.1f}s]")
    dt_tot = time.time() - t_start
    print(f"[run_all] total {dt_tot:.1f}s")

    return dict(
        sharpe=sharpe_tensor,
        dir_err=dir_tensor,
        conv=conv_tensor,
        kappa_C=kappa_C,
        regime_names=np.array(regime_names),
        tn_grid=np.array(TN_GRID),
        sig_kinds=np.array(SIGNAL_KINDS),
        gamma_grid=GAMMA_GRID,
        p_grid=np.array(P_GRID),
    )


# ================================================================
# Predictions
# ================================================================

def check_predictions(res: dict) -> dict:
    """Evaluate the 8 predictions from three_channels.md §A.4."""
    regime_names = list(res["regime_names"])
    sharpe = res["sharpe"]  # (reg, tn, sig, gamma, p)
    p_grid = res["p_grid"].tolist()
    gamma_grid = res["gamma_grid"].tolist()
    tn_grid = res["tn_grid"].tolist()
    sig_idx = {k: i for i, k in enumerate(res["sig_kinds"])}

    def idx_sig(s): return sig_idx[s]
    def idx_p(p): return p_grid.index(p)
    def idx_g(g): return int(np.argmin(np.abs(np.array(gamma_grid) - g)))
    def idx_tn(t): return tn_grid.index(t)
    def idx_reg(r): return regime_names.index(r)

    out = {}

    # Prediction 1: Flat curve at gamma = 0.3
    # std over p >= 5 across regimes and T/N (noisy), at gamma=0.3
    gi = idx_g(0.3); pi_lo = idx_p(5)
    stds = []
    for ri in range(len(regime_names)):
        for ti in range(len(tn_grid)):
            vec = sharpe[ri, ti, idx_sig("noisy"), gi, pi_lo:]
            vec = vec[np.isfinite(vec)]
            if len(vec):
                stds.append(float(np.std(vec)))
    out["P1_flat_at_gamma_03"] = {
        "max_std_over_p": float(max(stds)) if stds else None,
        "threshold": 0.02,
        "passed": (max(stds) < 0.02) if stds else None,
    }

    # Prediction 2: Plateau at gamma = 0.5 by p ~= 50
    gi = idx_g(0.5); pi_50 = idx_p(50); pi_conv = len(p_grid) - 1
    diffs = []
    for ri in range(len(regime_names)):
        for ti in range(len(tn_grid)):
            vec = sharpe[ri, ti, idx_sig("noisy"), gi]
            if np.isfinite(vec[pi_50]) and np.isfinite(vec[pi_conv]):
                diffs.append(float(abs(vec[pi_50] - vec[pi_conv])))
    out["P2_plateau_at_gamma_05"] = {
        "max_diff": max(diffs) if diffs else None,
        "threshold": 0.02,
        "passed": (max(diffs) < 0.02) if diffs else None,
    }

    # Prediction 3: Interior p* at gamma = 1
    gi = idx_g(1.0)
    gaps = []
    for ri in range(len(regime_names)):
        for ti in range(len(tn_grid)):
            vec = sharpe[ri, ti, idx_sig("noisy"), gi]
            vec = vec[np.isfinite(vec)]
            if len(vec) < 2:
                continue
            best = float(np.max(vec))
            last = float(vec[-1])
            gaps.append(best - last)
    out["P3_interior_p_at_gamma_1"] = {
        "max_gap": max(gaps) if gaps else None,
        "threshold": 0.05,
        "passed": (max(gaps) > 0.05) if gaps else None,
        "gaps": [round(g, 3) for g in gaps],
    }

    # Prediction 4: p*(gamma=1) shifts with T/N
    gi = idx_g(1.0)
    p_stars = {}
    for ti, tn in enumerate(tn_grid):
        best_ps = []
        for ri in range(len(regime_names)):
            vec = sharpe[ri, ti, idx_sig("noisy"), gi]
            if np.all(np.isnan(vec)):
                continue
            k = int(np.nanargmax(vec))
            best_ps.append(p_grid[k])
        if best_ps:
            p_stars[tn] = float(np.median(best_ps))
    tn_sorted = sorted(p_stars.keys())
    monotonic = all(p_stars[tn_sorted[i]] <= p_stars[tn_sorted[i + 1]]
                    for i in range(len(tn_sorted) - 1))
    out["P4_p_star_monotone_in_tn"] = {
        "p_stars": {str(k): v for k, v in p_stars.items()},
        "monotone_nondecreasing": monotonic,
        "passed": monotonic,
    }

    # Prediction 5: Ridge in (gamma, p)
    # Check that (gamma=1, p*) Sharpe is within 0.05 of (gamma=0.5, p=100) Sharpe
    # averaged across cells.
    gi_half = idx_g(0.5); gi_one = idx_g(1.0); pi_100 = idx_p(100)
    within = 0
    total = 0
    for ri in range(len(regime_names)):
        for ti in range(len(tn_grid)):
            a = sharpe[ri, ti, idx_sig("noisy"), gi_half, pi_100]
            v = sharpe[ri, ti, idx_sig("noisy"), gi_one]
            if not np.isfinite(a) or np.all(np.isnan(v)):
                continue
            total += 1
            if np.nanmax(v) >= a - 0.05:
                within += 1
    out["P5_ridge_substitutability"] = {
        "cells_within_005": within,
        "total": total,
        "passed": (total > 0 and within / total > 0.5),
    }

    # Prediction 6: Oracle mu reduces the p* decline at gamma=1
    gi = idx_g(1.0)
    gaps_noisy = []
    gaps_oracle = []
    for ri in range(len(regime_names)):
        for ti in range(len(tn_grid)):
            vn = sharpe[ri, ti, idx_sig("noisy"), gi]
            vo = sharpe[ri, ti, idx_sig("oracle"), gi]
            if np.all(np.isnan(vn)) or np.all(np.isnan(vo)):
                continue
            gn = np.nanmax(vn) - vn[-1]
            go = np.nanmax(vo) - vo[-1]
            gaps_noisy.append(gn)
            gaps_oracle.append(go)
    out["P6_oracle_mu_reduces_p_star"] = {
        "mean_gap_noisy": float(np.mean(gaps_noisy)) if gaps_noisy else None,
        "mean_gap_oracle": float(np.mean(gaps_oracle)) if gaps_oracle else None,
        "passed": (float(np.mean(gaps_oracle)) <
                   float(np.mean(gaps_noisy))) if gaps_oracle else None,
    }

    # Prediction 7: (gamma=0.5, p=100) is near global optimum
    gi = idx_g(0.5); pi = idx_p(100)
    diffs = []
    for ri in range(len(regime_names)):
        for ti in range(len(tn_grid)):
            a = sharpe[ri, ti, idx_sig("noisy"), gi, pi]
            tens = sharpe[ri, ti, idx_sig("noisy")]
            if not np.isfinite(a) or np.all(np.isnan(tens)):
                continue
            diffs.append(float(np.nanmax(tens) - a))
    out["P7_gamma_05_p_100_near_global_opt"] = {
        "max_gap": max(diffs) if diffs else None,
        "mean_gap": float(np.mean(diffs)) if diffs else None,
        "threshold": 0.05,
        "passed": (max(diffs) < 0.05) if diffs else None,
    }

    # Prediction 8: decline at gamma=1 is worst at spiked T/N=0.6
    ri_sp = idx_reg("spiked"); ti_low = idx_tn(0.6); gi = idx_g(1.0)
    vec = sharpe[ri_sp, ti_low, idx_sig("noisy"), gi]
    gap_sp06 = float(np.nanmax(vec) - vec[-1]) if not np.all(np.isnan(vec)) else None
    ri_fa = idx_reg("factor"); ti_hi = idx_tn(5.0)
    vec2 = sharpe[ri_fa, ti_hi, idx_sig("noisy"), gi]
    gap_fa50 = float(np.nanmax(vec2) - vec2[-1]) if not np.all(np.isnan(vec2)) else None
    out["P8_overfit_worse_spiked_lowTN"] = {
        "spiked_TN_06_gap": gap_sp06,
        "factor_TN_5_gap": gap_fa50,
        "passed": (gap_sp06 is not None and gap_fa50 is not None
                   and gap_sp06 > gap_fa50),
    }

    return out


# ================================================================
# Main
# ================================================================

def main():
    res = run_all()
    # Save tensor.
    np.savez(os.path.join(RESDIR, "sweep_grid.npz"), **res)
    print(f"[write] {os.path.join(RESDIR, 'sweep_grid.npz')}")

    # Convergence-mean table.
    conv = res["conv"]  # (reg, tn, sig, gamma)
    conv_per_gamma = np.nanmean(conv, axis=(0, 1, 2))
    conv_path = os.path.join(RESDIR, "convergence_table.csv")
    with open(conv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["gamma", "mean_sweeps_to_1e-6", "p100_converged"])
        for gi, g in enumerate(GAMMA_GRID):
            ns = float(conv_per_gamma[gi])
            wr.writerow([f"{g:.1f}", f"{ns:.1f}",
                         "yes" if ns <= 100 else "no"])
    print(f"[write] {conv_path}")

    # Predictions.
    preds = check_predictions(res)
    pred_path = os.path.join(RESDIR, "predictions.json")
    with open(pred_path, "w") as fh:
        json.dump(preds, fh, indent=2, default=str)
    print(f"[write] {pred_path}")

    print("\nPrediction summary:")
    for name, val in preds.items():
        ok = val.get("passed")
        mark = "OK " if ok else ("NO " if ok is False else "?? ")
        print(f"  {mark} {name}: {val}")


if __name__ == "__main__":
    main()
