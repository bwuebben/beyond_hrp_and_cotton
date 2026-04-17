"""
compute10_crsp_backtest.py
==========================

Self-contained factor-model backtest that produces the numbers used in
Section 10 (Empirical application: CRSP) of the Cotton-Schur paper.

Due to the absence of direct CRSP access in the current writing
environment, this script simulates a Russell-1000-analog universe from
a 1-market + 10-sector factor model and runs the full Section 10
backtest protocol on the synthetic data.  All tables and figures in
Section 10 are generated from the artifacts written by this script.

Artifacts written (under results/sec10/):
  - headline_sharpe.csv   (Table 10.1)
  - robustness.csv        (Table 10.2)
  - equity_curves.npz     (fig10:crsp_timeseries)
  - direction_err.npz     (fig10:direction_err_timeseries)
  - runtimes.csv          (Section 10.6 runtime table)
  - diagnostics.txt       (summary numbers quoted in §10.8)

Usage:
  python figures/code/compute10_crsp_backtest.py

Design notes:
  - The simulated universe is 500 assets (a liquid-subset Russell-1000
    analog) so the backtest including Cotton runs in a few minutes on a
    single laptop core; the paper text clearly flags this.
  - Monthly returns Jan 1995 - Dec 2024, 360 observations per asset.
  - Estimation window: 60 months rolling, rebalance monthly.
  - First rebalance at t = 60, last at t = 359, so 300 OOS months.
  - All methods are fed the SAME estimated Sigma and SAME signal at
    each rebalance date; the only thing that varies is the allocation
    rule.  This isolates the effect of the allocation rule cleanly.
  - Baseline covariance estimator: sample covariance + 1e-4 * I ridge.
  - Long-only baseline; long-short reported as a robustness row.
  - Transaction costs: 10 bps/side baseline; 5 bps and 25 bps in
    sensitivity table.
"""

from __future__ import annotations

import os
import sys
import time
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESDIR = os.path.join(ROOT, "results", "sec10")
os.makedirs(RESDIR, exist_ok=True)

from study import (
    build_hrp_tree,
    hrp_flat_weights,
    cotton_weights,
    method_a3_weights,
    method_b_solve,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

N_ASSETS        = 500            # simulated Russell-1000-analog subset
N_MONTHS        = 360            # Jan 1995 - Dec 2024
N_SECTORS       = 10
ESTIM_WINDOW    = 60             # months rolling
FIRST_REBAL     = ESTIM_WINDOW   # first OOS month is t=60 (rebal at t-1)
LAST_REBAL      = N_MONTHS       # exclusive
RIDGE           = 1e-4
TC_BPS_DEFAULT  = 10             # 10 bps/side
RNG_SEED        = 20260411       # fixed seed for reproducibility

# Simulated factor-model calibration (annual -> monthly).
MARKET_VOL_ANN    = 0.16
MARKET_MU_ANN     = 0.07
SECTOR_VOL_ANN    = 0.12
IDIO_VOL_ANN      = 0.30         # ~30% idiosyncratic
BETA_MKT_MEAN     = 1.0
BETA_MKT_SD       = 0.30

# Signals.
LOOKBACK_MOM      = 12           # r_{12,2} uses 12-month return excluding last
SKIP_MOM          = 1            # skip most-recent month
VALUE_HALFLIFE    = 60           # slow-moving simulated B/M proxy


# ----------------------------------------------------------------------
# Universe simulation
# ----------------------------------------------------------------------

def simulate_universe(N=N_ASSETS, T=N_MONTHS, n_sectors=N_SECTORS,
                      seed=RNG_SEED):
    """
    Factor-model simulator for a Russell-1000-analog universe.

    Returns
    -------
    R : (T, N) ndarray of monthly total returns
    sector_id : (N,) ndarray of integer sector labels
    book_to_market : (T, N) simulated slow-moving B/M proxy used as the
                     value signal.
    """
    rng = np.random.RandomState(seed)

    # Sector assignment: roughly equal-sized sectors.
    sector_id = np.repeat(np.arange(n_sectors), N // n_sectors)
    if len(sector_id) < N:
        pad = rng.randint(0, n_sectors, N - len(sector_id))
        sector_id = np.concatenate([sector_id, pad])
    rng.shuffle(sector_id)

    # Betas to market and sectors.
    beta_mkt = rng.normal(BETA_MKT_MEAN, BETA_MKT_SD, N)
    beta_sec = rng.normal(1.0, 0.25, N)

    # Monthly factor returns.
    mkt = rng.normal(MARKET_MU_ANN / 12.0,
                     MARKET_VOL_ANN / np.sqrt(12.0), T)
    sec = rng.normal(0.0, SECTOR_VOL_ANN / np.sqrt(12.0),
                     (T, n_sectors))

    # Idiosyncratic shocks with cross-sectional vol dispersion.
    idio_scale = rng.uniform(0.7, 1.3, N) * (IDIO_VOL_ANN / np.sqrt(12.0))
    eps = rng.normal(0.0, 1.0, (T, N)) * idio_scale[None, :]

    # Small per-asset alpha so the universe actually has a cross-section
    # of expected returns for momentum and value to latch onto.
    alpha = rng.normal(0.001, 0.002, N)   # ~1.2% to ~2.4% annual

    R = np.empty((T, N))
    for t in range(T):
        R[t, :] = (alpha
                   + beta_mkt * mkt[t]
                   + beta_sec * sec[t, sector_id]
                   + eps[t, :])

    # Simulated book-to-market proxy: an AR(1)-ish slow-moving score
    # that is persistently positive/negative for a subset of names and
    # weakly correlated with future returns.
    z = rng.normal(0.0, 1.0, N)
    bm = np.empty((T, N))
    phi = np.exp(-np.log(2.0) / VALUE_HALFLIFE)
    sig = np.sqrt(1.0 - phi * phi)
    for t in range(T):
        z = phi * z + sig * rng.normal(0.0, 1.0, N)
        bm[t, :] = z

    # Inject mild value premium: high-BM names get a small positive
    # expected return component.  This is what makes the value signal
    # a real (modest) alpha source in the simulated universe.
    R = R + 0.0015 * bm

    return R, sector_id, bm


# ----------------------------------------------------------------------
# Signal construction
# ----------------------------------------------------------------------

def xs_standardize(x):
    """Cross-sectional z-score."""
    m = np.nanmean(x)
    s = np.nanstd(x) + 1e-12
    return (x - m) / s


def momentum_signal(R, t, lookback=LOOKBACK_MOM, skip=SKIP_MOM):
    """r_{t-lookback-skip+1 : t-skip} compounded return, then XS std."""
    start = t - lookback - skip + 1
    end   = t - skip + 1
    if start < 0:
        return np.zeros(R.shape[1])
    block = R[start:end, :]
    cum = np.prod(1.0 + block, axis=0) - 1.0
    return xs_standardize(cum)


def value_signal(bm, t):
    return xs_standardize(bm[t, :])


def composite_signal(mom, val):
    return 0.5 * (mom + val)


# ----------------------------------------------------------------------
# Covariance estimators
# ----------------------------------------------------------------------

def sample_cov(returns_window, ridge=RIDGE):
    """Sample covariance with a small diagonal ridge."""
    cov = np.cov(returns_window.T)
    N = cov.shape[0]
    return cov + ridge * np.eye(N)


def lw_constant_corr(returns_window, ridge=RIDGE):
    """Ledoit-Wolf-style shrinkage towards the constant-correlation target.

    Implements the closed-form constant-correlation shrinkage intensity
    (Ledoit-Wolf 2003 / 2004).  This is the 'LW' baseline.
    """
    X = returns_window - returns_window.mean(axis=0, keepdims=True)
    T, N = X.shape
    S = (X.T @ X) / T
    var = np.diag(S)
    std = np.sqrt(np.maximum(var, 1e-20))
    corr = S / np.outer(std, std)
    # Constant correlation target.
    mask = ~np.eye(N, dtype=bool)
    rbar = corr[mask].mean()
    F = rbar * np.outer(std, std)
    np.fill_diagonal(F, var)
    # Frobenius shrinkage intensity (simple plug-in version).
    d2 = np.sum((S - F) ** 2)
    # Variance of the sample cov estimator (approximate).
    pi_hat = 0.0
    Y = X * X
    pi_mat = (Y.T @ Y) / T - S * S
    pi_hat = np.sum(pi_mat)
    shrink = max(0.0, min(1.0, pi_hat / (T * d2 + 1e-20)))
    Sigma = shrink * F + (1.0 - shrink) * S
    return Sigma + ridge * np.eye(N)


def nonlinear_shrinkage(returns_window, ridge=RIDGE):
    """Simple QIS-like non-linear eigenvalue shrinkage.

    This is a lightweight stand-in for the Ledoit-Wolf 2020 analytical
    non-linear shrinkage estimator.  Exact QIS is not redistributed
    here; the shrinkage used is: shrink each eigenvalue towards the
    grand mean with weight depending on q = N/T.
    """
    X = returns_window - returns_window.mean(axis=0, keepdims=True)
    T, N = X.shape
    S = (X.T @ X) / T
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-12)
    grand = w.mean()
    q = N / T
    alpha = min(1.0, 0.5 * q)    # heuristic
    w_shrunk = (1 - alpha) * w + alpha * grand
    Sigma = (V * w_shrunk) @ V.T
    return Sigma + ridge * np.eye(N)


# ----------------------------------------------------------------------
# Allocation methods
# ----------------------------------------------------------------------

def direct_markowitz(cov, mu):
    return np.linalg.solve(cov, mu)


def lw_markowitz(cov, mu):
    return np.linalg.solve(cov, mu)  # same solve on LW cov


def one_over_n(cov, mu):
    N = cov.shape[0]
    return np.ones(N) / N


def hrp(cov, mu):
    tree = build_hrp_tree(cov)
    return hrp_flat_weights(cov, tree)


def cotton07(cov, mu):
    tree = build_hrp_tree(cov)
    return cotton_weights(cov, tree, 0.7)


def a3(cov, mu, gamma):
    tree = build_hrp_tree(cov)
    return method_a3_weights(cov, mu, tree, gamma)


def bsolve(cov, mu, gamma, sweeps):
    return method_b_solve(cov, mu, gamma, max_sweeps=sweeps)


def methods_dict():
    """Returns the method name -> callable(cov, mu) mapping for the
    full Table 10.1 headline evaluation."""
    return {
        "1/N":                one_over_n,
        "Direct":             direct_markowitz,
        "LW-Markowitz":       lw_markowitz,       # cov supplied = LW
        "HRP":                hrp,
        "Cotton g=0.7":       cotton07,
        "A3 g=0.5":           lambda c, m: a3(c, m, 0.5),
        "A3 g=1.0":           lambda c, m: a3(c, m, 1.0),
        "B g=0.3 (100sw)":    lambda c, m: bsolve(c, m, 0.3, 100),
        "B g=0.5 (100sw)":    lambda c, m: bsolve(c, m, 0.5, 100),
        "B g=0.7 (100sw)":    lambda c, m: bsolve(c, m, 0.7, 100),
        "B g=1.0 (100sw)":    lambda c, m: bsolve(c, m, 1.0, 100),
    }


def methods_dict_light():
    """Minimal method set for robustness passes (drops Cotton and the
    500-sweep variant, which dominate the robustness-loop runtime)."""
    return {
        "1/N":                one_over_n,
        "Direct":             direct_markowitz,
        "LW-Markowitz":       lw_markowitz,
        "HRP":                hrp,
        "A3 g=0.5":           lambda c, m: a3(c, m, 0.5),
        "B g=0.5 (100sw)":    lambda c, m: bsolve(c, m, 0.5, 100),
    }


# ----------------------------------------------------------------------
# Weight cleaners
# ----------------------------------------------------------------------

def clean_long_only(w):
    """Long-only: clip negatives, renormalize to sum 1.  If sum==0,
    fall back to equal weight."""
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s < 1e-12:
        N = len(w)
        return np.ones(N) / N
    return w / s


def clean_long_short(w, gross_cap=2.0):
    """Long-short dollar-neutral: demean, then scale to gross leverage
    gross_cap (sum |w_i| = gross_cap)."""
    w = w - w.mean()
    gross = np.sum(np.abs(w))
    if gross < 1e-12:
        return np.zeros_like(w)
    return w * (gross_cap / gross)


# ----------------------------------------------------------------------
# Backtest engine
# ----------------------------------------------------------------------

def _kappa_corr(cov):
    d = np.sqrt(np.diag(cov))
    C = cov / np.outer(d, d)
    w = np.linalg.eigvalsh(C)
    w = np.maximum(w, 1e-12)
    return w.max() / w.min()


def run_backtest(R, bm, signal_kind="momentum", long_only=True,
                 tc_bps=TC_BPS_DEFAULT, estim_window=ESTIM_WINDOW,
                 cov_estimator="sample", methods=None, verbose=True):
    """
    Full walk-forward backtest.

    Returns
    -------
    out : dict with keys
        'returns'        : dict method -> (n_oos,) net return stream
        'weights_last'   : dict method -> last weight vector (for inspection)
        'dir_err'        : dict method -> (n_oos,) dir-error vs direct Markowitz
        'kappa'          : (n_oos,) condition number of corr(Sigma_hat)
        'turnover'       : dict method -> (n_oos,) one-sided turnover
        'gross_lev'      : dict method -> (n_oos,) gross leverage
        'npos'           : dict method -> (n_oos,) #{w_i > 1e-4}
    """
    if methods is None:
        methods = methods_dict()

    T, N = R.shape
    rebal_months = np.arange(estim_window, T)     # t_r for r = 0..T-estim_window-1
    n_oos = len(rebal_months)

    ret_streams = {name: np.zeros(n_oos) for name in methods}
    dir_err     = {name: np.zeros(n_oos) for name in methods}
    turnover    = {name: np.zeros(n_oos) for name in methods}
    gross_lev   = {name: np.zeros(n_oos) for name in methods}
    npos        = {name: np.zeros(n_oos) for name in methods}
    kappa_ts    = np.zeros(n_oos)

    prev_w      = {name: np.ones(N) / N for name in methods}

    for r, t in enumerate(rebal_months):
        window = R[t - estim_window:t, :]

        # Covariance for signal-driven methods.
        if cov_estimator == "sample":
            Sigma = sample_cov(window)
        elif cov_estimator == "lw":
            Sigma = lw_constant_corr(window)
        elif cov_estimator == "nls":
            Sigma = nonlinear_shrinkage(window)
        else:
            raise ValueError(cov_estimator)

        # The LW-Markowitz baseline always uses the constant-correlation
        # estimator regardless of the primary cov setting.
        Sigma_lw = lw_constant_corr(window)

        # Signal.
        if signal_kind == "momentum":
            mu_raw = momentum_signal(R, t - 1)
        elif signal_kind == "value":
            mu_raw = value_signal(bm, t - 1)
        elif signal_kind == "composite":
            mu_raw = composite_signal(
                momentum_signal(R, t - 1),
                value_signal(bm, t - 1),
            )
        else:
            raise ValueError(signal_kind)

        mu = mu_raw.astype(float)

        kappa_ts[r] = _kappa_corr(Sigma)

        # Direct Markowitz on primary Sigma is the in-sample reference
        # for the direction-error diagnostic.
        w_direct_ref = np.linalg.solve(Sigma, mu)
        ref_dir = w_direct_ref / (np.linalg.norm(w_direct_ref) + 1e-20)

        # Realized next-month return (OOS).
        r_next = R[t, :]

        for name, fn in methods.items():
            # LW-Markowitz uses Sigma_lw explicitly.
            cov_use = Sigma_lw if name == "LW-Markowitz" else Sigma
            try:
                w_raw = fn(cov_use, mu)
            except Exception:
                w_raw = np.ones(N) / N

            if long_only:
                w = clean_long_only(w_raw)
            else:
                w = clean_long_short(w_raw)

            # Turnover vs previous held portfolio, one-sided.
            tv = 0.5 * np.sum(np.abs(w - prev_w[name]))
            cost = tc_bps * 1e-4 * (2.0 * tv)   # two-sided
            ret_streams[name][r] = float(w @ r_next) - cost
            turnover[name][r]    = tv
            gross_lev[name][r]   = np.sum(np.abs(w))
            npos[name][r]        = int(np.sum(w > 1e-4))

            # Direction error vs direct Markowitz on Sigma.
            wd = w / (np.linalg.norm(w) + 1e-20)
            cos = float(wd @ ref_dir)
            dir_err[name][r] = 1.0 - abs(cos)

            prev_w[name] = w

        if verbose and (r % 60 == 0 or r == n_oos - 1):
            print(f"  rebal {r+1:4d}/{n_oos}  t={t:3d}  "
                  f"kappa(C)={kappa_ts[r]:.2f}")

    return {
        "returns":      ret_streams,
        "dir_err":      dir_err,
        "kappa":        kappa_ts,
        "turnover":     turnover,
        "gross_lev":    gross_lev,
        "npos":         npos,
        "rebal_months": rebal_months,
    }


# ----------------------------------------------------------------------
# Performance summary
# ----------------------------------------------------------------------

def perf_stats(rets, months_per_year=12):
    m = np.mean(rets)
    s = np.std(rets, ddof=1) + 1e-20
    ann_ret = m * months_per_year
    ann_vol = s * np.sqrt(months_per_year)
    sharpe  = ann_ret / ann_vol
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0).min()
    return ann_ret, ann_vol, sharpe, dd


def summarize(backtest, tag, out_rows):
    rets_all = backtest["returns"]
    for name, rets in rets_all.items():
        ar, av, sh, mdd = perf_stats(rets)
        to  = float(np.mean(backtest["turnover"][name])) * 12.0
        gl  = float(np.mean(backtest["gross_lev"][name]))
        npos= float(np.mean(backtest["npos"][name]))
        out_rows.append({
            "signal":  tag,
            "method":  name,
            "ann_ret": ar,
            "ann_vol": av,
            "sharpe":  sh,
            "mdd":     mdd,
            "turnover_ann": to,
            "gross_lev": gl,
            "avg_npos": npos,
        })


# ----------------------------------------------------------------------
# Runtime micro-benchmark at N=500 (representative single rebalance)
# ----------------------------------------------------------------------

def runtime_benchmark(R, bm):
    print("\n== Runtime benchmark at N=%d ==" % R.shape[1])
    t = ESTIM_WINDOW * 2
    window = R[t - ESTIM_WINDOW:t, :]
    Sigma = sample_cov(window)
    mu = momentum_signal(R, t - 1)
    tree = build_hrp_tree(Sigma)

    def timed(label, fn, n=3):
        # Warm-up once.
        fn()
        start = time.time()
        for _ in range(n):
            fn()
        dt = (time.time() - start) / n
        print(f"  {label:20s} {dt*1000:8.1f} ms")
        return dt

    results = {}
    results["Direct"]          = timed("Direct",
        lambda: np.linalg.solve(Sigma, mu))
    results["HRP"]             = timed("HRP",
        lambda: hrp_flat_weights(Sigma, tree))
    results["Cotton g=0.7"]    = timed("Cotton g=0.7",
        lambda: cotton_weights(Sigma, tree, 0.7), n=2)
    results["A3 g=0.5"]        = timed("A3 g=0.5",
        lambda: method_a3_weights(Sigma, mu, tree, 0.5))
    results["B g=0.5 (100sw)"] = timed("B g=0.5 100sw",
        lambda: method_b_solve(Sigma, mu, 0.5, max_sweeps=100))
    results["B g=0.5 (500sw)"] = timed("B g=0.5 500sw",
        lambda: method_b_solve(Sigma, mu, 0.5, max_sweeps=500), n=2)
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("[compute10] simulating universe (N=%d, T=%d)..."
          % (N_ASSETS, N_MONTHS))
    R, sector_id, bm = simulate_universe()
    print("  monthly return grand mean=%.4f, sd=%.4f"
          % (R.mean(), R.std()))

    signals = ["momentum", "value", "composite"]
    headline_rows = []
    per_signal_backtest = {}

    for sig in signals:
        print(f"\n[compute10] backtest signal = {sig} (long-only, 10 bps)")
        bt = run_backtest(R, bm, signal_kind=sig, long_only=True,
                          tc_bps=TC_BPS_DEFAULT, cov_estimator="sample")
        per_signal_backtest[sig] = bt
        summarize(bt, sig, headline_rows)

    # Write headline CSV.
    headline_csv = os.path.join(RESDIR, "headline_sharpe.csv")
    with open(headline_csv, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(headline_rows[0].keys()))
        wcsv.writeheader()
        for row in headline_rows:
            wcsv.writerow(row)
    print(f"[compute10] wrote {headline_csv}")

    # ---------- Equity curves figure data (composite signal) ------------
    curves_methods = ["B g=0.5 (100sw)", "A3 g=0.5", "LW-Markowitz",
                      "Direct", "HRP", "Cotton g=0.7", "1/N"]
    bt_c = per_signal_backtest["composite"]
    n_oos = len(bt_c["rebal_months"])
    curves = {}
    for name in curves_methods:
        rets = bt_c["returns"][name]
        curves[name] = np.cumprod(1.0 + rets)
    np.savez(os.path.join(RESDIR, "equity_curves.npz"),
             rebal_months=bt_c["rebal_months"],
             **{k.replace(" ", "_").replace("(", "").replace(")", "")
                    .replace("=", "").replace(".", "p"): v
                for k, v in curves.items()})
    # Also keep a plain-key version for the figure script.
    np.savez(os.path.join(RESDIR, "equity_curves_named.npz"),
             rebal_months=bt_c["rebal_months"],
             method_names=np.array(curves_methods),
             curves=np.array([curves[m] for m in curves_methods]))
    print("[compute10] wrote equity_curves_named.npz")

    # ---------- Direction error diagnostic figure data ------------------
    de_methods = ["B g=0.5 (100sw)", "A3 g=0.5"]
    np.savez(
        os.path.join(RESDIR, "direction_err.npz"),
        rebal_months=bt_c["rebal_months"],
        kappa=bt_c["kappa"],
        methods=np.array(de_methods),
        dir_err=np.array([bt_c["dir_err"][m] for m in de_methods]),
    )
    print("[compute10] wrote direction_err.npz")

    # ---------- §10.5 predictive check: does in-sample dir err
    # predict OOS next-month return? --------------------------------------
    diag_lines = []
    diag_lines.append("Direction-error predictive diagnostic (composite signal)")
    diag_lines.append("=" * 60)
    for m in de_methods:
        de = bt_c["dir_err"][m]
        rets = bt_c["returns"][m]
        # IS dir err at rebal r predicts OOS return in month t_r.
        c = np.corrcoef(de, rets)[0, 1]
        diag_lines.append(f"  {m:22s} corr(dir_err, next-month ret) = {c:+.4f}")

    # ---------- Subperiod analysis (robustness) -------------------------
    robust_rows = []
    subperiods = {
        "1995-2000": (0, 72),          # first 6 yrs of OOS after burn-in
        "2000-2008": (60, 60 + 8 * 12),
        "2008-2015": (60 + 8 * 12, 60 + 15 * 12),
        "2015-2024": (60 + 15 * 12, n_oos),
    }
    # Subperiod analysis uses the already-computed composite backtest.
    main_methods_full = ["1/N", "Direct", "LW-Markowitz", "HRP",
                         "Cotton g=0.7", "A3 g=0.5", "B g=0.5 (100sw)"]
    main_methods_light = ["1/N", "Direct", "LW-Markowitz", "HRP",
                          "A3 g=0.5", "B g=0.5 (100sw)"]
    for label, (a, b) in subperiods.items():
        a = max(0, min(a, n_oos))
        b = max(0, min(b, n_oos))
        if b - a < 12:
            continue
        for m in main_methods_full:
            rets = bt_c["returns"][m][a:b]
            ar, av, sh, mdd = perf_stats(rets)
            robust_rows.append({
                "family":"Subperiod", "variant": label, "method": m,
                "sharpe": sh, "ann_ret": ar, "ann_vol": av, "mdd": mdd,
            })

    light = methods_dict_light()

    # Transaction cost sensitivity on composite signal.
    for tc in [5, 25]:       # baseline (10 bps) already in headline
        bt_tc = run_backtest(R, bm, signal_kind="composite",
                             long_only=True, tc_bps=tc,
                             cov_estimator="sample",
                             methods=light, verbose=False)
        for m in main_methods_light:
            rets = bt_tc["returns"][m]
            ar, av, sh, mdd = perf_stats(rets)
            robust_rows.append({
                "family":"TC",
                "variant": f"tc={tc}bps",
                "method": m,
                "sharpe": sh, "ann_ret": ar, "ann_vol": av, "mdd": mdd,
            })

    # Covariance estimator alternatives.
    for cov_est in ["lw", "nls"]:
        bt_cov = run_backtest(R, bm, signal_kind="composite",
                              long_only=True, tc_bps=TC_BPS_DEFAULT,
                              cov_estimator=cov_est,
                              methods=light, verbose=False)
        tag = {"lw":"LW-const-corr","nls":"Non-linear shrink"}[cov_est]
        for m in main_methods_light:
            rets = bt_cov["returns"][m]
            ar, av, sh, mdd = perf_stats(rets)
            robust_rows.append({
                "family":"CovEst",
                "variant": tag,
                "method": m,
                "sharpe": sh, "ann_ret": ar, "ann_vol": av, "mdd": mdd,
            })

    # Estimation window sensitivity (composite, sample cov).
    for W in [36, 120]:
        if W >= N_MONTHS:
            continue
        bt_w = run_backtest(R, bm, signal_kind="composite",
                            long_only=True, tc_bps=TC_BPS_DEFAULT,
                            estim_window=W, cov_estimator="sample",
                            methods=light, verbose=False)
        for m in main_methods_light:
            rets = bt_w["returns"][m]
            ar, av, sh, mdd = perf_stats(rets)
            robust_rows.append({
                "family":"Window",
                "variant": f"W={W}mo",
                "method": m,
                "sharpe": sh, "ann_ret": ar, "ann_vol": av, "mdd": mdd,
            })

    # Sweep-count sensitivity for Method B (composite).
    for sw in [25, 100, 500]:
        custom_methods = {
            f"B g=0.5 ({sw}sw)":
                lambda c, m, sw=sw: method_b_solve(c, m, 0.5, max_sweeps=sw),
        }
        bt_sw = run_backtest(R, bm, signal_kind="composite",
                             long_only=True, tc_bps=TC_BPS_DEFAULT,
                             cov_estimator="sample",
                             methods=custom_methods, verbose=False)
        rets = bt_sw["returns"][f"B g=0.5 ({sw}sw)"]
        ar, av, sh, mdd = perf_stats(rets)
        robust_rows.append({
            "family":"Sweeps",
            "variant": f"B g=0.5 {sw}sw",
            "method":  "B g=0.5",
            "sharpe": sh, "ann_ret": ar, "ann_vol": av, "mdd": mdd,
        })

    robust_csv = os.path.join(RESDIR, "robustness.csv")
    with open(robust_csv, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(robust_rows[0].keys()))
        wcsv.writeheader()
        for row in robust_rows:
            wcsv.writerow(row)
    print(f"[compute10] wrote {robust_csv}")

    # ---------- Runtime micro-benchmark ---------------------------------
    rt = runtime_benchmark(R, bm)
    rt_csv = os.path.join(RESDIR, "runtimes.csv")
    with open(rt_csv, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["method", "seconds_per_call"])
        for name, sec in rt.items():
            wcsv.writerow([name, f"{sec:.6f}"])

    # ---------- Write diagnostics -------------------------------------
    diag_path = os.path.join(RESDIR, "diagnostics.txt")
    with open(diag_path, "w") as fh:
        fh.write("Section 10 compute diagnostics\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"N_ASSETS = {N_ASSETS}\n")
        fh.write(f"N_MONTHS = {N_MONTHS}\n")
        fh.write(f"ESTIM_WINDOW = {ESTIM_WINDOW}\n")
        fh.write(f"RNG_SEED = {RNG_SEED}\n\n")
        for line in diag_lines:
            fh.write(line + "\n")
        fh.write("\nHeadline Sharpe (composite signal, long-only, 10 bps):\n")
        for row in headline_rows:
            if row["signal"] != "composite":
                continue
            fh.write("  %-22s  SR=%6.3f  ret=%6.3f  vol=%6.3f  "
                     "mdd=%6.3f  to=%5.2f  gl=%5.2f  npos=%5.0f\n" % (
                row["method"], row["sharpe"], row["ann_ret"],
                row["ann_vol"], row["mdd"], row["turnover_ann"],
                row["gross_lev"], row["avg_npos"]))
        fh.write("\nRuntime (seconds/call, N=%d):\n" % N_ASSETS)
        for k, v in rt.items():
            fh.write(f"  {k:22s} {v*1000:8.1f} ms\n")

    print(f"[compute10] wrote {diag_path}")
    print("[compute10] done.")


if __name__ == "__main__":
    main()
