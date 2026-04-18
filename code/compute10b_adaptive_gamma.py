"""
compute10b_adaptive_gamma.py
============================

Time-varying adaptive gamma* on the simulated CRSP-analog backtest.

This script re-runs the Section 10 walk-forward backtest using Method B
at an adaptive shrinkage level

    gamma*_t  =  1 / ( 1  +  c_bar * kappa(hat_C_t)^2 * N / (T * IC_t^2) )

derived in Section 6.5, and compares it against fixed-gamma Method B
at gamma in {0.3, 0.5, 0.7} as well as fixed Markowitz and HRP. Three
IC estimators are evaluated:

  trailing_ic       : rank correlation of hat_mu_t with realised next-
                      month return, averaged over a 12-month trailing
                      window (baseline).
  expanding_ic      : same but over the full trailing history [0, t).
  fixed_ic          : constant IC = 0.05 (the "I do not want to estimate
                      IC" case; tests whether kappa(C) time variation
                      alone is enough to outperform fixed gamma).

Artifacts (under results/sec10/)
--------------------------------
    adaptive_gamma_panel.csv      one row per (IC estimator, method)
    adaptive_gamma_ts.npz         time series of gamma*_t, kappa(C)_t,
                                  estimated IC_t, and per-method Sharpe
                                  differences for fig10:gamma_t_series
                                  and fig10:adaptive_rolling_diff.

Usage
-----
    python figures/code/compute10b_adaptive_gamma.py

Reads c_bar from results/sec09_adaptive/summary.txt if available,
otherwise falls back to a default of 1e-5 (documented).
"""

from __future__ import annotations

import os
import re
import csv
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import shared machinery from compute10_crsp_backtest.
# The import style keeps the two scripts consistent and avoids drifting
# factor-model or signal definitions.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from compute10_crsp_backtest import (  # noqa: E402
    simulate_universe, momentum_signal, value_signal, composite_signal,
    sample_cov, lw_constant_corr, perf_stats, clean_long_only,
    N_ASSETS, N_MONTHS, ESTIM_WINDOW, RNG_SEED, TC_BPS_DEFAULT,
    hrp, direct_markowitz, one_over_n, lw_markowitz,
)
from study import method_b_solve  # noqa: E402

RESDIR = os.path.join(ROOT, "results", "sec10")
os.makedirs(RESDIR, exist_ok=True)
SEC09_DIR = os.path.join(ROOT, "results", "sec09_adaptive")


# ----------------------------------------------------------------------
# Read c_bar from the synthetic calibration summary
# ----------------------------------------------------------------------

DEFAULT_C_BAR = 1.0e-5   # conservative fallback
DEFAULT_ALPHA = 1.0      # theoretical exponent


def load_calibration() -> tuple[float, float]:
    """Return (c_bar, alpha) loaded from the Exp 1 summary file, or
    defaults if the file is missing or unparseable."""
    path = os.path.join(SEC09_DIR, "summary.txt")
    if not os.path.exists(path):
        print(f"[compute10b] {path} not found; falling back to "
              f"c_bar = {DEFAULT_C_BAR}, alpha = {DEFAULT_ALPHA}")
        return DEFAULT_C_BAR, DEFAULT_ALPHA
    txt = open(path).read()
    mc = re.search(r"c_bar\s*=\s*([0-9.eE+-]+)", txt)
    ma = re.search(r"alpha\s*=\s*([0-9.eE+-]+)", txt)
    c_bar = float(mc.group(1)) if mc else DEFAULT_C_BAR
    alpha = float(ma.group(1)) if ma else DEFAULT_ALPHA
    print(f"[compute10b] loaded c_bar = {c_bar:.3e}, alpha = {alpha:.3f}")
    return c_bar, alpha


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def kappa_corr(cov: np.ndarray) -> float:
    d = np.sqrt(np.diag(cov))
    C = cov / np.outer(d, d)
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 1e-12, None)
    return float(w.max() / w.min())


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, using scipy.stats-free implementation
    (we avoid the scipy dependency in the compute scripts to keep the
    paper's reproducibility footprint small)."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    sx = float(np.sqrt((rx * rx).sum()))
    sy = float(np.sqrt((ry * ry).sum()))
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return float((rx * ry).sum() / (sx * sy))


def adaptive_gamma(kappa_C: float, ic: float, N: int, T: int,
                   c_bar: float, alpha: float = 1.0,
                   floor: float = 0.05,
                   cap: float = 1.0) -> float:
    """Adaptive gamma* from Section 6.5 eq (6.10), two-parameter fit:

        gamma* = 1 / ( 1 + c_bar * NSR^alpha )

    where NSR = kappa(C)^2 * N / (T * IC^2). Clipped to [floor, cap].
    alpha = 1 is the theoretical value; the empirical fit in
    §9.5 reports a typically-smaller alpha.
    """
    ic = max(abs(ic), 0.005)
    nsr = (kappa_C ** 2) * (N / T) / (ic * ic)
    g = 1.0 / (1.0 + c_bar * (nsr ** alpha))
    return float(np.clip(g, floor, cap))


# ----------------------------------------------------------------------
# IC estimators
# ----------------------------------------------------------------------

class ICTrailing:
    """Rolling mean of the last `window` single-period Spearman ICs of
    hat_mu_r against realised next-month return."""

    def __init__(self, window: int = 12, default: float = 0.05):
        self.window = window
        self.default = default
        self.history: list[float] = []

    def update(self, mu_t: np.ndarray, r_next: np.ndarray) -> None:
        ic = rank_correlation(mu_t, r_next)
        self.history.append(ic)

    def value(self) -> float:
        if not self.history:
            return self.default
        recent = self.history[-self.window:]
        return float(np.mean(recent))


class ICExpanding:
    """Cumulative mean of all historical single-period Spearman ICs."""

    def __init__(self, default: float = 0.05):
        self.default = default
        self.sum = 0.0
        self.n = 0

    def update(self, mu_t: np.ndarray, r_next: np.ndarray) -> None:
        ic = rank_correlation(mu_t, r_next)
        self.sum += ic
        self.n += 1

    def value(self) -> float:
        if self.n == 0:
            return self.default
        return float(self.sum / self.n)


class ICFixed:
    """Constant IC, used for the 'no IC estimate available' variant."""

    def __init__(self, value: float = 0.05):
        self.ic = value

    def update(self, *args) -> None:
        pass

    def value(self) -> float:
        return self.ic


# ----------------------------------------------------------------------
# Adaptive-gamma backtest engine
# ----------------------------------------------------------------------

def run_adaptive_backtest(R: np.ndarray, bm: np.ndarray, c_bar: float,
                          alpha: float,
                          ic_estimator_name: str,
                          fixed_ic: float = 0.05,
                          signal_kind: str = "composite",
                          estim_window: int = ESTIM_WINDOW,
                          tc_bps: int = TC_BPS_DEFAULT,
                          sweeps: int = 100,
                          verbose: bool = True) -> dict:
    """
    Walk-forward backtest that runs Method B at an adaptive gamma* per
    rebalance, alongside fixed-gamma comparators and 1/N.

    Returns a dict containing Sharpe and return-series for each method,
    the gamma*_t time series, kappa(C_t), and estimated IC_t.
    """
    T, N = R.shape
    rebal_months = np.arange(estim_window, T)
    n_oos = len(rebal_months)

    # Instantiate IC estimator.
    if ic_estimator_name == "trailing_ic":
        icest = ICTrailing(window=12, default=fixed_ic)
    elif ic_estimator_name == "expanding_ic":
        icest = ICExpanding(default=fixed_ic)
    elif ic_estimator_name == "fixed_ic":
        icest = ICFixed(value=fixed_ic)
    else:
        raise ValueError(ic_estimator_name)

    def label_adaptive() -> str:
        return f"B adaptive ({ic_estimator_name})"

    methods = {
        "1/N":                 lambda c, m: np.ones(N) / N,
        "HRP":                 lambda c, m: hrp(c, m),
        "LW-Markowitz":        lambda c, m: lw_markowitz(c, m),
        "B g=0.3":             lambda c, m: method_b_solve(c, m, 0.3,
                                                           max_sweeps=sweeps),
        "B g=0.5":             lambda c, m: method_b_solve(c, m, 0.5,
                                                           max_sweeps=sweeps),
        "B g=0.7":             lambda c, m: method_b_solve(c, m, 0.7,
                                                           max_sweeps=sweeps),
        label_adaptive():      None,  # gamma supplied per-rebal below
    }

    returns = {name: np.zeros(n_oos) for name in methods}
    turnover = {name: np.zeros(n_oos) for name in methods}
    prev_w = {name: np.ones(N) / N for name in methods}

    gamma_star_ts = np.zeros(n_oos)
    kappa_ts = np.zeros(n_oos)
    ic_ts = np.zeros(n_oos)

    for r, t in enumerate(rebal_months):
        window = R[t - estim_window:t, :]
        Sigma = sample_cov(window)
        Sigma_lw = lw_constant_corr(window)

        if signal_kind == "momentum":
            mu = momentum_signal(R, t - 1)
        elif signal_kind == "value":
            mu = value_signal(bm, t - 1)
        elif signal_kind == "composite":
            mu = composite_signal(
                momentum_signal(R, t - 1), value_signal(bm, t - 1)
            )
        else:
            raise ValueError(signal_kind)
        mu = mu.astype(float)

        kC = kappa_corr(Sigma)
        ic_val = icest.value()
        gamma_t = adaptive_gamma(kC, ic_val, N=N, T=estim_window,
                                 c_bar=c_bar, alpha=alpha)

        gamma_star_ts[r] = gamma_t
        kappa_ts[r] = kC
        ic_ts[r] = ic_val

        r_next = R[t, :]

        for name in methods:
            if name == label_adaptive():
                w_raw = method_b_solve(Sigma, mu, gamma_t,
                                       max_sweeps=sweeps)
            elif name == "LW-Markowitz":
                w_raw = methods[name](Sigma_lw, mu)
            else:
                try:
                    w_raw = methods[name](Sigma, mu)
                except Exception:
                    w_raw = np.ones(N) / N

            w = clean_long_only(w_raw)

            tv = 0.5 * float(np.sum(np.abs(w - prev_w[name])))
            cost = tc_bps * 1e-4 * (2.0 * tv)
            returns[name][r] = float(w @ r_next) - cost
            turnover[name][r] = tv
            prev_w[name] = w

        # Update IC estimator with realised return for the current
        # rebal's signal -> next-month return pair.
        icest.update(mu, r_next)

        if verbose and (r % 60 == 0 or r == n_oos - 1):
            print(f"  [{ic_estimator_name}] rebal {r+1:4d}/{n_oos}  "
                  f"t={t:3d}  kappa(C)={kC:8.2f}  "
                  f"IC={ic_val:+.3f}  gamma*={gamma_t:.3f}")

    return {
        "ic_estimator": ic_estimator_name,
        "rebal_months": rebal_months,
        "gamma_star_ts": gamma_star_ts,
        "kappa_ts": kappa_ts,
        "ic_ts": ic_ts,
        "returns": returns,
        "turnover": turnover,
        "methods": list(methods.keys()),
        "adaptive_label": label_adaptive(),
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def summarise_panel(bt: dict) -> list[dict]:
    rows = []
    for name in bt["methods"]:
        rets = bt["returns"][name]
        ar, av, sh, mdd = perf_stats(rets)
        rows.append(dict(
            ic_estimator=bt["ic_estimator"],
            method=name,
            sharpe=float(sh),
            ann_ret=float(ar),
            ann_vol=float(av),
            mdd=float(mdd),
            avg_turnover=float(np.mean(bt["turnover"][name])),
            mean_gamma_star=float(np.mean(bt["gamma_star_ts"])
                                  if "adaptive" in name else np.nan),
            std_gamma_star=float(np.std(bt["gamma_star_ts"])
                                 if "adaptive" in name else np.nan),
        ))
    return rows


def main() -> None:
    c_bar, alpha = load_calibration()
    print("[compute10b] simulating universe...")
    R, sector_id, bm = simulate_universe()
    print(f"  R shape = {R.shape}")

    all_panels = []
    ts_output = {}

    for ic_name in ("trailing_ic", "expanding_ic", "fixed_ic"):
        print(f"\n[compute10b] adaptive backtest with {ic_name}")
        bt = run_adaptive_backtest(R, bm, c_bar=c_bar, alpha=alpha,
                                   ic_estimator_name=ic_name,
                                   signal_kind="composite")
        panel = summarise_panel(bt)
        all_panels.extend(panel)
        # Save time series for the trailing-IC run which is the headline.
        if ic_name == "trailing_ic":
            adap = bt["adaptive_label"]
            ret_adap = bt["returns"][adap]
            ret_05 = bt["returns"]["B g=0.5"]
            roll = 12
            # Rolling mean Sharpe difference over `roll` months.
            diffs = ret_adap - ret_05
            rolling = np.convolve(diffs, np.ones(roll) / roll, mode="same")
            ts_output = dict(
                rebal_months=bt["rebal_months"],
                gamma_star_ts=bt["gamma_star_ts"],
                kappa_ts=bt["kappa_ts"],
                ic_ts=bt["ic_ts"],
                ret_adaptive=ret_adap,
                ret_fixed_05=ret_05,
                rolling_diff_12mo=rolling,
                c_bar=np.array([c_bar]),
                alpha=np.array([alpha]),
            )

    # Write panel CSV.
    panel_csv = os.path.join(RESDIR, "adaptive_gamma_panel.csv")
    if all_panels:
        with open(panel_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_panels[0].keys()))
            w.writeheader()
            for row in all_panels:
                w.writerow(row)
        print(f"[compute10b] wrote {panel_csv}")

    if ts_output:
        ts_path = os.path.join(RESDIR, "adaptive_gamma_ts.npz")
        np.savez(ts_path, **ts_output)
        print(f"[compute10b] wrote {ts_path}")

    # Short diagnostic dump.
    print("\n[compute10b] headline panel (composite signal)")
    for row in all_panels:
        print(f"  {row['ic_estimator']:14s}  "
              f"{row['method']:22s}  "
              f"SR={row['sharpe']:+.3f}  "
              f"ret={row['ann_ret']:+.3f}  "
              f"vol={row['ann_vol']:.3f}  "
              f"to={row['avg_turnover']:.3f}")


if __name__ == "__main__":
    main()
