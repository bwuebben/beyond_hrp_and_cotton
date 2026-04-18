"""
Walk-forward Monte Carlo experiments for the Method B / Method A3 paper.

This script reproduces the out-of-sample (OOS) experiments reported in
outline_new.md, §9.8 through §9.10:

  Experiment 1: Sensitivity sweep across mu seeds and sample sizes,
                with oracle and sample-mean mu estimation.
  Experiment 2: Structural (sector-tilt) mu.
  Experiment 3: A1 deep dive -- direction cosines under estimation noise
                and on the true covariance (includes the sign-pathology
                diagnostic).
  Experiment 4: Minimum variance MC including Cotton as a baseline.
  Experiment 4b: Cotton as a signal-blind baseline on a signal-aware
                 universe.

Run from the project root:

    python3 walkforward.py                     # all experiments
    python3 walkforward.py --exp 1             # only sensitivity sweep
    python3 walkforward.py > results/all.txt   # tee to file

All experiments use synthetic data generated from make_structured_cov
in study.py. Base universe N=100 (unless changed), 5 sectors, rho_w=0.6,
rho_c=0.15, vols uniform [0.15, 0.40], seed=42 for cov_true.
"""
import argparse
import numpy as np

from study import (make_structured_cov, build_hrp_tree, hrp_flat_weights,
                   cotton_weights, method_a1_weights, method_a1_l1_weights,
                   method_a2_weights, method_a3_weights, method_b_solve,
                   _kappa_corr)

np.set_printoptions(precision=3, suppress=True)


# ================================================================
# Shared machinery
# ================================================================

def make_signal_methods(N):
    """All methods to benchmark in the signal-aware MC experiments."""
    return {
        '1/N':       lambda cov, mu, tr: np.ones(N) / N,
        'HRP':       lambda cov, mu, tr: hrp_flat_weights(cov, tr),
        'Direct':    lambda cov, mu, tr: np.linalg.solve(cov, mu),
        'A1  g=0.0': lambda cov, mu, tr: method_a1_weights(cov, mu, tr, 0.0),
        'A1  g=0.5': lambda cov, mu, tr: method_a1_weights(cov, mu, tr, 0.5),
        'A1  g=1.0': lambda cov, mu, tr: method_a1_weights(cov, mu, tr, 1.0),
        'A1L1 g=0.0': lambda cov, mu, tr: method_a1_l1_weights(cov, mu, tr, 0.0),
        'A1L1 g=0.5': lambda cov, mu, tr: method_a1_l1_weights(cov, mu, tr, 0.5),
        'A1L1 g=1.0': lambda cov, mu, tr: method_a1_l1_weights(cov, mu, tr, 1.0),
        'A3  g=0.0': lambda cov, mu, tr: method_a3_weights(cov, mu, tr, 0.0),
        'A3  g=0.5': lambda cov, mu, tr: method_a3_weights(cov, mu, tr, 0.5),
        'A3  g=1.0': lambda cov, mu, tr: method_a3_weights(cov, mu, tr, 1.0),
        'B g=0.3':   lambda cov, mu, tr: method_b_solve(cov, mu, 0.3, max_sweeps=100),
        'B g=0.5':   lambda cov, mu, tr: method_b_solve(cov, mu, 0.5, max_sweeps=100),
        'B g=0.7':   lambda cov, mu, tr: method_b_solve(cov, mu, 0.7, max_sweeps=100),
        'B g=1.0':   lambda cov, mu, tr: method_b_solve(cov, mu, 1.0, max_sweeps=100),
    }


def run_signal_mc(cov_true, mu_true, n_mc, T, mu_estimator='oracle',
                  seed=0, ridge=1e-4):
    """
    Monte Carlo walk-forward on a signal-aware problem.

    For each of n_mc trials, draw T returns from N(mu_true, cov_true),
    estimate cov_hat from samples, build an HRP tree from cov_hat, and
    run every method on (cov_hat, mu_use) where mu_use is either
    mu_true ('oracle') or the sample mean.

    OOS metric: Sharpe = (w' mu_true) / sqrt(w' cov_true w).
    Returns dict{method_name -> list of Sharpes}.
    """
    rng = np.random.RandomState(seed)
    N = cov_true.shape[0]
    methods = make_signal_methods(N)
    results = {name: [] for name in methods}
    for _ in range(n_mc):
        samples = rng.multivariate_normal(mu_true, cov_true, T)
        cov_hat = np.cov(samples.T) + ridge * np.eye(N)
        mu_use = mu_true if mu_estimator == 'oracle' else samples.mean(axis=0)
        try:
            tree = build_hrp_tree(cov_hat)
        except Exception:
            continue
        for name, fn in methods.items():
            try:
                w = fn(cov_hat, mu_use, tree)
                if not np.all(np.isfinite(w)):
                    results[name].append(np.nan); continue
                exp_ret = float(w @ mu_true)
                var = float(w @ cov_true @ w)
                sharpe = exp_ret / np.sqrt(var) if var > 1e-15 else 0.0
                results[name].append(sharpe)
            except Exception:
                results[name].append(np.nan)
    return results


def tabulate(results, title):
    print(f"\n{title}")
    print(f"  {'method':<12} {'mean Sharpe':>12} {'std':>8} "
          f"{'min':>8} {'max':>8}")
    print("  " + "-" * 54)
    for name, vals in results.items():
        v = np.array(vals, dtype=float)
        valid = v[np.isfinite(v)]
        if len(valid) == 0:
            print(f"  {name:<12} {'nan':>12}"); continue
        print(f"  {name:<12} {valid.mean():12.3f} {valid.std():8.3f} "
              f"{valid.min():8.3f} {valid.max():8.3f}")


# ================================================================
# Experiment 1: Sensitivity across mu seeds
# ================================================================

def experiment_1_sensitivity(N=100, n_mc=40, Ts=(60, 120, 240, 500),
                              mu_seeds=(0, 1, 7, 42, 100, 999, 2024, 13)):
    print("=" * 78)
    print(f"EXPERIMENT 1: Sensitivity across mu seeds "
          f"(N={N}, n_mc={n_mc}/seed)")
    print("=" * 78)
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    print(f"Base universe: kappa(corr) = {_kappa_corr(cov_true):.2f}")

    for T in Ts:
        for mu_est in ('oracle', 'sample'):
            print(f"\n--- T={T}, mu estimator: {mu_est} ---")
            dist = {}
            for mu_seed in mu_seeds:
                rng = np.random.RandomState(mu_seed)
                mu_true = rng.randn(N) * 0.02
                results = run_signal_mc(cov_true, mu_true, n_mc, T, mu_est,
                                        seed=mu_seed + 1000)
                for name, vals in results.items():
                    dist.setdefault(name, []).append(
                        float(np.nanmean(vals)))
            print(f"  (across {len(mu_seeds)} mu seeds)")
            print(f"  {'method':<12} {'mean_of_means':>14} {'min':>8} "
                  f"{'max':>8} {'n_pos':>6}")
            print("  " + "-" * 56)
            for name, vals in dist.items():
                v = np.array(vals)
                print(f"  {name:<12} {v.mean():14.3f} {v.min():8.3f} "
                      f"{v.max():8.3f} {(v > 0).sum():>4}/{len(v)}")


# ================================================================
# Experiment 2: Structural (sector-tilt) signal
# ================================================================

def make_sector_tilt_mu(N, n_sectors=5, tilts=(0.04, -0.04, 0.02, -0.02, 0.0)):
    mu = np.zeros(N)
    per = N // n_sectors
    for s in range(n_sectors):
        mu[s * per:(s + 1) * per] = tilts[s]
    return mu


def experiment_2_structural(N=100, n_mc=60, Ts=(60, 120, 240)):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 2: Structural mu (sector tilts), N={N}, n_mc={n_mc}")
    print("=" * 78)
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    mu_true = make_sector_tilt_mu(N)
    w_or = np.linalg.solve(cov_true, mu_true)
    oracle_sharpe = (w_or @ mu_true) / np.sqrt(w_or @ cov_true @ w_or)
    print(f"\nOracle Sharpe (structured mu): {oracle_sharpe:.3f}")
    print(f"sign(mu) == sign(w_oracle) fraction: "
          f"{(np.sign(mu_true) == np.sign(w_or)).mean():.2f}")
    print(f"||w_oracle||_1 (gross leverage) = {np.abs(w_or).sum():.2f}")
    for T in Ts:
        for mu_est in ('oracle', 'sample'):
            results = run_signal_mc(cov_true, mu_true, n_mc, T, mu_est,
                                    seed=999)
            tabulate(results, f"T={T}, mu={mu_est}")


# ================================================================
# Experiment 3: A1 deep dive
# ================================================================

def experiment_3_a1_deep_dive(N=100, n_mc=60, Ts=(60, 240, 1000)):
    print("\n" + "=" * 78)
    print("EXPERIMENT 3: A1 deep dive -- direction diagnostic")
    print("=" * 78)
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    mu_true = make_sector_tilt_mu(N)
    w_or = np.linalg.solve(cov_true, mu_true)
    print("\nCosine vs oracle under estimation noise (structured mu)")
    for T in Ts:
        rng = np.random.RandomState(5)
        cos_a1, cos_a3, cos_a1l1 = [], [], []
        sh_a1, sh_a3, sh_a1l1 = [], [], []
        for _ in range(n_mc):
            samples = rng.multivariate_normal(mu_true, cov_true, T)
            cov_hat = np.cov(samples.T) + 1e-4 * np.eye(N)
            tree = build_hrp_tree(cov_hat)
            for (w, cos_list, sh_list) in [
                (method_a1_weights(cov_hat, mu_true, tree, 0.5), cos_a1, sh_a1),
                (method_a1_l1_weights(cov_hat, mu_true, tree, 0.5), cos_a1l1, sh_a1l1),
                (method_a3_weights(cov_hat, mu_true, tree, 0.5), cos_a3, sh_a3),
            ]:
                var = w @ cov_true @ w
                sh = (w @ mu_true) / np.sqrt(var) if var > 1e-15 else 0.0
                cs = float(w @ w_or) / (
                    np.linalg.norm(w) * np.linalg.norm(w_or) + 1e-30)
                cos_list.append(cs); sh_list.append(sh)
        print(f"\n  T = {T}:")
        print(f"    A1   g=0.5: Sharpe {np.mean(sh_a1):+.3f} +/- "
              f"{np.std(sh_a1):.3f}, "
              f"cos {np.mean(cos_a1):+.3f} +/- {np.std(cos_a1):.3f}, "
              f"frac_neg {(np.array(cos_a1) < 0).mean():.2f}")
        print(f"    A1L1 g=0.5: Sharpe {np.mean(sh_a1l1):+.3f} +/- "
              f"{np.std(sh_a1l1):.3f}, "
              f"cos {np.mean(cos_a1l1):+.3f} +/- {np.std(cos_a1l1):.3f}, "
              f"frac_neg {(np.array(cos_a1l1) < 0).mean():.2f}")
        print(f"    A3   g=0.5: Sharpe {np.mean(sh_a3):+.3f} +/- "
              f"{np.std(sh_a3):.3f}, "
              f"cos {np.mean(cos_a3):+.3f} +/- {np.std(cos_a3):.3f}, "
              f"frac_neg {(np.array(cos_a3) < 0).mean():.2f}")

    print("\n--- All methods on TRUE cov (no estimation error) ---")
    for label, mu in [("structured sector-tilt mu", mu_true),
                      ("random mu (seed 7)",
                       np.random.RandomState(7).randn(N) * 0.02)]:
        w_or2 = np.linalg.solve(cov_true, mu)
        osh = (w_or2 @ mu) / np.sqrt(w_or2 @ cov_true @ w_or2)
        tree_true = build_hrp_tree(cov_true)
        methods = make_signal_methods(N)
        print(f"\n  {label}, Oracle Sharpe = {osh:.3f}")
        print(f"  {'method':<12} {'Sharpe':>10} {'cos(w,oracle)':>14} "
              f"{'|w|1':>8}")
        print("  " + "-" * 50)
        for name, fn in methods.items():
            try:
                w = fn(cov_true, mu, tree_true)
                if not np.all(np.isfinite(w)): continue
                var = w @ cov_true @ w
                sh = (w @ mu) / np.sqrt(var) if var > 1e-15 else 0.0
                cs = float(w @ w_or2) / (
                    np.linalg.norm(w) * np.linalg.norm(w_or2) + 1e-30)
                print(f"  {name:<12} {sh:10.3f} {cs:14.3f} "
                      f"{np.abs(w).sum():8.2f}")
            except Exception:
                pass


# ================================================================
# Experiment 4: Minimum variance MC (mu = 1), Cotton included
# ================================================================

def experiment_4_minvar(N=100, n_mc=80, Ts=(60, 120, 240, 500)):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 4: Minimum variance MC (mu = 1), N={N}, n_mc={n_mc}")
    print("=" * 78)
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    mu1 = np.ones(N)
    w_mv = np.linalg.solve(cov_true, mu1); w_mv /= w_mv.sum()
    oracle_var = float(w_mv @ cov_true @ w_mv)
    oracle_sharpe = 1.0 / np.sqrt(oracle_var)
    print(f"\nOracle min-var variance = {oracle_var:.6f}, "
          f"1/vol = {oracle_sharpe:.3f}")

    methods = {
        '1/N':             lambda cov, tr: np.ones(N) / N,
        'HRP (De Prado)':  lambda cov, tr: hrp_flat_weights(cov, tr),
        'Cotton g=0.0':    lambda cov, tr: cotton_weights(cov, tr, 0.0),
        'Cotton g=0.3':    lambda cov, tr: cotton_weights(cov, tr, 0.3),
        'Cotton g=0.5':    lambda cov, tr: cotton_weights(cov, tr, 0.5),
        'Cotton g=0.7':    lambda cov, tr: cotton_weights(cov, tr, 0.7),
        'Cotton g=1.0':    lambda cov, tr: cotton_weights(cov, tr, 1.0),
        'A2 g=0.0':        lambda cov, tr: method_a2_weights(cov, mu1, tr, 0.0),
        'A2 g=0.7':        lambda cov, tr: method_a2_weights(cov, mu1, tr, 0.7),
        'A1L1 g=0.0':      lambda cov, tr: method_a1_l1_weights(cov, mu1, tr, 0.0),
        'A1L1 g=0.5':      lambda cov, tr: method_a1_l1_weights(cov, mu1, tr, 0.5),
        'A1L1 g=1.0':      lambda cov, tr: method_a1_l1_weights(cov, mu1, tr, 1.0),
        'A3 g=0.0':        lambda cov, tr: method_a3_weights(cov, mu1, tr, 0.0),
        'A3 g=0.5':        lambda cov, tr: method_a3_weights(cov, mu1, tr, 0.5),
        'A3 g=1.0':        lambda cov, tr: method_a3_weights(cov, mu1, tr, 1.0),
        'B  g=0.3':        lambda cov, tr: method_b_solve(cov, mu1, 0.3, max_sweeps=100),
        'B  g=0.5':        lambda cov, tr: method_b_solve(cov, mu1, 0.5, max_sweeps=100),
        'B  g=0.7':        lambda cov, tr: method_b_solve(cov, mu1, 0.7, max_sweeps=100),
        'B  g=1.0':        lambda cov, tr: method_b_solve(cov, mu1, 1.0, max_sweeps=100),
        'Direct min-var':  lambda cov, tr: np.linalg.solve(cov, mu1),
    }

    def norm1(w):
        s = w.sum()
        return w / s if abs(s) > 1e-15 else w

    for T in Ts:
        rng = np.random.RandomState(1)
        out = {name: {'var': [], 'sharpe': []} for name in methods}
        for _ in range(n_mc):
            samples = rng.multivariate_normal(np.zeros(N), cov_true, T)
            cov_hat = np.cov(samples.T) + 1e-4 * np.eye(N)
            tree = build_hrp_tree(cov_hat)
            for name, fn in methods.items():
                try:
                    w = fn(cov_hat, tree)
                    if not np.all(np.isfinite(w)):
                        out[name]['var'].append(np.nan)
                        out[name]['sharpe'].append(np.nan); continue
                    w = norm1(w)
                    var = float(w @ cov_true @ w)
                    out[name]['var'].append(var)
                    out[name]['sharpe'].append(
                        1.0 / np.sqrt(var) if var > 1e-15 else 0.0)
                except Exception:
                    out[name]['var'].append(np.nan)
                    out[name]['sharpe'].append(np.nan)
        print(f"\nT = {T}  (N/T = {N/T:.2f})")
        print(f"  {'method':<18} {'mean OOS vol':>14} "
              f"{'mean OOS Sharpe':>18} {'vs oracle':>12}")
        print("  " + "-" * 64)
        for name, d in out.items():
            v = np.array(d['var']); s = np.array(d['sharpe'])
            ok = np.isfinite(v)
            if ok.sum() == 0: continue
            mean_vol = float(np.sqrt(np.mean(v[ok])))
            mean_sh = float(np.mean(s[ok]))
            pct = 100 * mean_sh / oracle_sharpe
            print(f"  {name:<18} {mean_vol:14.4f} {mean_sh:18.3f} "
                  f"{pct:11.1f}%")


# ================================================================
# Experiment 4b: Cotton as signal-blind baseline
# ================================================================

def experiment_4b_cotton_signal_blind(N=100, n_mc=60, T=120):
    print("\n" + "=" * 78)
    print("EXPERIMENT 4b: Cotton as signal-blind baseline on signal-aware problem")
    print("=" * 78)
    cov_true, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
    mu = make_sector_tilt_mu(N)
    w_or = np.linalg.solve(cov_true, mu)
    oracle_sh = (w_or @ mu) / np.sqrt(w_or @ cov_true @ w_or)
    print(f"\nOracle Sharpe on structured signal: {oracle_sh:.3f}")

    methods = {
        '1/N':            lambda cov, mu, tr: np.ones(N) / N,
        'HRP':            lambda cov, mu, tr: hrp_flat_weights(cov, tr),
        'Cotton g=0.5':   lambda cov, mu, tr: cotton_weights(cov, tr, 0.5),
        'Cotton g=0.7':   lambda cov, mu, tr: cotton_weights(cov, tr, 0.7),
        'Cotton g=1.0':   lambda cov, mu, tr: cotton_weights(cov, tr, 1.0),
        'A1L1 g=0.5':     lambda cov, mu, tr: method_a1_l1_weights(cov, mu, tr, 0.5),
        'A1L1 g=1.0':     lambda cov, mu, tr: method_a1_l1_weights(cov, mu, tr, 1.0),
        'A3 g=0.5':       lambda cov, mu, tr: method_a3_weights(cov, mu, tr, 0.5),
        'B g=0.5':        lambda cov, mu, tr: method_b_solve(cov, mu, 0.5, max_sweeps=100),
        'B g=0.7':        lambda cov, mu, tr: method_b_solve(cov, mu, 0.7, max_sweeps=100),
        'Direct':         lambda cov, mu, tr: np.linalg.solve(cov, mu),
    }
    rng = np.random.RandomState(999)
    out = {name: [] for name in methods}
    for _ in range(n_mc):
        samples = rng.multivariate_normal(mu, cov_true, T)
        cov_hat = np.cov(samples.T) + 1e-4 * np.eye(N)
        tree = build_hrp_tree(cov_hat)
        for name, fn in methods.items():
            try:
                w = fn(cov_hat, mu, tree)
                if not np.all(np.isfinite(w)):
                    out[name].append(np.nan); continue
                var = w @ cov_true @ w
                sh = (w @ mu) / np.sqrt(var) if var > 1e-15 else 0.0
                out[name].append(sh)
            except Exception:
                out[name].append(np.nan)
    print(f"\nT = {T}, n_mc = {n_mc}, oracle mu")
    print(f"  {'method':<16} {'mean Sharpe':>14} {'std':>10} "
          f"{'vs oracle':>12}")
    print("  " + "-" * 56)
    for name, vals in out.items():
        v = np.array(vals); valid = v[np.isfinite(v)]
        if len(valid) == 0: continue
        pct = 100 * valid.mean() / oracle_sh
        print(f"  {name:<16} {valid.mean():14.3f} {valid.std():10.3f} "
              f"{pct:11.1f}%")


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=int, default=0,
                    help='Which experiment to run (1-4, 0 = all)')
    args = ap.parse_args()

    if args.exp in (0, 1):
        experiment_1_sensitivity()
    if args.exp in (0, 2):
        experiment_2_structural()
    if args.exp in (0, 3):
        experiment_3_a1_deep_dive()
    if args.exp in (0, 4):
        experiment_4_minvar()
        experiment_4b_cotton_signal_blind()


if __name__ == '__main__':
    main()
