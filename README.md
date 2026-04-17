# Beyond HRP and Cotton

Reproducibility repository for the paper

> **Beyond HRP and Cotton: Hierarchical and Iterative Methods for General Mean-Variance Portfolios**
> Bernd J. Wuebben

This directory contains the Python code that produced every figure, table, and
numerical result quoted in the paper, together with the committed result
artifacts so reviewers can inspect the numbers without re-running the full
pipeline.

---

## What the paper is about

The paper studies hierarchical and iterative alternatives to Markowitz
mean-variance optimization, with a focus on the regime where the covariance
matrix is ill-conditioned and the signal `mu` is noisy. It compares:

- **HRP** (De Prado's Hierarchical Risk Parity) and its minimum-variance
  interpretation,
- **Cotton** (Schur-complement shrinkage, `gamma in [0, 1]`),
- **Method A1 / A2 / A3** — three recursive-allocation variants that expose
  the sign and normalization pathologies discussed in Appendix C,
- **Method B** — a Gauss-Seidel iterative solver on the shrunk system
  `P_gamma w = mu` that the paper recommends as a general-`mu` replacement for
  Cotton,
- and an **adaptive-gamma** rule derived in Section 6.5,
  `gamma* approx 1 / (1 + c * kappa(C)^2 * (N/T) / IC^2)`.

All methods are evaluated under identical covariance estimates and signals so
that only the allocation rule varies.

---

## Repository layout

```
beyond_hrp_and_cotton/
├── code/        # one script per figure or computation
└── results/    # committed numerical artifacts used by the paper
```

### `code/`

Scripts are named by the section of the paper they feed:

| Prefix | Role | Paper section |
|---|---|---|
| `computeB_*` | Worked HRP example | Appendix B |
| `computeC_*`, `figC_*` | A1 sign pathology | Appendix C |
| `computeE_*` | A1 / A1-L1 robustness | Appendix E |
| `fig05_*` | Shrinkage-operator / Gauss-Seidel diagnostics | Section 5 |
| `fig06_*` | Bias-variance curves, adaptive-gamma trajectory | Section 6 |
| `compute09_*`, `fig09_*` | Adaptive gamma calibration, sweep regularization | Section 9 |
| `compute10_*`, `fig10_*`, `emit10_tables.py` | CRSP-analog backtest | Section 10 |

Every script writes its outputs under `results/` (or a section-specific
subdirectory such as `results/sec09_adaptive/` or `results/sec10/`) and is
runnable from the project root, e.g.

```bash
python code/compute09_adaptive_gamma.py
python code/fig09_sweep_heatmap.py
```

The `compute*` scripts produce CSV / NPZ data; the `fig*` scripts consume
those artifacts and emit the figures used in the LaTeX source.

> **Note on the `study` module.** Several scripts import primitives
> (`build_hrp_tree`, `hrp_flat_weights`, `cotton_weights`, `method_a3_weights`,
> `method_b_solve`, …) from a top-level `study.py` module. That module is the
> paper's shared allocation library; it is expected to sit alongside the
> `code/` and `results/` directories (the scripts add the project root to
> `sys.path` automatically).

### `results/`

Plain-text result snapshots are committed so the numbers quoted in the paper
can be checked directly:

| File | Content |
|---|---|
| `00_insample_direction_error_suite.txt` | Cotton / A1 / A2 / Method B comparison on the common `sum(w) = 1` scale |
| `01_walkforward_*` | Walk-forward sensitivity, random-`mu` and A1-L1 variants |
| `02_walkforward_*` | Walk-forward sensitivity and structural `mu` tests |
| `03_a1_deep_dive*` | A1 pathology deep dive (noiseless panel of Appendix C) |
| `04_minvar_mc_*` | Monte Carlo minimum-variance comparison (with and without Cotton / A1-L1) |
| `appendix_c_with_a1l1.txt` | Appendix C numbers including the A1-L1 variant |
| `appendix_e_a1l1_robustness.txt` | Appendix E robustness block |

Section 9 and Section 10 pipelines additionally populate
`results/sec09_*/` and `results/sec10/` when the corresponding `compute*`
scripts are run.

---

## Reproducing the paper

### Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`

### Quickstart

From the project root:

```bash
# Appendix B — worked HRP example
python code/computeB_hrp_example.py

# Appendix C — A1 pathology
python code/computeC_a1_pathology.py
python code/figC_a1_cosine_histogram.py

# Section 5 — shrinkage diagnostics
python code/fig05_shrinkage_schematic.py

# Section 6 — bias-variance and adaptive-gamma trajectory
python code/fig06_bias_variance_curves.py
python code/fig06_trajectory.py

# Section 9 — adaptive gamma calibration + sweep regularization
python code/compute09_adaptive_gamma.py
python code/compute09_sweep_regularization.py
python code/fig09_sweep_heatmap.py
python code/fig09_sweep_slices.py
python code/fig09_gamma_scatter.py
python code/fig09_plateau_width.py
python code/fig09_sharpe_vs_T.py
python code/compute09_table_numbers.py

# Section 10 — CRSP-analog backtest
python code/compute10_crsp_backtest.py
python code/compute10b_adaptive_gamma.py
python code/emit10_tables.py
python code/fig10_crsp_timeseries.py
python code/fig10_direction_err_timeseries.py
python code/fig10_gamma_t_series.py
python code/fig10_adaptive_rolling_diff.py
```

All scripts are seeded (typically seed `42`, or `20260411` for the Section 10
backtest) so results are bit-reproducible on a given `numpy` / BLAS stack.

### A note on CRSP data

Section 10's backtest is **self-contained**: because direct CRSP access was
not available in the writing environment, `compute10_crsp_backtest.py`
simulates a Russell-1000-analog universe from a 1-market + 10-sector factor
model (500 assets, monthly returns Jan 1995 – Dec 2024) and runs the full
backtest protocol on the synthetic data. The paper flags this explicitly; the
construction is designed so that allocation-rule differences are the only
thing that varies across methods.

---

## Citation

If you use this code or the results, please cite the paper:

```
Wuebben, B. J. (2026). Beyond HRP and Cotton: Hierarchical and Iterative
Methods for General Mean-Variance Portfolios.
```

---

## Contact

Bernd J. Wuebben — <wuebben@gmail.com>
