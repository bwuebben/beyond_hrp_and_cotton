"""
emit10_tables.py
================

Reads results/sec10/headline_sharpe.csv and results/sec10/robustness.csv
and prints the LaTeX table bodies used in Section 10. This is a small
helper so the tex file can be regenerated from the CSVs without
manual typing.

Usage:
    python figures/code/emit10_tables.py
"""

from __future__ import annotations

import os
import csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESDIR = os.path.join(ROOT, "results", "sec10")


METHOD_LATEX = {
    "1/N":             r"$1/N$",
    "Direct":          "Direct Markowitz",
    "LW-Markowitz":    "LW Markowitz",
    "HRP":             "HRP",
    "Cotton g=0.7":    r"Cotton $\gamma{=}0.7$",
    "A3 g=0.5":        r"HRP-$\mu$ $\gamma{=}0.5$",
    "A3 g=1.0":        r"HRP-$\mu$ $\gamma{=}1$",
    "B g=0.3 (100sw)": r"CRISP $\gamma{=}0.3$, 100 sw",
    "B g=0.5 (100sw)": r"CRISP $\gamma{=}0.5$, 100 sw",
    "B g=0.7 (100sw)": r"CRISP $\gamma{=}0.7$, 100 sw",
    "B g=1.0 (100sw)": r"CRISP $\gamma{=}1$, 100 sw",
    "B g=1.0 (500sw)": r"CRISP $\gamma{=}1$, 500 sw",
}

METHOD_ORDER = [
    "1/N",
    "HRP",
    "Cotton g=0.7",
    "Direct",
    "LW-Markowitz",
    "A3 g=0.5",
    "A3 g=1.0",
    "B g=0.3 (100sw)",
    "B g=0.5 (100sw)",
    "B g=0.7 (100sw)",
    "B g=1.0 (100sw)",
]


def read_headline():
    rows = list(csv.DictReader(open(os.path.join(RESDIR, "headline_sharpe.csv"))))
    # index[signal][method] = row
    idx = defaultdict(dict)
    for r in rows:
        idx[r["signal"]][r["method"]] = r
    return idx


def read_robust():
    rows = list(csv.DictReader(open(os.path.join(RESDIR, "robustness.csv"))))
    return rows


def read_diag():
    with open(os.path.join(RESDIR, "diagnostics.txt")) as fh:
        return fh.read()


def f2(x):   return f"{float(x):.2f}"
def f3(x):   return f"{float(x):.3f}"
def pct(x):  return f"{100.0*float(x):.1f}\\%"
def neg(x):  return f"{float(x):+.2f}"


def emit_headline():
    idx = read_headline()
    print("% -- headline Sharpe by signal x method ----------------------")
    print(r"""\begin{table}[t]
\centering
\caption{Headline net-of-cost annualised Sharpe ratios on the simulated
Russell-1000-analog universe, 1995--2024. Long-only baseline, 60-month
rolling estimation window, monthly rebalance, 10 bps/side transaction
cost. Best in each column in \textbf{bold}.}
\label{tab10:headline_sharpe}
\begin{adjustbox}{max width=\textwidth, center}
\small
\begin{tabular}{@{}l ccc ccc cccc@{}}
\toprule
 & \multicolumn{3}{c}{\textbf{Sharpe}}
 & \multicolumn{3}{c}{\textbf{Ann.\ return (\%)}}
 & \textbf{Ann.\ vol} & \textbf{Max DD} & \textbf{Turn/y} & \textbf{Avg $n$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & Mom.\ & Val.\ & Comp.\ & Mom.\ & Val.\ & Comp.\ & (\%) & (\%) & ($\times$) & (pos.) \\
\midrule""")

    # Determine best Sharpe per signal.
    bests = {}
    for sig in ("momentum", "value", "composite"):
        best = max(idx[sig].values(), key=lambda r: float(r["sharpe"]))
        bests[sig] = best["method"]

    for m in METHOD_ORDER:
        if m not in idx["composite"]:
            continue
        rm = idx["momentum"][m]; rv = idx["value"][m]; rc = idx["composite"][m]
        def sh(rr, sig):
            v = f2(rr["sharpe"])
            if bests[sig] == m:
                return r"\textbf{" + v + "}"
            return v
        name = METHOD_LATEX.get(m, m)
        print(f"{name} & "
              f"{sh(rm, 'momentum')} & {sh(rv, 'value')} & {sh(rc, 'composite')} & "
              f"{100*float(rm['ann_ret']):.1f} & "
              f"{100*float(rv['ann_ret']):.1f} & "
              f"{100*float(rc['ann_ret']):.1f} & "
              f"{100*float(rc['ann_vol']):.1f} & "
              f"{100*float(rc['mdd']):.1f} & "
              f"{float(rc['turnover_ann']):.1f} & "
              f"{float(rc['avg_npos']):.0f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")


def emit_robust():
    rows = read_robust()
    # Reduce each family to a (method x variant) pivot on Sharpe.
    per_family = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        per_family[r["family"]][r["variant"]][r["method"]] = r["sharpe"]

    print("% -- robustness table ---------------------------------------")
    print(r"""\begin{table}[t]
\centering
\caption{Section~10 robustness checks (composite signal, long-only).
Each cell reports the net-of-cost annualised Sharpe ratio under the
perturbation indicated. ``Base'' is the headline setting: sample
covariance, 60-month window, 10 bps/side, monthly rebalance. Best
in each row in \textbf{bold}.}
\label{tab10:robustness}
\begin{adjustbox}{max width=\textwidth, center}
\footnotesize
\begin{tabular}{@{}ll cccccc@{}}
\toprule
Family & Variant & $1/N$ & Direct & LW-M. & HRP & HRP-$\mu$ $\gamma{=}0.5$ & CRISP $\gamma{=}0.5$ \\
\midrule""")

    cols = ["1/N", "Direct", "LW-Markowitz", "HRP", "A3 g=0.5",
            "B g=0.5 (100sw)"]  # data keys stay, display via METHOD_LATEX

    idx = read_headline()
    # Base row from headline composite.
    base = {m: idx["composite"][m]["sharpe"] for m in cols if m in idx["composite"]}
    best = max(base, key=lambda m: float(base[m]))
    base_fmt = []
    for m in cols:
        v = f2(base.get(m, float('nan')))
        if m == best:
            v = r"\textbf{" + v + "}"
        base_fmt.append(v)
    print("Base & --- & " + " & ".join(base_fmt) + r" \\")

    for fam, label in [("CovEst", "Cov estimator"),
                       ("Window", "Window"),
                       ("TC",     "Txn cost"),
                       ("Subperiod","Subperiod")]:
        fam_rows = per_family.get(fam, {})
        for variant, mvals in fam_rows.items():
            row = []
            best_m = max(mvals, key=lambda m: float(mvals.get(m, -1e9))
                         if m in cols else -1e9)
            for m in cols:
                v = mvals.get(m)
                if v is None:
                    row.append("---")
                else:
                    s = f2(v)
                    if m == best_m:
                        s = r"\textbf{" + s + "}"
                    row.append(s)
            print(f"{label} & {variant} & " + " & ".join(row) + r" \\")

    # Sweep-count row uses the Sweeps family: just append as its own
    # mini-panel.
    sweeps = per_family.get("Sweeps", {})
    if sweeps:
        print(r"\midrule")
        print(r"\multicolumn{2}{l}{\textit{CRISP $\gamma{=}0.5$ sweep count}}"
              + " & & & & & & \\\\")
        for variant, mvals in sweeps.items():
            sh = list(mvals.values())[0]
            print(f"CRISP sweeps & {variant} & --- & --- & --- & --- & --- & "
                  f"{f2(sh)} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")


def emit_runtime():
    rt_path = os.path.join(RESDIR, "runtimes.csv")
    rows = list(csv.DictReader(open(rt_path)))
    print("% -- runtime (inline) ---------------------------------------")
    for r in rows:
        print(f"%   {r['method']:22s} {float(r['seconds_per_call'])*1000:8.1f} ms")


if __name__ == "__main__":
    emit_headline()
    print()
    emit_robust()
    print()
    emit_runtime()
