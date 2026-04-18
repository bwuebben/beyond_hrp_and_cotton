"""
computeB_hrp_example.py

Self-contained worked example for Appendix B.  Builds a 4-asset
sector-structured covariance matrix (two clusters of two assets),
runs De Prado's recursive-bisection HRP, and prints every
intermediate quantity needed by the appendix text:

    - volatilities and correlations
    - covariance matrix
    - correlation distance matrix
    - Ward linkage Z
    - tree leaf order (quasi-diagonalization)
    - cluster variances via flat IVP (at root and at each child)
    - root split alpha and child split alphas
    - final HRP weights

Usage:
    python figures/code/computeB_hrp_example.py

Numbers are printed to three decimal places throughout so the output
can be copied verbatim into the LaTeX appendix.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from study import build_hrp_tree, hrp_flat_weights  # noqa: E402


def fmt_mat(M: np.ndarray, nd: int = 3) -> str:
    rows = []
    for row in M:
        rows.append("  " + "  ".join(f"{v:+.{nd}f}" for v in row))
    return "\n".join(rows)


def fmt_vec(v: np.ndarray, nd: int = 3) -> str:
    return "  ".join(f"{x:+.{nd}f}" for x in v)


def cluster_var_ivp(cov: np.ndarray, indices: list[int]) -> float:
    """Flat IVP cluster variance (De Prado)."""
    if len(indices) == 1:
        return float(cov[indices[0], indices[0]])
    d = np.array([1.0 / cov[i, i] for i in indices])
    w = d / d.sum()
    sub = cov[np.ix_(indices, indices)]
    return float(w @ sub @ w)


def main() -> None:
    # ------------------------------------------------------------------
    # Step 0.  Inputs.
    # 4 assets in two sectors: A1,A2 (sector 1) and A3,A4 (sector 2).
    # Volatilities chosen to be moderately heterogeneous.
    # ------------------------------------------------------------------
    names = ["A1", "A2", "A3", "A4"]
    sigma = np.array([0.20, 0.25, 0.30, 0.15])

    # Within-sector correlation 0.80, cross-sector 0.20.
    rho_within = 0.80
    rho_cross = 0.20
    corr = np.array([
        [1.00,       rho_within, rho_cross,  rho_cross],
        [rho_within, 1.00,       rho_cross,  rho_cross],
        [rho_cross,  rho_cross,  1.00,       rho_within],
        [rho_cross,  rho_cross,  rho_within, 1.00],
    ])
    cov = corr * np.outer(sigma, sigma)

    print("=" * 72)
    print("Appendix B: HRP worked example (4 assets)")
    print("=" * 72)

    print("\nAsset names        :", names)
    print("Volatilities sigma :", fmt_vec(sigma))
    print("rho_within         : {:.2f}".format(rho_within))
    print("rho_cross          : {:.2f}".format(rho_cross))

    print("\nCorrelation matrix C:")
    print(fmt_mat(corr))

    print("\nCovariance matrix Sigma = diag(sigma) C diag(sigma):")
    print(fmt_mat(cov))

    # ------------------------------------------------------------------
    # Step 1.  Correlation distance and Ward linkage.
    # ------------------------------------------------------------------
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    print("\nCorrelation distance d_ij = sqrt((1 - C_ij) / 2):")
    print(fmt_mat(dist))

    Z = linkage(squareform(dist, checks=False), method="ward")
    print("\nWard linkage Z (each row: [left, right, height, size]):")
    for row in Z:
        print("  [{:.0f} {:.0f} {:.3f} {:.0f}]".format(*row))

    # ------------------------------------------------------------------
    # Step 2.  Build tree and read off the quasi-diagonal leaf order.
    # ------------------------------------------------------------------
    tree = build_hrp_tree(cov)
    order = tree.indices
    print("\nLeaf order (quasi-diagonalization):",
          [names[i] for i in order], "  indices:", order)

    cov_ord = cov[np.ix_(order, order)]
    print("\nReordered covariance Sigma[order, order]:")
    print(fmt_mat(cov_ord))

    # ------------------------------------------------------------------
    # Step 3.  Recursive bisection, printed explicitly.
    # ------------------------------------------------------------------
    left_idx = tree.left.indices
    right_idx = tree.right.indices
    print("\nRoot split: left =",
          [names[i] for i in left_idx],
          "  right =", [names[i] for i in right_idx])

    v_L_root = cluster_var_ivp(cov, left_idx)
    v_R_root = cluster_var_ivp(cov, right_idx)
    alpha_root = (1.0 / v_L_root) / (1.0 / v_L_root + 1.0 / v_R_root)
    print(f"  v_L (flat IVP) = {v_L_root:.6f}")
    print(f"  v_R (flat IVP) = {v_R_root:.6f}")
    print(f"  alpha_root     = (1/v_L)/(1/v_L + 1/v_R) = {alpha_root:.3f}")
    print(f"  1 - alpha_root = {1 - alpha_root:.3f}")

    # Left child split (two leaves A,B: IVP closed form).
    def leaf_split(indices):
        i, j = indices
        sig_i2 = cov[i, i]
        sig_j2 = cov[j, j]
        a = (1.0 / sig_i2) / (1.0 / sig_i2 + 1.0 / sig_j2)
        return a, sig_i2, sig_j2

    a_L, siL_i2, siL_j2 = leaf_split(left_idx)
    a_R, siR_i2, siR_j2 = leaf_split(right_idx)
    print(f"\nLeft child split ({names[left_idx[0]]} vs {names[left_idx[1]]}):")
    print(f"  sigma^2_{names[left_idx[0]]} = {siL_i2:.6f}, "
          f"sigma^2_{names[left_idx[1]]} = {siL_j2:.6f}")
    print(f"  alpha_L = {a_L:.3f},  1 - alpha_L = {1 - a_L:.3f}")

    print(f"\nRight child split ({names[right_idx[0]]} vs {names[right_idx[1]]}):")
    print(f"  sigma^2_{names[right_idx[0]]} = {siR_i2:.6f}, "
          f"sigma^2_{names[right_idx[1]]} = {siR_j2:.6f}")
    print(f"  alpha_R = {a_R:.3f},  1 - alpha_R = {1 - a_R:.3f}")

    # ------------------------------------------------------------------
    # Step 4.  Multiply down and compare with study.hrp_flat_weights.
    # ------------------------------------------------------------------
    w = np.zeros(4)
    w[left_idx[0]] = alpha_root * a_L
    w[left_idx[1]] = alpha_root * (1.0 - a_L)
    w[right_idx[0]] = (1.0 - alpha_root) * a_R
    w[right_idx[1]] = (1.0 - alpha_root) * (1.0 - a_R)

    print("\nFinal HRP weights (manual multiplication):")
    for i, nm in enumerate(names):
        print(f"  w[{nm}] = {w[i]:.3f}")
    print(f"  sum    = {w.sum():.3f}")

    w_ref = hrp_flat_weights(cov, tree)
    print("\nReference: study.hrp_flat_weights:")
    for i, nm in enumerate(names):
        print(f"  w[{nm}] = {w_ref[i]:.3f}")
    print(f"  sum    = {w_ref.sum():.3f}")

    assert np.allclose(w, w_ref, atol=1e-9), "manual vs study.py disagree"
    print("\nOK: manual recursion matches study.hrp_flat_weights.")

    # ------------------------------------------------------------------
    # HRP-mu extension: add a mixed-sign alpha signal.
    # ------------------------------------------------------------------
    from study import method_a3_weights  # noqa: E402

    mu = np.array([+0.03, -0.01, +0.02, -0.04])
    gamma = 0.5

    print("\n" + "=" * 72)
    print(f"HRP-mu extension (gamma = {gamma})")
    print("=" * 72)
    print(f"\nSignal mu = [{', '.join(f'{m:+.2f}' for m in mu)}]")

    # Signed IVP at root children.
    def signed_ivp(indices):
        signs = np.array([1.0 if mu[i] >= 0 else -1.0 for i in indices])
        d = np.array([1.0 / cov[i, i] for i in indices])
        return signs * (d / d.sum())

    wf_L = signed_ivp(left_idx)
    wf_R = signed_ivp(right_idx)
    print(f"\nSigned IVP (left  {[names[i] for i in left_idx]}): "
          f"[{', '.join(f'{x:+.4f}' for x in wf_L)}]")
    print(f"Signed IVP (right {[names[i] for i in right_idx]}): "
          f"[{', '.join(f'{x:+.4f}' for x in wf_R)}]")

    v_L = float(wf_L @ cov[np.ix_(left_idx, left_idx)] @ wf_L)
    v_R = float(wf_R @ cov[np.ix_(right_idx, right_idx)] @ wf_R)
    s_L = float(wf_L @ mu[left_idx])
    s_R = float(wf_R @ mu[right_idx])
    c_val = float(wf_L @ cov[np.ix_(left_idx, right_idx)] @ wf_R)
    print(f"\nCluster stats: v_L={v_L:.5f}  v_R={v_R:.5f}  "
          f"s_L={s_L:+.4f}  s_R={s_R:+.4f}  c={c_val:+.5f}")

    det = v_L * v_R - (gamma * c_val) ** 2
    aL = (v_R * s_L - gamma * c_val * s_R) / det
    aR = (v_L * s_R - gamma * c_val * s_L) / det
    aL_n = aL / (aL + aR)
    aR_n = 1.0 - aL_n
    print(f"Root split: alpha_L={aL_n:.4f}  alpha_R={aR_n:.4f}")

    # Leaf splits (single-asset nodes).
    for label, idx_pair in [("left", left_idx), ("right", right_idx)]:
        i, j = idx_pair
        si = abs(mu[i])
        sj = abs(mu[j])
        vi = cov[i, i]
        vj = cov[j, j]
        sgn_i = 1.0 if mu[i] >= 0 else -1.0
        sgn_j = 1.0 if mu[j] >= 0 else -1.0
        cc = sgn_i * cov[i, j] * sgn_j
        dd = vi * vj - (gamma * cc) ** 2
        ai = (vj * si - gamma * cc * sj) / dd
        aj = (vi * sj - gamma * cc * si) / dd
        ai_n = ai / (ai + aj)
        aj_n = 1.0 - ai_n
        print(f"  {label} child ({names[i]} vs {names[j]}): "
              f"alpha={ai_n:.4f}, {aj_n:.4f}")

    # Final weights via study.method_a3_weights.
    w_a3 = method_a3_weights(cov, mu, tree, gamma)
    print(f"\nHRP-mu weights (gamma={gamma}):")
    for i, nm in enumerate(names):
        print(f"  w[{nm}] = {w_a3[i]:+.4f}")
    print(f"  sum = {w_a3.sum():+.4f}")

    sh = (w_a3 @ mu) / np.sqrt(w_a3 @ cov @ w_a3)
    sh_hrp = (w_ref @ mu) / np.sqrt(w_ref @ cov @ w_ref)
    w_mk = np.linalg.solve(cov, mu)
    sh_mk = (w_mk @ mu) / np.sqrt(w_mk @ cov @ w_mk)
    print(f"\nSharpe: HRP-mu={sh:.4f}  HRP={sh_hrp:.4f}  Markowitz={sh_mk:.4f}")

    # Recovery check: gamma=0, mu=1 => HRP.
    w_a3_g0 = method_a3_weights(cov, np.ones(4), tree, 0.0)
    assert np.allclose(w_a3_g0, w_ref, atol=1e-9), \
        "HRP-mu at gamma=0 mu=1 does not match HRP"
    print("OK: HRP-mu at gamma=0, mu=1 matches HRP (Prop. A3 recovery).")


if __name__ == "__main__":
    main()
