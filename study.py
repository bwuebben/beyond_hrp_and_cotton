"""
Comparative Study: Cotton, Method A1, Method A2, Method B
=========================================================

All weight vectors are normalized to sum to 1 (fully invested portfolios)
so that comparisons are on a common scale.

Cotton:    Schur complement augmentation, mu=1 only, O(N^3/6) for gamma>0
A1:        Recursive tree pass (bottom-up), normalized, O(N^2)
A2:        Flat IVP tree pass (De Prado recovery, top-down), O(N^2)
Method B:  Scalar Gauss-Seidel for P_gamma w = mu, O(pN^2)
"""

import numpy as np
import time
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


# ================================================================
# Tree
# ================================================================

class TreeNode:
    def __init__(self, indices, left=None, right=None):
        self.indices = indices
        self.left = left
        self.right = right
        self.is_leaf = (left is None and right is None)


def build_tree_from_linkage(Z, N):
    scipy_root = to_tree(Z)
    def _convert(node):
        if node.is_leaf():
            return TreeNode([node.id])
        left = _convert(node.get_left())
        right = _convert(node.get_right())
        return TreeNode(left.indices + right.indices, left, right)
    return _convert(scipy_root)


def build_hrp_tree(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method='ward')
    return build_tree_from_linkage(Z, cov.shape[0])


# ================================================================
# Normalization
# ================================================================

def normalize(w):
    """Normalize weights to sum to 1 (fully invested portfolio)."""
    s = w.sum()
    if abs(s) < 1e-15:
        return w  # degenerate: leave unchanged
    return w / s


# ================================================================
# Cotton: Schur Complementary Allocation (Section 2.2)
# mu=1 only.  O(N^3) for gamma > 0.
# ================================================================

def cotton_weights(cov, tree, gamma):
    """
    Cotton (2024).  At each node:
    1. Augment covariance via Schur complement: A^c = A - gamma B D^{-1} B^T
    2. Adjust signal: b_A = 1 - gamma B D^{-1} 1
    3. Compute fitness via linear solve: nu = 1/(b^T (A^c)^{-1} b)
    4. Split by inverse fitness, recurse with augmented sub-covariance.

    At gamma=0: A^c = A, b = 1, fitness = 1/(1^T A^{-1} 1) = min-var fitness.
    At gamma=1: recovers exact Sigma^{-1}1 (min-var).
    Only handles mu = 1.  Output sums to 1.
    """
    N = cov.shape[0]

    # Reorder covariance to tree leaf order so that at each node
    # the left child occupies the first n_L rows and right the rest.
    order = tree.indices
    cov_ord = cov[np.ix_(order, order)]

    def _recurse(node, sub_cov, sub_b):
        """
        sub_cov: covariance for this node's assets (in tree leaf order)
        sub_b: signal vector (starts as 1, adjusted by Schur complement)
        """
        if node.is_leaf:
            return np.array([1.0])

        n_L = len(node.left.indices)

        A = sub_cov[:n_L, :n_L]
        D = sub_cov[n_L:, n_L:]
        B = sub_cov[:n_L, n_L:]
        bL = sub_b[:n_L]
        bR = sub_b[n_L:]

        if gamma > 1e-14:
            # Schur complement augmentation (Eq. cotton_schur_gamma)
            D_inv_BT = np.linalg.solve(D, B.T)
            A_aug = A - gamma * B @ D_inv_BT
            b_A = bL - gamma * B @ np.linalg.solve(D, bR)

            A_inv_B = np.linalg.solve(A, B)
            D_aug = D - gamma * B.T @ A_inv_B
            b_D = bR - gamma * B.T @ np.linalg.solve(A, bL)
        else:
            A_aug = A
            D_aug = D
            b_A = bL
            b_D = bR

        # Solve augmented systems
        x_L = np.linalg.solve(A_aug, b_A)   # (A^c)^{-1} b_A
        x_R = np.linalg.solve(D_aug, b_D)   # (D^c)^{-1} b_D

        # Split by sum of solution vectors: Sigma^{-1}1 = [x_L; x_R],
        # so the fraction belonging to L is sum(x_L)/(sum(x_L)+sum(x_R)).
        sum_L = x_L.sum()
        sum_R = x_R.sum()
        alpha_L = sum_L / (sum_L + sum_R)
        alpha_R = 1.0 - alpha_L

        # Recurse with augmented sub-covariance and adjusted signal
        w_L = _recurse(node.left, A_aug, b_A)
        w_R = _recurse(node.right, D_aug, b_D)

        return np.concatenate([alpha_L * w_L, alpha_R * w_R])

    w_tree = _recurse(tree, cov_ord, np.ones(N))

    # Map back from tree leaf order to original order
    w = np.zeros(N)
    for i, idx in enumerate(order):
        w[idx] = w_tree[i]
    return w  # sums to 1


# ================================================================
# Method A1: Recursive Weights (Section 3.2)
# Bottom-up, normalized, O(N^2).
# ================================================================

def method_a1_weights(cov, mu, tree, gamma):
    """Variant A1: within-branch weights from sub-tree recursion.
    Output sums to 1."""
    N = len(mu)

    def _pass(node):
        if node.is_leaf:
            i = node.indices[0]
            return np.array([1.0]), cov[i, i], mu[i]

        w_L, v_L, s_L = _pass(node.left)
        w_R, v_R, s_R = _pass(node.right)

        idx_L = node.left.indices
        idx_R = node.right.indices
        c = w_L @ cov[np.ix_(idx_L, idx_R)] @ w_R

        det = v_L * v_R - (gamma * c) ** 2
        if abs(det) < 1e-15:
            aL = s_L / max(v_L, 1e-15)
            aR = s_R / max(v_R, 1e-15)
        else:
            aL = (v_R * s_L - gamma * c * s_R) / det
            aR = (v_L * s_R - gamma * c * s_L) / det

        a_sum = aL + aR
        if abs(a_sum) < 1e-15:
            aL, aR = 0.5, 0.5
        else:
            aL /= a_sum; aR /= a_sum

        w = np.concatenate([aL * w_L, aR * w_R])
        v = aL**2 * v_L + aR**2 * v_R + 2 * aL * aR * c
        s = aL * s_L + aR * s_R
        return w, v, s

    w_tree, _, _ = _pass(tree)
    w = np.zeros(N)
    for i, idx in enumerate(tree.indices):
        w[idx] = w_tree[i]
    return w  # already sums to 1


def method_a1_l1_weights(cov, mu, tree, gamma):
    """Variant A1 with L1 normalization.

    Identical to method_a1_weights except the budget-normalization step
    at each internal node divides by (|aL| + |aR|) rather than (aL + aR).
    This preserves signs (the divisor is strictly positive) and therefore
    eliminates A1's sign-flip pathology at the cost of giving up the
    sum-to-one interpretation at intermediate nodes.

    By scale invariance of the 2x2 system at each level, the *direction*
    of the final portfolio is the same mathematical object as A1's, just
    re-scaled.  The representative portfolio at every internal node
    satisfies ||w||_1 = 1, so intermediate computations stay on a clean
    numerical scale.

    Output satisfies ||w||_1 = 1 at the root.
    """
    N = len(mu)

    def _pass(node):
        if node.is_leaf:
            i = node.indices[0]
            return np.array([1.0]), cov[i, i], mu[i]

        w_L, v_L, s_L = _pass(node.left)
        w_R, v_R, s_R = _pass(node.right)

        idx_L = node.left.indices
        idx_R = node.right.indices
        c = w_L @ cov[np.ix_(idx_L, idx_R)] @ w_R

        det = v_L * v_R - (gamma * c) ** 2
        if abs(det) < 1e-15:
            aL = s_L / max(v_L, 1e-15)
            aR = s_R / max(v_R, 1e-15)
        else:
            aL = (v_R * s_L - gamma * c * s_R) / det
            aR = (v_L * s_R - gamma * c * s_L) / det

        # L1 normalization: divide by |aL| + |aR| (recursion invariant
        # maintains ||w_L||_1 = ||w_R||_1 = 1, so the combined vector's
        # L1 norm equals |aL| + |aR|).
        abs_sum = abs(aL) + abs(aR)
        if abs_sum < 1e-15:
            aL, aR = 0.5, 0.5
        else:
            aL /= abs_sum
            aR /= abs_sum

        w = np.concatenate([aL * w_L, aR * w_R])
        v = aL**2 * v_L + aR**2 * v_R + 2 * aL * aR * c
        s = aL * s_L + aR * s_R
        return w, v, s

    w_tree, _, _ = _pass(tree)
    w = np.zeros(N)
    for i, idx in enumerate(tree.indices):
        w[idx] = w_tree[i]
    return w  # ||w||_1 = 1


# ================================================================
# Method A2: Flat IVP Weights (Section 3.3)
# Top-down, De Prado recovery, O(N^2).
# ================================================================

def method_a2_weights(cov, mu, tree, gamma):
    """Variant A2: flat IVP weights at each node (De Prado recovery).
    Output sums to 1."""
    N = len(mu)
    w = np.zeros(N)

    def _recurse(node, budget):
        if node.is_leaf:
            w[node.indices[0]] = budget
            return

        idx_L = node.left.indices
        idx_R = node.right.indices

        def _flat_ivp(indices):
            d = np.array([1.0 / cov[i, i] for i in indices])
            return d / d.sum()

        wf_L = _flat_ivp(idx_L)
        wf_R = _flat_ivp(idx_R)

        v_L = wf_L @ cov[np.ix_(idx_L, idx_L)] @ wf_L
        v_R = wf_R @ cov[np.ix_(idx_R, idx_R)] @ wf_R
        s_L = wf_L @ mu[idx_L]
        s_R = wf_R @ mu[idx_R]
        c = wf_L @ cov[np.ix_(idx_L, idx_R)] @ wf_R

        det = v_L * v_R - (gamma * c) ** 2
        if abs(det) < 1e-10 * v_L * v_R:
            aL = s_L / max(v_L, 1e-15)
            aR = s_R / max(v_R, 1e-15)
        else:
            aL = (v_R * s_L - gamma * c * s_R) / det
            aR = (v_L * s_R - gamma * c * s_L) / det

        a_sum = aL + aR
        if abs(a_sum) < 1e-15:
            aL, aR = 0.5, 0.5
        else:
            aL /= a_sum; aR /= a_sum

        aL = np.clip(aL, -2.0, 3.0)
        aR = 1.0 - aL

        _recurse(node.left, budget * aL)
        _recurse(node.right, budget * aR)

    _recurse(tree, 1.0)
    return w  # already sums to 1


# ================================================================
# Method A3: Signed Flat IVP Weights
# Economic fix for A2 instability under mixed-sign mu.
# ================================================================

def method_a3_weights(cov, mu, tree, gamma, leaf_sign=True):
    """Variant A3: signed flat IVP weights at each node.

    At each internal node the within-branch 'representative portfolio'
    is the signed inverse-variance portfolio
        w_i^{signed} = sign(mu_i) * (1/sigma_i^2) / sum_j (1/sigma_j^2),
    rather than the unsigned flat IVP used by A2.  This makes the
    aggregate branch signal
        s_L = sum_i |mu_i|/sigma_i^2 / sum_j 1/sigma_j^2  >= 0,
    bounded away from zero whenever mu is not identically zero inside
    the branch, so the Cramer's rule 2x2 solve is not driven into its
    near-singular regime by sign cancellation.

    Because the new s_L is always non-negative, the 2x2 split alphas
    also stay non-negative at gamma = 0, which would otherwise yield a
    long-only portfolio.  To restore signed output we multiply each
    leaf weight by sign(mu_i) -- the cleanest way to carry the signal
    direction through the tree when leaf_sign=True (default).  The
    resulting construction is a hierarchical analog of cash-neutral
    risk parity: tree-derived magnitude, leaf-level signal direction.

    At mu = 1, sign(mu_i) = 1 everywhere, signed IVP reduces to flat
    IVP, leaf signs are all +1, so A3 reduces to A2 and nests De Prado
    HRP exactly."""
    N = len(mu)
    w = np.zeros(N)

    def _signed_ivp(indices):
        signs = np.sign(mu[indices])
        # Map zero-signal to +1 so an asset with mu_i = 0 stays in the
        # portfolio at its flat-IVP weight (long by default).
        signs = np.where(signs == 0.0, 1.0, signs)
        d = np.array([1.0 / cov[i, i] for i in indices])
        return signs * (d / d.sum())

    def _recurse(node, budget):
        if node.is_leaf:
            i = node.indices[0]
            if leaf_sign:
                sign_i = 1.0 if mu[i] >= 0 else -1.0
                w[i] = budget * sign_i
            else:
                w[i] = budget
            return

        idx_L = node.left.indices
        idx_R = node.right.indices

        wf_L = _signed_ivp(idx_L)
        wf_R = _signed_ivp(idx_R)

        v_L = wf_L @ cov[np.ix_(idx_L, idx_L)] @ wf_L
        v_R = wf_R @ cov[np.ix_(idx_R, idx_R)] @ wf_R
        s_L = wf_L @ mu[idx_L]
        s_R = wf_R @ mu[idx_R]
        c = wf_L @ cov[np.ix_(idx_L, idx_R)] @ wf_R

        det = v_L * v_R - (gamma * c) ** 2
        if abs(det) < 1e-10 * max(v_L * v_R, 1e-30):
            aL = s_L / max(v_L, 1e-15)
            aR = s_R / max(v_R, 1e-15)
        else:
            aL = (v_R * s_L - gamma * c * s_R) / det
            aR = (v_L * s_R - gamma * c * s_L) / det

        a_sum = aL + aR
        if abs(a_sum) < 1e-15:
            aL, aR = 0.5, 0.5
        else:
            aL /= a_sum
            aR /= a_sum

        _recurse(node.left, budget * aL)
        _recurse(node.right, budget * (1.0 - aL))

    _recurse(tree, 1.0)
    return w


# ================================================================
# HRP baseline (De Prado flat, for verification)
# ================================================================

def hrp_flat_weights(cov, tree):
    """De Prado's HRP.  For verification only."""
    N = cov.shape[0]
    def _recurse(node):
        if node.is_leaf:
            return np.array([1.0])
        idx_L = node.left.indices
        idx_R = node.right.indices
        def _cv(indices):
            if len(indices) == 1:
                return cov[indices[0], indices[0]]
            d = np.array([1.0 / cov[i, i] for i in indices])
            w = d / d.sum()
            return w @ cov[np.ix_(indices, indices)] @ w
        v_L, v_R = _cv(idx_L), _cv(idx_R)
        aL = (1.0/v_L) / (1.0/v_L + 1.0/v_R)
        w_L = _recurse(node.left)
        w_R = _recurse(node.right)
        return np.concatenate([aL * w_L, (1 - aL) * w_R])
    w_tree = _recurse(tree)
    w = np.zeros(N)
    for i, idx in enumerate(tree.indices):
        w[idx] = w_tree[i]
    return w


# ================================================================
# Method B: Gauss-Seidel for P_gamma w = mu (Section 4)
# ================================================================

def method_b_solve(cov, mu, gamma, max_sweeps=200, tol=1e-14,
                   w_init=None):
    """Returns raw (unnormalized) solution to P_gamma w = mu."""
    N = len(mu)
    diag = np.diag(cov)
    if gamma < 1e-14:
        return mu / diag
    w = w_init.copy() if w_init is not None else mu / diag
    for _ in range(max_sweeps):
        w_prev = w.copy()
        for i in range(N):
            off = cov[i, :] @ w - cov[i, i] * w[i]
            w[i] = (mu[i] - gamma * off) / diag[i]
        if np.linalg.norm(w - w_prev) / max(np.linalg.norm(w_prev), 1e-15) < tol:
            break
    return w


# ================================================================
# Synthetic data
# ================================================================

def make_structured_cov(N, n_sectors, rho_within, rho_cross, seed=42):
    rng = np.random.RandomState(seed)
    n_per = N // n_sectors
    corr = np.full((N, N), rho_cross)
    for s in range(n_sectors):
        idx = slice(s * n_per, (s + 1) * n_per)
        corr[idx, idx] = rho_within
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(0.15, 0.40, N)
    return np.outer(vols, vols) * corr, vols


# ================================================================
# Covariance/signal test suite (Experiment 7)
# ================================================================
# Each builder returns (cov, mu, label).  The suite stress-tests the
# direction-error metric across regimes where the diagonal solution
# mu/diag(Sigma) should differ substantially from Sigma^{-1}mu.

def _project_psd(corr, floor=1e-4):
    """Eigenvalue-clip then re-normalize correlation diagonal to 1."""
    w, V = np.linalg.eigh(corr)
    w = np.clip(w, floor, None)
    corr = (V * w) @ V.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def case_1_high_within(N=200, seed=42):
    """Tight blocks: rho_w=0.9, rho_c=0.3.  Each sector ~ one factor."""
    cov, _ = make_structured_cov(N, 5, 0.9, 0.3, seed=seed)
    rng = np.random.RandomState(seed + 1)
    mu = rng.randn(N) * 0.02
    return cov, mu, "Case 1: High within-sector (rho_w=0.9, rho_c=0.3)"


def case_2_equi(N=200, seed=42):
    """Near equi-correlation rho=0.9: one dominant eigenvalue."""
    rng = np.random.RandomState(seed)
    rho = 0.9
    corr = np.full((N, N), rho)
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(0.15, 0.40, N)
    cov = np.outer(vols, vols) * corr
    mu = rng.randn(N) * 0.02
    return cov, mu, "Case 2: Equi-correlation (rho=0.9)"


def case_3_factor(N=200, k=3, seed=42):
    """Factor model: Sigma = B diag(sigma_f^2) B^T + diag(idio)."""
    rng = np.random.RandomState(seed)
    B = rng.randn(N, k) * 0.3
    sigma_f = rng.uniform(0.10, 0.30, k) ** 2
    idio = rng.uniform(0.10, 0.30, N) ** 2
    cov = (B * sigma_f) @ B.T + np.diag(idio)
    mu = rng.randn(N) * 0.02
    return cov, mu, f"Case 3: Factor model (k={k} factors)"


def case_4_adversarial(N=200, seed=42):
    """Alternating-sign signal inside a tight sector.  The diagonal
    solution takes huge opposing bets inside the block; Sigma^{-1}mu
    must hedge them because within-sector rho is 0.9."""
    cov, _ = make_structured_cov(N, 5, 0.9, 0.2, seed=seed)
    rng = np.random.RandomState(seed + 1)
    mu = rng.randn(N) * 0.02
    n_per = N // 5
    mu[:n_per] = 0.02 * np.array([(-1.0) ** i for i in range(n_per)])
    return cov, mu, "Case 4: Adversarial alternating mu in tight sector"


def case_5_smallest_eig(N=200, seed=42):
    """mu nearly aligned with the smallest-eigenvalue direction of Sigma.
    Sigma^{-1}mu amplifies v_min by 1/lambda_min -- maximally far from
    mu/diag(Sigma).  This is the worst-case directional test."""
    cov, _ = make_structured_cov(N, 5, 0.7, 0.3, seed=seed)
    w, V = np.linalg.eigh(cov)
    v_min = V[:, 0]
    rng = np.random.RandomState(seed + 100)
    noise = rng.randn(N); noise /= np.linalg.norm(noise)
    mu_dir = 0.95 * v_min + 0.05 * noise
    mu = mu_dir / np.linalg.norm(mu_dir)
    return cov, mu, "Case 5: mu aligned with smallest eigenvector"


def case_6_wide_vol(N=200, seed=42):
    """Wide volatility spread sigma in [0.05, 1.0] (current is [0.15, 0.40])."""
    rng = np.random.RandomState(seed)
    n_sectors = 5
    n_per = N // n_sectors
    corr = np.full((N, N), 0.15)
    for s in range(n_sectors):
        idx = slice(s * n_per, (s + 1) * n_per)
        corr[idx, idx] = 0.6
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(0.05, 1.0, N)
    cov = np.outer(vols, vols) * corr
    mu = rng.randn(N) * 0.02
    return cov, mu, "Case 6: Wide vol spread sigma in [0.05, 1.0]"


def _kappa_corr(cov):
    """Condition number of the correlation matrix = kappa(D^{-1} Sigma).

    The symmetrized form D^{-1/2} Sigma D^{-1/2} is exactly the correlation
    matrix, so its eigenvalues (and those of D^{-1}Sigma via similarity) are
    the eigenvalues of corr(Sigma).  This is the *right* difficulty knob for
    the direction-error problem: when it's small, mu/diag(Sigma) and
    Sigma^{-1}mu are forced to be nearly parallel regardless of mu.
    """
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    w = np.linalg.eigvalsh(corr)
    w = np.clip(w, 1e-30, None)
    return w[-1] / w[0]


def worst_case_mu(cov, n_starts=40, seed=0):
    """Numerically find mu maximizing dir_err(mu/diag(Sigma), Sigma^{-1}mu).

    This is the direction-optimal adversarial signal: for a given Sigma,
    the mu that makes the diagonal solution as directionally far as possible
    from the Markowitz solution.  We parameterize mu on the unit sphere and
    use L-BFGS-B from multiple random starts.
    """
    from scipy.optimize import minimize
    N = cov.shape[0]
    D = np.diag(cov)
    Sinv = np.linalg.inv(cov)

    def neg_dir(x):
        nrm = np.linalg.norm(x)
        if nrm < 1e-30:
            return 0.0
        mu = x / nrm
        a = mu / D
        b = Sinv @ mu
        na2 = float(a @ a); nb2 = float(b @ b)
        if na2 < 1e-30 or nb2 < 1e-30:
            return 0.0
        c = float(a @ b)
        return -(1.0 - (c * c) / (na2 * nb2))

    rng = np.random.RandomState(seed)
    best_val = 0.0
    best_mu = None
    for _ in range(n_starts):
        x0 = rng.randn(N); x0 /= np.linalg.norm(x0)
        res = minimize(neg_dir, x0, method='L-BFGS-B',
                       options={'maxiter': 500})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_mu = res.x / np.linalg.norm(res.x)
    return best_mu, best_val


def case_5b_gen_smallest_eig(N=200, seed=42):
    """Corrected Case 5: mu aligned with the smallest *generalized*
    eigenvector of (Sigma, D), i.e., the smallest eigenvector of the
    correlation matrix D^{-1/2} Sigma D^{-1/2}.  This is the direction
    along which Sigma^{-1} acts maximally differently from D^{-1}."""
    cov, _ = make_structured_cov(N, 5, 0.7, 0.3, seed=seed)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    w, V = np.linalg.eigh(corr)
    v_min_corr = V[:, 0]  # smallest eigvec of correlation matrix
    # In the ordinary basis, the generalized eigvec is D^{1/2}-related:
    # if (D^{-1/2} S D^{-1/2}) v = lambda v, then S (D^{-1/2} v) = lambda D (D^{-1/2} v)
    # so the generalized eigvec of (S, D) is D^{-1/2} v.  We want mu such
    # that D^{-1}mu aligns with that generalized eigvec -- i.e., set
    # mu = D * (D^{-1/2} v_min_corr) = D^{1/2} v_min_corr.
    mu = d * v_min_corr  # d = sqrt(diag(cov))
    return cov, mu, "Case 5b: mu via smallest gen-eigvec of (Sigma, D)"


def case_8_worst_case_hedges(N=200, seed=42):
    """Case 7's hedges covariance, with numerically worst-case mu.
    This tests how adversarial mu can be against a fixed Sigma."""
    cov, _, _ = case_7_hedges(N, seed)
    mu, _ = worst_case_mu(cov, n_starts=30, seed=seed)
    return cov, mu, "Case 8: Case 7 Sigma + numerical worst-case mu"


def case_9_worst_case_highcond(N=200, seed=42):
    """Covariance with a deliberately high correlation-matrix condition
    number (built from a near-singular correlation structure) plus the
    numerically worst-case mu.  This should push B at gamma=0 toward
    the theoretical max direction error."""
    rng = np.random.RandomState(seed)
    # Build a correlation matrix with one near-zero eigenvalue:
    # start from an equi-correlation base at rho=0.95 (one huge eigval
    # + N-1 tiny eigvals), then add structured perturbation so it is
    # not literally rank-1.
    rho = 0.95
    corr = np.full((N, N), rho)
    np.fill_diagonal(corr, 1.0)
    # Add a small perturbation from a low-rank factor structure so the
    # second eigenvalue is also lifted (we want spread, not rank-1).
    B = rng.randn(N, 4) * 0.1
    corr = corr + 0.05 * (B @ B.T)
    corr = _project_psd(corr, floor=1e-5)
    vols = rng.uniform(0.15, 0.40, N)
    cov = np.outer(vols, vols) * corr
    mu, _ = worst_case_mu(cov, n_starts=30, seed=seed)
    return cov, mu, "Case 9: high-kappa(corr) Sigma + worst-case mu"


def case_7_hedges(N=200, seed=42):
    """Tight intra-sector blocks (rho=0.8) plus negative hedge pairs
    (rho=-0.6) across sectors 0 and 1.  Diagonal ignores the hedges;
    the inverse uses them."""
    rng = np.random.RandomState(seed)
    n_sectors = 5
    n_per = N // n_sectors
    corr = np.full((N, N), 0.1)
    for s in range(n_sectors):
        idx = slice(s * n_per, (s + 1) * n_per)
        corr[idx, idx] = 0.8
    for i in range(0, n_per, 2):
        j = n_per + i
        corr[i, j] = corr[j, i] = -0.6
    np.fill_diagonal(corr, 1.0)
    corr = _project_psd(corr, floor=1e-3)
    vols = rng.uniform(0.15, 0.40, N)
    cov = np.outer(vols, vols) * corr
    mu = rng.randn(N) * 0.02
    return cov, mu, "Case 7: Hedges (rho=-0.6) + tight blocks (rho=0.8)"


# ================================================================
# Helpers
# ================================================================

def rel_err(w, w_ref):
    return np.linalg.norm(w - w_ref) / max(np.linalg.norm(w_ref), 1e-15)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return a @ b / (na * nb)


def dir_err(w, w_ref):
    """Scale-invariant direction error: 1 - (w^T w*)^2 / (||w||^2 ||w*||^2).

    Equals sin^2(theta) where theta is the angle between w and w_ref.
    Zero iff w is collinear with w_ref (any sign, any scale).  The
    Markowitz problem determines only a ray in R^N -- normalization picks
    a point on that ray -- so this is the intrinsic quality measure.
    """
    na2 = float(w @ w)
    nb2 = float(w_ref @ w_ref)
    if na2 < 1e-30 or nb2 < 1e-30:
        return 0.0
    c = float(w @ w_ref)
    return max(0.0, 1.0 - (c * c) / (na2 * nb2))


def time_fn(fn, n_trials=3):
    t0 = time.perf_counter()
    for _ in range(n_trials):
        result = fn()
    return (time.perf_counter() - t0) / n_trials, result


# ================================================================
# Experiments
# ================================================================

def experiment_1_recovery():
    """Verify gamma=0, mu=1 endpoints."""
    print("=" * 72)
    print("Experiment 1: Recovery at gamma=0, mu=1")
    print("=" * 72)

    N = 100
    cov, _ = make_structured_cov(N, 5, 0.6, 0.15)
    tree = build_hrp_tree(cov)
    mu1 = np.ones(N)

    w_hrp = hrp_flat_weights(cov, tree)
    w_cotton0 = cotton_weights(cov, tree, gamma=0.0)
    w_a1 = method_a1_weights(cov, mu1, tree, gamma=0.0)
    w_a1l1 = method_a1_l1_weights(cov, mu1, tree, gamma=0.0)
    w_a2 = method_a2_weights(cov, mu1, tree, gamma=0.0)
    w_a3 = method_a3_weights(cov, mu1, tree, gamma=0.0)

    print(f"  All weights sum to 1.  Comparison on common scale.")
    print()
    print(f"  {'Comparison':40s} {'Rel Diff':>14s} {'Cos':>10s} {'Verdict':>10s}")
    print("  " + "-" * 78)
    for name, w in [("Cotton(g=0) vs HRP-flat", w_cotton0),
                    ("A2(g=0,mu=1) vs HRP-flat", w_a2),
                    ("A3(g=0,mu=1) vs HRP-flat", w_a3),
                    ("A1(g=0,mu=1) vs HRP-flat", w_a1),
                    ("A1-L1(g=0,mu=1) vs HRP-flat", w_a1l1)]:
        d = rel_err(w, w_hrp)
        cs = cosine_sim(w, w_hrp)
        print(f"  {name:40s} {d:14.2e} {cs:10.4f} "
              f"{'MATCH' if d < 1e-12 else 'CLOSE' if d < 0.05 else 'DIFFER':>10s}")

    d_a1_a2 = rel_err(w_a1, w_a2)
    print(f"  {'A1 vs A2':40s} {d_a1_a2:14.2e} "
          f"{cosine_sim(w_a1, w_a2):10.4f} "
          f"{'MATCH' if d_a1_a2 < 1e-12 else 'DIFFER':>10s}")
    print()


def experiment_2_minvar():
    """mu=1 comparison, all normalized to sum=1."""
    print("=" * 72)
    print("Experiment 2: Minimum Variance (mu=1), gamma sweep")
    print("=" * 72)

    N = 200
    cov, _ = make_structured_cov(N, 5, 0.6, 0.15)
    tree = build_hrp_tree(cov)
    mu1 = np.ones(N)
    w_exact = normalize(np.linalg.solve(cov, mu1))

    print(f"  All weights normalized to sum=1.  N={N}")
    print(f"  Reporting BOTH scale-sensitive relative error  rel = ||w-w*||/||w*||")
    print(f"  AND scale-invariant direction error  dir = 1 - cos^2(theta).")
    print()
    print(f"  {'gamma':>6s} {'metric':>8s} {'Cotton':>12s} {'A1':>12s} "
          f"{'A1-L1':>12s} {'A2':>12s} {'B(50sw)':>12s} {'B(200sw)':>12s}")
    print("  " + "-" * 88)

    for g in [0.0, 0.3, 0.5, 0.7, 1.0]:
        w_c = cotton_weights(cov, tree, g)
        w_a1 = method_a1_weights(cov, mu1, tree, g)
        w_a1l1 = method_a1_l1_weights(cov, mu1, tree, g)
        w_a2 = method_a2_weights(cov, mu1, tree, g)
        w_b50 = normalize(method_b_solve(cov, mu1, g, max_sweeps=50))
        w_b200 = normalize(method_b_solve(cov, mu1, g, max_sweeps=200))

        print(f"  {g:6.1f} {'rel':>8s} {rel_err(w_c, w_exact):12.4e} "
              f"{rel_err(w_a1, w_exact):12.4e} "
              f"{rel_err(w_a1l1, w_exact):12.4e} "
              f"{rel_err(w_a2, w_exact):12.4e} "
              f"{rel_err(w_b50, w_exact):12.4e} "
              f"{rel_err(w_b200, w_exact):12.4e}")
        print(f"  {'':>6s} {'dir':>8s} {dir_err(w_c, w_exact):12.4e} "
              f"{dir_err(w_a1, w_exact):12.4e} "
              f"{dir_err(w_a1l1, w_exact):12.4e} "
              f"{dir_err(w_a2, w_exact):12.4e} "
              f"{dir_err(w_b50, w_exact):12.4e} "
              f"{dir_err(w_b200, w_exact):12.4e}")

    print()


def experiment_3_general_mu():
    """General mu comparison, all normalized to sum=1."""
    print("=" * 72)
    print("Experiment 3: General Signal (mu ~ N(0,0.02^2)), gamma sweep")
    print("=" * 72)

    N = 200
    cov, _ = make_structured_cov(N, 5, 0.6, 0.15)
    tree = build_hrp_tree(cov)
    np.random.seed(42)
    mu = np.random.randn(N) * 0.02
    w_exact = normalize(np.linalg.solve(cov, mu))

    print(f"  All weights normalized to sum=1.  N={N}")
    print(f"  Relative error ||w - w*|| / ||w*||")
    print()
    print(f"  A2 omitted: unstable for general mu at gamma > 0.")
    print(f"  Reporting BOTH rel = ||w-w*||/||w*|| and dir = 1 - cos^2(theta).")
    print()
    print(f"  {'gamma':>6s} {'metric':>8s} {'A1':>12s} {'A1-L1':>12s} "
          f"{'A3':>12s} {'B(20sw)':>12s} {'B(50sw)':>12s} {'B(200sw)':>12s}")
    print("  " + "-" * 84)

    for g in [0.0, 0.3, 0.5, 0.7, 1.0]:
        w_a1 = method_a1_weights(cov, mu, tree, g)
        w_a1l1 = method_a1_l1_weights(cov, mu, tree, g)
        w_a3 = method_a3_weights(cov, mu, tree, g)
        w_b20 = normalize(method_b_solve(cov, mu, g, max_sweeps=20))
        w_b50 = normalize(method_b_solve(cov, mu, g, max_sweeps=50))
        w_b200 = normalize(method_b_solve(cov, mu, g, max_sweeps=200))

        print(f"  {g:6.1f} {'rel':>8s} {rel_err(w_a1, w_exact):12.4e} "
              f"{rel_err(w_a1l1, w_exact):12.4e} "
              f"{rel_err(w_a3, w_exact):12.4e} "
              f"{rel_err(w_b20, w_exact):12.4e} "
              f"{rel_err(w_b50, w_exact):12.4e} "
              f"{rel_err(w_b200, w_exact):12.4e}")
        print(f"  {'':>6s} {'dir':>8s} {dir_err(w_a1, w_exact):12.4e} "
              f"{dir_err(w_a1l1, w_exact):12.4e} "
              f"{dir_err(w_a3, w_exact):12.4e} "
              f"{dir_err(w_b20, w_exact):12.4e} "
              f"{dir_err(w_b50, w_exact):12.4e} "
              f"{dir_err(w_b200, w_exact):12.4e}")

    print()


def experiment_4_cosine():
    """Pairwise cosine similarity (scale-free)."""
    print("=" * 72)
    print("Experiment 4: Pairwise Cosine Similarity (N=200, gamma=0.7)")
    print("=" * 72)

    N = 200
    cov, _ = make_structured_cov(N, 5, 0.6, 0.15)
    tree = build_hrp_tree(cov)
    np.random.seed(42)
    mu = np.random.randn(N) * 0.02
    g = 0.7

    mu1 = np.ones(N)
    methods = {
        'Markowitz': normalize(np.linalg.solve(cov, mu)),
        'Min-var': normalize(np.linalg.solve(cov, mu1)),
        'Cotton': cotton_weights(cov, tree, g),
        'A1(mu)': method_a1_weights(cov, mu, tree, g),
        'A1L1(mu)': method_a1_l1_weights(cov, mu, tree, g),
        'A3(mu)': method_a3_weights(cov, mu, tree, g),
        'B(mu)': normalize(method_b_solve(cov, mu, g, max_sweeps=200)),
        'A1(1)': method_a1_weights(cov, mu1, tree, g),
        'A1L1(1)': method_a1_l1_weights(cov, mu1, tree, g),
        'A2(1)': method_a2_weights(cov, mu1, tree, g),
    }

    names = list(methods.keys())
    n = len(names)

    hdr = f"  {'':12s}" + "".join(f"{nm:>12s}" for nm in names)
    print(hdr)
    print("  " + "-" * (12 + 12 * n))

    for ni in names:
        row = f"  {ni:12s}"
        for nj in names:
            row += f"{cosine_sim(methods[ni], methods[nj]):12.4f}"
        print(row)
    print()


def experiment_5_runtime():
    """Runtime scaling across N."""
    print("=" * 72)
    print("Experiment 5: Runtime Scaling (gamma=0.7)")
    print("=" * 72)

    print(f"  {'N':>6s} {'Direct':>10s} {'Cotton':>10s} "
          f"{'A1':>10s} {'A1-L1':>10s} {'A2':>10s} {'B(20sw)':>10s}")
    print("  " + "-" * 70)

    for N in [100, 200, 500, 1000, 2000, 5000, 10000]:
        cov, _ = make_structured_cov(N, 5, 0.6, 0.15, seed=42)
        tree = build_hrp_tree(cov)
        mu1 = np.ones(N)
        g = 0.7

        t_direct, _ = time_fn(lambda: np.linalg.solve(cov, mu1))

        if N <= 2000:
            t_cotton, _ = time_fn(lambda: cotton_weights(cov, tree, g))
            s_cotton = f"{t_cotton:10.4f}"
        else:
            s_cotton = f"{'---':>10s}"

        t_a1, _ = time_fn(lambda: method_a1_weights(cov, mu1, tree, g))
        t_a1l1, _ = time_fn(lambda: method_a1_l1_weights(cov, mu1, tree, g))
        t_a2, _ = time_fn(lambda: method_a2_weights(cov, mu1, tree, g))
        t_b, _ = time_fn(lambda: method_b_solve(cov, mu1, g, max_sweeps=20))

        print(f"  {N:6d} {t_direct:10.4f} {s_cotton} "
              f"{t_a1:10.4f} {t_a1l1:10.4f} {t_a2:10.4f} {t_b:10.4f}")

    print()
    print("  Times in seconds.  Cotton is O(N^3/6); A1/A2 are O(N^2);")
    print("  Method B is O(p*N^2) with p=20 sweeps.")
    print()


def experiment_6_weight_properties():
    """Weight properties, all normalized to sum=1."""
    print("=" * 72)
    print("Experiment 6: Weight Properties (N=200, gamma=0.7, sum=1)")
    print("=" * 72)

    N = 200
    cov, _ = make_structured_cov(N, 5, 0.6, 0.15)
    tree = build_hrp_tree(cov)
    np.random.seed(42)
    mu = np.random.randn(N) * 0.02
    g = 0.7

    mu1 = np.ones(N)
    methods = [
        ('Markowitz', normalize(np.linalg.solve(cov, mu))),
        ('Min-var', normalize(np.linalg.solve(cov, mu1))),
        ('Cotton', cotton_weights(cov, tree, g)),
        ('A1(mu)', method_a1_weights(cov, mu, tree, g)),
        ('A1-L1(mu)', method_a1_l1_weights(cov, mu, tree, g)),
        ('A3(mu)', method_a3_weights(cov, mu, tree, g)),
        ('A1(mu=1)', method_a1_weights(cov, mu1, tree, g)),
        ('A1-L1(mu=1)', method_a1_l1_weights(cov, mu1, tree, g)),
        ('A2(mu=1)', method_a2_weights(cov, mu1, tree, g)),
        ('B(mu)', normalize(method_b_solve(cov, mu, g, max_sweeps=200))),
    ]

    print(f"  All weights normalized to sum=1.")
    print()
    print(f"  {'Method':16s} {'sum(w)':>8s} {'min(w)':>10s} "
          f"{'max(w)':>10s} {'||w||':>10s} {'gross':>10s} {'#neg':>6s}")
    print("  " + "-" * 74)

    for name, w in methods:
        gross = np.sum(np.abs(w))
        print(f"  {name:16s} {w.sum():8.4f} {w.min():10.4f} "
              f"{w.max():10.4f} {np.linalg.norm(w):10.4f} "
              f"{gross:10.4f} {np.sum(w < -1e-8):6d}")

    print()
    print("  gross = sum|w_i| (gross leverage).  1.0 = long-only.")
    print()


def experiment_7_covariance_suite():
    """Direction error across diverse covariance/signal regimes.

    Stress-tests whether the diagonal solution mu/diag(Sigma) is still
    directionally close to Sigma^{-1}mu.  When it is, the covariance
    regime fails to discriminate methods; when it is not, we see the
    full spread between diagonal, tree-based, and iterative solvers.
    """
    print("=" * 78)
    print("Experiment 7: Direction Error Across Covariance Regimes (N=200)")
    print("=" * 78)
    print()
    print("  Metric: dir(w, w*) = 1 - cos^2(angle) = sin^2(angle).")
    print("  Diagnostics per case:")
    print("    kappa(S)   = condition number of Sigma (includes vol spread)")
    print("    kappa(corr)= cond. number of corr(Sigma) = kappa(D^{-1}Sigma)")
    print("                 -- the *right* difficulty knob: small kappa(corr)")
    print("                 forces mu/diag and Sigma^{-1}mu to be parallel")
    print("                 regardless of mu.")
    print("    dir_diag   = dir(mu/diag(Sigma), Sigma^{-1}mu)  [problem difficulty]")
    print("    |cos_vmin| = |cos angle between mu and smallest eigvec of Sigma|")
    print()

    cases = [
        case_1_high_within(),
        case_2_equi(),
        case_3_factor(),
        case_4_adversarial(),
        case_5_smallest_eig(),
        case_5b_gen_smallest_eig(),
        case_6_wide_vol(),
        case_7_hedges(),
        case_8_worst_case_hedges(),
        case_9_worst_case_highcond(),
    ]

    gammas = [0.0, 0.3, 0.5, 0.7, 1.0]

    for cov, mu, label in cases:
        N = len(mu)
        tree = build_hrp_tree(cov)
        w_exact_raw = np.linalg.solve(cov, mu)
        w_exact = normalize(w_exact_raw)

        # Diagnostics
        kappa = np.linalg.cond(cov)
        kappa_c = _kappa_corr(cov)
        w_diag = mu / np.diag(cov)
        diff_diag = dir_err(w_diag, w_exact_raw)
        w_eig, V_eig = np.linalg.eigh(cov)
        v_min = V_eig[:, 0]
        cos_mu_vmin = abs(float(mu @ v_min)) / (
            np.linalg.norm(mu) * np.linalg.norm(v_min) + 1e-30)

        print("-" * 78)
        print(label)
        print(f"  kappa(S)={kappa:.2e}  kappa(corr)={kappa_c:.2e}  "
              f"dir_diag={diff_diag:.4f}  |cos_vmin|={cos_mu_vmin:.4f}")
        print()
        print(f"  {'gamma':>6s} {'A1':>14s} {'A1-L1':>14s} "
              f"{'A3':>14s} {'A2':>14s} "
              f"{'B(20sw)':>14s} {'B(200sw)':>14s}")
        print("  " + "-" * 98)

        for g in gammas:
            w_a1 = method_a1_weights(cov, mu, tree, g)
            w_a1l1 = method_a1_l1_weights(cov, mu, tree, g)
            w_a3 = method_a3_weights(cov, mu, tree, g)
            try:
                w_a2 = method_a2_weights(cov, mu, tree, g)
                if not np.all(np.isfinite(w_a2)):
                    raise ValueError
                a2_d = dir_err(w_a2, w_exact)
                a2_str = f"{a2_d:14.4e}"
            except Exception:
                a2_str = f"{'---':>14s}"
            w_b20 = normalize(method_b_solve(cov, mu, g, max_sweeps=20))
            w_b200 = normalize(method_b_solve(cov, mu, g, max_sweeps=200))
            print(f"  {g:6.1f} {dir_err(w_a1, w_exact):14.4e} "
                  f"{dir_err(w_a1l1, w_exact):14.4e} "
                  f"{dir_err(w_a3, w_exact):14.4e} "
                  f"{a2_str} "
                  f"{dir_err(w_b20, w_exact):14.4e} "
                  f"{dir_err(w_b200, w_exact):14.4e}")
        print()


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    print()
    print("Comparative Study: Cotton, A1, A2, Method B")
    print("All portfolios normalized to sum(w) = 1")
    print("=" * 72)
    print()

    experiment_1_recovery()
    experiment_2_minvar()
    experiment_3_general_mu()
    experiment_4_cosine()
    experiment_5_runtime()
    experiment_6_weight_properties()
    experiment_7_covariance_suite()
