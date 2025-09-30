# MIT License
#
# Copyright (c) 2024 Vincent Guan, Joseph Janssen, Hossein Rahmani, Andrew Warren, Stephen Zhang, Elina Robeva, Geoffrey Schiebinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from itertools import product
from typing import Callable, Union


def extract_marginal_samples(X, shuffle=True):
    """
    Extract marginal distributions per time from measured population snapshots

    Parameters:
        X (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).
        shuffle: whether to shuffle the time marginals (X should already break dependencies between trajectories)

    Returns:
        list of numpy.ndarray: Each element is an array containing samples from the marginal distribution at each time step.
    """
    num_trajectories, num_steps, d = X.shape
    marginal_samples = []

    for t in range(num_steps):
        # Extract all samples at time t from each trajectory
        samples_at_t = X[:, t, :]
        if shuffle:
            samples_at_t_copy = samples_at_t.copy()
            np.random.shuffle(samples_at_t_copy)
            marginal_samples.append(samples_at_t_copy)
        else:
            marginal_samples.append(samples_at_t)
    return marginal_samples

# ----------------------------------------------------------------------
#  Robust trajectory normaliser
# ----------------------------------------------------------------------
def _normalise_trajectory(ds) -> np.ndarray:
    raw = ds.trajectory

    # ---- NEW: handle dict of snapshots ----
    if isinstance(raw, dict):
        times = sorted(raw.keys())
        mats = [np.asarray(raw[t]) for t in times]  # shapes: (n_t, d)
        n_min = min(m.shape[0] for m in mats)
        if any(m.shape[0] != n_min for m in mats):
            # downsample uniformly to equalize counts
            rng = np.random.default_rng(0)
            mats = [m[rng.choice(m.shape[0], n_min, replace=False)] for m in mats]
        X = np.stack(mats, axis=0)  # (T, N, d)
        X = np.transpose(X, (1, 0, 2))  # (N, T, d)
        print("[SB]   stacked from dict →", X.shape)
        return X
    # --------------------------------------

    # existing unwrap branches...
    try:
        import jax
        if isinstance(raw, jax.Array):
            raw = np.asarray(raw)
    except Exception:
        pass
    depth = 0
    while isinstance(raw, np.ndarray) and raw.ndim == 0:
        depth += 1
        raw = raw.item()
    if isinstance(raw, list):
        raw = np.stack([np.asarray(a) for a in raw], axis=0)  # (n_steps, N, d)
    X = np.asarray(raw)
    d = ds.data_dim
    if X.ndim == 2:
        X = X[None, ...]
    elif X.ndim == 3 and X.shape[0] < X.shape[1]:
        X = np.transpose(X, (1, 0, 2))
    print("[SB]   final trajectory shape:", X.shape)
    return X

def _safe_row_choice(p: np.ndarray, n: int) -> int:
    """Return a valid integer index, falling back to uniform if p is invalid."""
    if p is None or p.ndim != 1 or p.shape[0] != n:
        return int(np.random.randint(n))
    if not np.all(np.isfinite(p)):
        return int(np.random.randint(n))
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(n))
    p = p / s
    # numerical clean-up
    p = np.clip(p, 0.0, 1.0)
    p = p / float(p.sum())
    return int(np.random.choice(n, p=p))

def normalize_rows(matrix):
    """
    Normalize each row of the matrix to sum to 1.

    Parameters:
        matrix (numpy.ndarray): The matrix to normalize.

    Returns:
        numpy.ndarray: The row-normalized matrix.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums

def left_Var_Equation(A1, B1):
    """
    Stable solver for np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    via least squares formulation of XA = B  <=> A^T X^T = B^T
    """
    m = B1.shape[0]
    n = A1.shape[0]
    X = np.zeros((m, n))
    for i in range(m):
        X[i, :] = np.linalg.lstsq(np.transpose(A1), B1[i, :], rcond=None)[0]
    return X


def _drift_eval(est_A: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
                X: np.ndarray) -> np.ndarray:
    """Evaluate drift b(x). Accepts callable or matrix A with b(x)=A x."""
    if callable(est_A):
        out = np.asarray(est_A(X))
        if out.shape != X.shape:
            raise ValueError(f"drift_fn(X) must have shape {X.shape}, got {out.shape}")
        return out
    A = np.asarray(est_A)
    return X @ A.T

def _build_gaussian_cost(Xt: np.ndarray, Xt1: np.ndarray,
                         drift_t: np.ndarray, Sigma_dt_inv: np.ndarray) -> np.ndarray:
    """
    C[i,j] = 1/2 (Xt1[j] - Xt[i] - drift_t[i])^T Sigma_dt_inv (Xt1[j] - Xt[i] - drift_t[i])
    """
    D = Xt1[None, :, :] - Xt[:, None, :] - drift_t[:, None, :]  # (n,n,d)
    return 0.5 * np.einsum('...i,ij,...j->...', D, Sigma_dt_inv, D)  # (n,n)

def _safe_log_kernel_from_cost(C: np.ndarray, eps: float) -> np.ndarray:
    """Return logK = -(C - c0)/eps with c0 = min(C) for numerical scale."""
    if eps <= 0:
        raise ValueError("epsilon must be > 0.")
    c0 = float(np.min(C))
    return -(C - c0) / eps

def build_monomials(k, d):
    """
    All exponent tuples e = (e1,…,ed) with total degree ≤ k.
    """
    exps = []
    for total in range(k + 1):
        for e in product(range(total + 1), repeat=d):
            if sum(e) == total:
                exps.append(e)
    return exps

def design_matrix(X, exps):
    Φ = np.empty((X.shape[0], len(exps)))
    for j, e in enumerate(exps):
        Φ[:, j] = np.prod(X ** e, axis=1)   # ∏_m x_m^{e_m}
    return Φ
