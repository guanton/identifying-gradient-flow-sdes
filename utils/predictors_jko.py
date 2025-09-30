# utils/predictors_jko.py
from __future__ import annotations
import json, os
import numpy as np

def _load_bundle(bundle_path: str):
    """
    Load an inference bundle saved by train.py/_save_inference_bundle (or per-time bundles).
    Returns (z, meta) where:
      z    : npz object / dict-like with keys {random_points, random_drift, grid_points, grid_drift}
      meta : dict with at least {"sigma2": float, "dim": int}
    """
    if not isinstance(bundle_path, (str, os.PathLike)):
        raise TypeError(f"bundle_path must be a path; got {type(bundle_path)}")

    z = np.load(bundle_path)
    meta_path = os.path.join(os.path.dirname(bundle_path), "inference_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        # Fallback if meta missing; try to infer σ² from arrays
        meta = {"sigma2": 0.2, "dim": int(z["random_points"].shape[1])}
    return z, meta

def _match_dims(points: np.ndarray, d_target: int) -> np.ndarray:
    """
    Make 'points' match the target spatial dimension d_target by:
      - slicing off extra trailing dims if points.shape[1] > d_target (e.g., time channel),
      - raising if points.shape[1] < d_target.
    """
    d_bundle = points.shape[1]
    if d_bundle == d_target:
        return points
    if d_bundle > d_target:
        # Drop trailing channels (e.g., last col is time for JKO-time)
        return points[:, :d_target]
    raise ValueError(
        f"Bundle has fewer dims ({d_bundle}) than required target dim ({d_target}). "
        f"Cannot up-cast."
    )

def _drift_from_bundle(X_t: np.ndarray, z) -> np.ndarray:
    """
    Estimate the drift at X_t using the bundle's random cloud and drifts.
    We do a simple kernel/Nystrom interpolation over (random_points, random_drift).
    """
    X = np.asarray(X_t, dtype=np.float32)          # [B,d]
    Xr = np.asarray(z["random_points"])            # [N,d_b]
    Gr = np.asarray(z["random_drift"])             # [N,d_b] (drift in same space)

    # Match bundle dims to X_t dims (drop extra channels if present)
    Xr = _match_dims(Xr, X.shape[1])
    Gr = _match_dims(Gr, X.shape[1])

    # Kernel interpolation: K(x, xr) with Gaussian kernel
    # Bandwidth: median distance heuristic on Xr
    Xr2 = np.sum(Xr * Xr, axis=1, keepdims=True)                  # [N,1]
    XX2 = np.sum(X  * X,  axis=1, keepdims=True)                  # [B,1]
    D2  = XX2 + Xr2.T - 2.0 * (X @ Xr.T)                          # [B,N]
    D2  = np.maximum(D2, 0.0)

    # bandwidth ~ median distance on the random cloud
    # Avoid O(N^2) here; take a random subset if huge
    N = Xr.shape[0]
    if N > 5000:
        idx = np.random.default_rng(0).choice(N, 5000, replace=False)
        Xr_sub = Xr[idx]
        D2_sub = np.sum(Xr_sub * Xr_sub, axis=1, keepdims=True) + np.sum(Xr_sub * Xr_sub, axis=1)[None, :] - 2.0 * (Xr_sub @ Xr_sub.T)
        D2_sub = np.maximum(D2_sub, 0.0)
        h2 = np.median(D2_sub)
    else:
        D2_self = np.sum(Xr * Xr, axis=1, keepdims=True) + np.sum(Xr * Xr, axis=1)[None, :] - 2.0 * (Xr @ Xr.T)
        D2_self = np.maximum(D2_self, 0.0)
        # ignore zeros on diagonal by adding a small eps before median
        h2 = np.median(D2_self + 1e-8)

    if not np.isfinite(h2) or h2 <= 0:
        h2 = 1.0

    W = np.exp(-D2 / (h2 + 1e-8))                                 # [B,N]
    W_sum = np.sum(W, axis=1, keepdims=True) + 1e-12
    drift = (W @ Gr) / W_sum                                      # [B,d]
    return drift.astype(np.float32)

def predict_one_step(ckpt_path: str, X_t: np.ndarray, dt: float) -> np.ndarray:
    """
    One EM step using the exported bundle. Signature expected by jko_scripts.eval_rna_emd:
        X_pred = predict_one_step(ckpt_path, A_s, dt)

    - Loads (random_points, random_drift) from the bundle.
    - Interpolates drift at X_t via a smooth kernel on the random cloud.
    - EM step with σ² from meta (if present), else defaults to 0.2.

    Args:
        ckpt_path : path to `.../inference_bundle.npz`
        X_t       : [n, d] numpy array at time t
        dt        : scalar Δt between snapshots t→t+1

    Returns:
        X_{t+1} prediction, shape [n, d]
    """
    z, meta = _load_bundle(ckpt_path)
    sigma2 = float(meta.get("sigma2", 0.2))    # σ²
    sigma  = float(np.sqrt(max(sigma2, 1e-12)))

    X_t = np.asarray(X_t, dtype=np.float32)
    B, d = X_t.shape

    drift = _drift_from_bundle(X_t, z)         # [B,d]
    noise = np.random.default_rng(0).standard_normal(size=(B, d)).astype(np.float32)
    dX    = drift * float(dt) + sigma * np.sqrt(float(dt)) * noise
    return (X_t + dX).astype(np.float32)