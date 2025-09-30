from __future__ import annotations
import numpy as np
import equinox as eqx, jax.numpy as jnp, jax, optax
from utils.APPEX_helpers import _normalise_trajectory
from utils.MLE_parameter_estimation import fit_nn_drift, _estimate_sigma2_isotropic
from utils.SB_solvers import AEOT_trajectory_inference, MMOT_trajectory_inference
from typing import Callable, Optional
import os, json, csv
import time
import csv
import math
import pandas as pd


def sb_refine(
        model=None,
        state=None,
        eval_dataset=None,
        n_outer: int = 30, # set to 1 for the non-iterative WOT algorithm
        n_traj_sample: int = 2000,
        fix_diffusion: bool = False, # True for naive SBIRR, which fixes the reference diffusion
        init_sigma2: float = 1.0,
        init_drift: Callable[[np.ndarray], np.ndarray] | None = None,
        # --- NN hyper-parameters
        nn_width: int = 128,
        nn_depth: int = 2,
        nn_lr: float = 3e-3,
        nn_epochs: int = 500,
        nn_conservative: bool = True,
        nn_activation: str | None = None,
        save_dir: Optional[str] = None
) -> tuple:
    """
    Returns (drift_fn, theta_or_model, sigma2) or (..., trace) if return_trace.

    This version augments the loop with a KL-based surrogate (unchanged)
    and **per-iteration timing** of:
      - Trajectory inference
      - Drift MLE
      - Diffusion MLE (σ² update)

    Two CSVs (if save_dir is provided):
      * sb_timing_iter.csv: per-iteration times
      * sb_timing_summary.csv: mean/std/sem per phase
    """
    assert eval_dataset is not None, "sb_refine needs eval_dataset"
    dt = float(eval_dataset.dt)
    d = int(eval_dataset.data_dim)

    # ------------- measured snapshots (N, T, d) -------------
    X_meas = _normalise_trajectory(eval_dataset)  # (N, T, d)
    N_meas, T_meas, _ = X_meas.shape

    # ------------- initial/reference drift ------------------
    if init_drift is not None:
        drift_ref = init_drift
    elif model is not None and state is not None:
        phi_hat = model.get_potential(state)
        def _single_grad(z): return -jax.grad(lambda u: phi_hat(u))(z)
        def drift_ref(X):
            Xj = jnp.asarray(X)
            out = _single_grad(Xj) if Xj.ndim == 1 else jax.vmap(_single_grad)(Xj)
            return np.asarray(out)
    else:
        drift_ref = lambda X: np.zeros_like(X)

    # current iterate
    sigma2 = float(init_sigma2)
    Sigma = sigma2 * np.eye(d)
    theta_or_model, drift_fn = None, drift_ref
    print(f"[SB] initial estimated diffusivity (σ²): {sigma2:.6g}")
    trace: list[dict] = []  # per-iteration metrics

    # NEW: timing containers
    timing_rows: list[dict] = []  # one row per iteration

    # ===================== outer iterations =====================
    for it in range(n_outer):
        print(f"[SB]  outer iter {it + 1}/{n_outer}")

        # ------------------ E-step: infer trajectories ------------------
        t0 = time.perf_counter()  # NEW: start traj inference timer

        X_OT, P_list = MMOT_trajectory_inference(
            X_meas, dt, est_A=drift_fn, est_Sigma=Sigma,
            max_iter=200, tol=1e-5,
            N_sample_traj=n_traj_sample, use_log_domain=True
        )

        t1 = time.perf_counter()
        t_traj = t1 - t0

        # ------------------ M-step: fit drift ---------------------------
        t2 = time.perf_counter()
        fit_kwargs = dict(
            X=X_OT, dt=dt,
            key=jax.random.PRNGKey(it),
            width=nn_width, depth=nn_depth,
            lr=nn_lr, n_epochs=nn_epochs,
            conservative=nn_conservative
        )
        if nn_activation is not None:
            fit_kwargs["activation"] = nn_activation
        try:
            drift_new, nn_model = fit_nn_drift(**fit_kwargs)
        except TypeError:
            fit_kwargs.pop("activation", None)
            drift_new, nn_model = fit_nn_drift(**fit_kwargs)
        theta_or_model = nn_model
        t3 = time.perf_counter()
        t_drift = t3 - t2

        # ------------------ σ² update -----------------------------------
        t4 = time.perf_counter()  # NEW: start diffusion MLE timer
        if fix_diffusion:
            sigma2_new = sigma2
        else:
            sigma2_new = _estimate_sigma2_isotropic(X_OT, dt, drift_new)
        t5 = time.perf_counter()
        t_sigma2 = t5 - t4  # NEW

        print(f"[SB]  iter {it+1} estimated diffusivity (σ²): {sigma2_new:.6g}")

        Sigma = max(float(sigma2_new), 1e-8) * np.eye(d)
        sigma2 = float(Sigma[0, 0])

        # accept parameters
        drift_prev = drift_fn
        drift_fn = drift_new

        timing_rows.append({
            "iter": it + 1,
            "t_traj_s": t_traj,
            "t_drift_s": t_drift,
            "t_sigma2_s": t_sigma2,
            "t_total_s": (t_traj + t_drift + t_sigma2),
            "nn_width": nn_width,
            "nn_depth": nn_depth,
            "nn_epochs": nn_epochs,
            "n_traj_sample": n_traj_sample,
            "dt": dt,
            "data_dim": d,
            "fix_diffusion": bool(fix_diffusion),
        })

    # --------- NEW: save timing CSVs ----------
    if save_dir is not None and len(timing_rows) > 0:
        os.makedirs(save_dir, exist_ok=True)
        iter_path = os.path.join(save_dir, "sb_timing_iter.csv")
        summary_path = os.path.join(save_dir, "sb_timing_summary.csv")
        _save_timing_csvs(timing_rows, iter_path, summary_path)

    return (drift_fn, theta_or_model, sigma2)




def _save_timing_csvs(timing_rows: list[dict], iter_csv_path: str, summary_csv_path: str) -> None:
    """Write per-iteration timing and an across-iteration summary (mean/std/sem)."""
    # Per-iteration
    if pd is not None:
        df = pd.DataFrame(timing_rows)
        df.to_csv(iter_csv_path, index=False)
        # Summary
        phases = ["t_traj_s", "t_drift_s", "t_sigma2_s", "t_total_s"]
        stats = []
        for ph in phases:
            vals = df[ph].to_numpy(dtype=float)
            n = int(np.isfinite(vals).sum())
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                mean = std = sem = float("nan")
                n_eff = 0
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                sem = float(std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                n_eff = len(vals)
            stats.append({"phase": ph, "mean_s": mean, "std_s": std, "sem_s": sem, "n": n_eff})
        pd.DataFrame(stats).to_csv(summary_csv_path, index=False)
    else:
        # Fallback: stdlib csv
        fieldnames = sorted({k for row in timing_rows for k in row.keys()})
        with open(iter_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in timing_rows:
                w.writerow(r)
        # Summary with minimal deps
        phases = ["t_traj_s", "t_drift_s", "t_sigma2_s", "t_total_s"]
        with open(summary_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "mean_s", "std_s", "sem_s", "n"])
            w.writeheader()
            for ph in phases:
                vals = [float(r[ph]) for r in timing_rows if r.get(ph) is not None]
                n = len(vals)
                if n == 0:
                    w.writerow({"phase": ph, "mean_s": float("nan"), "std_s": float("nan"),
                                "sem_s": float("nan"), "n": 0})
                    continue
                mean = sum(vals) / n
                if n > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                    std = math.sqrt(var)
                    sem = std / math.sqrt(n)
                else:
                    std = 0.0
                    sem = 0.0
                w.writerow({"phase": ph, "mean_s": mean, "std_s": std, "sem_s": sem, "n": n})

