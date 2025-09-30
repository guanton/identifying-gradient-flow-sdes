#!/usr/bin/env python3
import os, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

from utils.compute_metrics import (
    load_bundle,
    eval_gt_on_grid,
    parse_run_folder,
    compute_diff_mae_for_run,
    sem
)

# ---------- IQR whiskers + boxagg ----------
def whiskers_iqr(v: np.ndarray, q1: float, q3: float, k: float) -> tuple[float, float]:
    iqr = q3 - q1
    lo_f, hi_f = q1 - k * iqr, q3 + k * iqr
    mask = (v >= lo_f) & (v <= hi_f)
    if not np.any(mask):
        return float(np.min(v)), float(np.max(v))
    vv = v[mask]
    return float(np.min(vv)), float(np.max(vv))

def summarize_boxagg(df_pointwise: pd.DataFrame,
                     *,
                     value_col: str = "value",
                     whisker_mult: float = 1.5) -> pd.DataFrame:
    rows = []
    for (pot, meth), g in df_pointwise.groupby(["potential", "method"], sort=False):
        v = g[value_col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        q1  = float(np.percentile(v, 25))
        med = float(np.percentile(v, 50))
        q3  = float(np.percentile(v, 75))
        lw, uw = whiskers_iqr(v, q1, q3, whisker_mult)
        rows.append({
            "potential": pot, "method": meth,
            "mode": f"wiqr{whisker_mult:g}",
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "median": med,
            "q1": q1, "q3": q3, "iqr": q3 - q1,
            "lw": lw, "uw": uw,
            "min": float(np.min(v)), "max": float(np.max(v)),
        })
    return pd.DataFrame(rows, columns=[
        "potential","method","mode","n","mean","median","q1","q3","iqr","lw","uw","min","max"
    ])

# ---------- internals for per-metric extraction ----------
def iter_runs(base: Path, req_pots, req_seeds):
    for name in os.listdir(base):
        run_dir = base / name
        if not run_dir.is_dir():
            continue
        parsed = parse_run_folder(name, prefix=None)
        if not parsed:
            continue
        pot, step, sigma2, seed = parsed
        if pot not in req_pots:
            continue
        if req_seeds is not None and seed not in req_seeds:
            continue
        yield run_dir, pot, step, sigma2, seed

def collect_pointwise_mae(base: Path, req_pots, req_methods, req_seeds):
    rows = []
    for run_dir, pot, step, _, seed in iter_runs(base, req_pots, req_seeds):
        for method in req_methods:
            panel_dir = run_dir / method
            npz = panel_dir / "inference_bundle.npz"
            if not npz.exists():
                continue
            try:
                bundle = load_bundle(panel_dir)
                Z, Ugt, Vgt = eval_gt_on_grid(pot, bundle["X"], bundle["Y"])
                if Z is None:  # JAX/GT unavailable
                    continue
            except Exception:
                continue
            # L1 normalized pointwise
            T1 = np.abs(Ugt) + np.abs(Vgt)
            D1 = np.abs(bundle["U"] - Ugt) + np.abs(bundle["V"] - Vgt)
            X, Y = bundle["X"], bundle["Y"]
            H, W = X.shape
            for i in range(H):
                for j in range(W):
                    if T1[i, j] <= 1e-12:
                        continue
                    rows.append({
                        "potential": pot,
                        "method": method,
                        "seed": int(seed),
                        "x": float(X[i, j]),
                        "y": float(Y[i, j]),
                        "value": float(D1[i, j] / T1[i, j]),
                    })
    return pd.DataFrame(rows, columns=["potential","method","seed","x","y","value"]) if rows else pd.DataFrame(
        columns=["potential","method","seed","x","y","value"]
    )

def collect_pointwise_cos(base: Path, req_pots, req_methods, req_seeds):
    rows = []
    for run_dir, pot, step, _, seed in iter_runs(base, req_pots, req_seeds):
        for method in req_methods:
            panel_dir = run_dir / method
            npz = panel_dir / "inference_bundle.npz"
            if not npz.exists():
                continue
            try:
                bundle = load_bundle(panel_dir)
                Z, Ugt, Vgt = eval_gt_on_grid(pot, bundle["X"], bundle["Y"])
                if Z is None:
                    continue
            except Exception:
                continue
            Ue, Ve = bundle["U"], bundle["V"]
            eps = 1e-12
            num = (Ue * Ugt + Ve * Vgt)
            den = (np.sqrt(Ue**2 + Ve**2) * np.sqrt(Ugt**2 + Vgt**2) + eps)
            cos = np.clip(num / den, -1.0, 1.0)
            X, Y = bundle["X"], bundle["Y"]
            H, W = X.shape
            for i in range(H):
                for j in range(W):
                    rows.append({
                        "potential": pot,
                        "method": method,
                        "seed": int(seed),
                        "x": float(X[i, j]),
                        "y": float(Y[i, j]),
                        "value": float(cos[i, j]),
                    })
    return pd.DataFrame(rows, columns=["potential","method","seed","x","y","value"]) if rows else pd.DataFrame(
        columns=["potential","method","seed","x","y","value"]
    )

def write_pointwise(df: pd.DataFrame, base_dir: Path, setting: str, metric: str, save_csv: str|None):
    # resolve output path
    if save_csv:
        out = Path(save_csv)
        if not out.is_absolute():  # keep relative overrides inside base_dir
            out = base_dir / out
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        out = base_dir / f"metric={metric}_{setting}_pointwise.csv"

    if df.empty:
        print(f"[warn] no pointwise rows for {metric}; nothing written.")
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[saved {metric} pointwise] {out}")
    return out


def write_boxagg_from_df(df_pointwise: pd.DataFrame,
                         base_dir: Path,
                         pointwise_path: Path|None,
                         setting: str,
                         metric: str,
                         k: float):
    if df_pointwise.empty:
        print(f"[warn] no rows to boxagg for {metric}.")
        return None

    df_box = summarize_boxagg(df_pointwise, value_col="value", whisker_mult=k)
    tag = df_box["mode"].iloc[0] if not df_box.empty else f"wiqr{k:g}"

    if pointwise_path is not None and pointwise_path.name.endswith("_pointwise.csv"):
        box_path = pointwise_path.with_name(
            pointwise_path.name.replace("_pointwise.csv", f"_boxagg_{tag}.csv")
        )
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        box_path = base_dir / f"metric={metric}_{setting}_boxagg_{tag}.csv"

    df_box.to_csv(box_path, index=False)
    print(f"[saved {metric} boxagg] {box_path}")
    return box_path


def write_diff_agg(df_agg: pd.DataFrame, base_dir: Path, setting: str, save_csv: str|None):
    if save_csv:
        out = Path(save_csv)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        out = base_dir / f"metric=diff_mae_{setting}_agg.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out, index=False)
    print(f"[saved agg] {out}")
    return out

# ------------------------------- main -------------------------------
def main():
    p = argparse.ArgumentParser(description="Export CSVs for pointwise L1 drift MAE / cosine (and optional IQR boxagg), or diffusivity MAE.")
    # Structure
    p.add_argument("--root", type=str, default="../main_experiments")
    p.add_argument("--setting", type=str, default="p0-gibbs", choices=["p0-gmm","p0-unif","p0-gibbs"])
    # Filters
    p.add_argument("--potentials", type=str, default="poly,styblinski_tang,bohachevsky,wavy_plateau,oakley_ohagan")
    p.add_argument("--methods", type=str, default="nn_appex,wot,sbirr,jkonet_star,jkonet")
    p.add_argument("--seeds", type=str, default=None)
    # Metrics selection
    p.add_argument("--metrics", type=str, default=None,
                   help="Comma list of metrics to export in one run, e.g. 'grid_mae,grid_cos,diff_mae'.")
    # Outputs
    p.add_argument("--save-dir", type=str, default='results_csvs',
                   help="Base directory for saving")
    p.add_argument("--save-csv", type=str, default=None,
                   help="Override output file path **only when exactly one metric is selected**.")
    # Box options
    p.add_argument("--box", action="store_true", help="Also write IQR boxagg CSV for grid metrics.")
    p.add_argument("--box-only", action="store_true", help="Write only the IQR boxagg CSV (no pointwise CSV).")
    p.add_argument("--whisker-mult", type=float, default=1.5)

    args = p.parse_args()

    # resolve metrics
    metrics = []
    if args.metrics and args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    base = Path(args.root) / args.setting
    if not base.exists():
        raise SystemExit(f"[error] Base directory does not exist: {base}")

    req_pots = {p.strip() for p in args.potentials.split(",") if p.strip()}
    req_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    req_seeds = None
    if args.seeds and args.seeds.strip():
        req_seeds = {int(s) for s in args.seeds.split(",") if s.strip()}

    # Where to drop defaults if save_dir not given: the p0 folder
    def resolve_under_project_root(p: str | Path) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p  # always place relative save-dirs at repo root

    # Where to drop defaults if save_dir not given:
    # - if --save-dir is given (default 'results_csvs'), write under PROJECT_ROOT/--save-dir
    # - else write next to the data base folder
    default_base_dir = resolve_under_project_root(args.save_dir) if args.save_dir else base
    default_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save base] {default_base_dir}")

    # Guard: --save-csv only meaningful when exactly one metric selected
    if args.save_csv and len(metrics) != 1:
        print("[warn] --save-csv is ignored because multiple metrics were requested.")

    for metric in metrics:
        if metric == "grid_mae":
            df_pp = collect_pointwise_mae(base, req_pots, req_methods, req_seeds)
            if df_pp.empty:
                print("[warn] No pointwise MAE rows; skipping outputs.")
                continue
            if args.box_only:
                write_boxagg_from_df(df_pp, default_base_dir, pointwise_path=None,
                                     setting=args.setting, metric="grid_mae", k=args.whisker_mult)
            else:
                pw_path = write_pointwise(df_pp, default_base_dir, args.setting, "grid_mae",
                                          args.save_csv if len(metrics) == 1 else None)
                if args.box:
                    write_boxagg_from_df(df_pp, default_base_dir, pointwise_path=pw_path,
                                         setting=args.setting, metric="grid_mae", k=args.whisker_mult)

        elif metric == "grid_cos":
            df_pp = collect_pointwise_cos(base, req_pots, req_methods, req_seeds)
            if df_pp.empty:
                print("[warn] No pointwise cosine rows; skipping outputs.")
                continue
            if args.box_only:
                write_boxagg_from_df(df_pp, default_base_dir, pointwise_path=None,
                                     setting=args.setting, metric="grid_cos", k=args.whisker_mult)
            else:
                pw_path = write_pointwise(df_pp, default_base_dir, args.setting, "grid_cos",
                                          args.save_csv if len(metrics) == 1 else None)
                if args.box:
                    write_boxagg_from_df(df_pp, default_base_dir, pointwise_path=pw_path,
                                         setting=args.setting, metric="grid_cos", k=args.whisker_mult)

        elif metric == "diff_mae":
            # per-seed -> mean/sem across seeds
            rows = []
            for run_dir, pot, step, sigma2, seed in iter_runs(base, req_pots, req_seeds):
                for method in req_methods:
                    mdir = run_dir / method
                    if not mdir.exists():
                        continue
                    rec = compute_diff_mae_for_run(run_dir, method, (pot, step, sigma2, seed))
                    if rec is not None:
                        rec["method"] = method
                        rows.append(rec)
            if not rows:
                print("[warn] No diffusivity records; skipping outputs.")
                continue
            df_raw = pd.DataFrame(rows, columns=["potential","method","seed","step","value"])
            g = df_raw.groupby(["potential","method"], as_index=False)["value"]
            df_agg = g.agg(mean="mean", sem=sem, n="count")[["potential","method","mean","sem","n"]]
            write_diff_agg(df_agg, default_base_dir, args.setting,
                           args.save_csv if len(metrics) == 1 else None)

        else:
            print(f"[warn] Unknown metric '{metric}', skipping.")

if __name__ == "__main__":
    main()