#!/usr/bin/env python3
"""
Minimal drift-field plotter.

Folder layout expected:
  <root>/p0-<p0>/<dataset>/<panel>/inference_bundle.npz

Add "gt" to --panels to include a ground-truth panel (requires JAX + utils.functions).
Example
-------
python plot_drift_fields.py \
  --root ../main_experiments --p0 unif \
  --dataset 3_margs_langevin_oakley_ohagan_diff-0.2_seed-1000 \
  --panels gt,appex,wot,sbirr,jkonet_star \
  --potential oakley_ohagan \
  --stream --background speed --debug-paths
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from utils.compute_metrics import compute_metrics, eval_gt_on_grid, load_bundle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 24,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
})

BASE_TITLES = {
    "gt":          r"Ground truth",
    "wot":         r"WOT",
    "sbirr":       r"SBIRR",
    "nn_appex":    r"nn-APPEX",
    "jkonet_star": r"JKONet$^\ast$",
    "jkonet":      r"JKONet",
}

def parse_key_value_list(s):
    if not s:
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            out[item] = None
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def parse_title_overrides(title_map_str, panels_order):
    raw = parse_key_value_list(title_map_str)
    if not raw:
        return {}
    if any(v is not None for v in raw.values()):
        return {k: v for k, v in raw.items() if v is not None}
    vals = [k for k in raw.keys()]
    return {m: vals[i] for i, m in enumerate(panels_order) if i < len(vals)}

def pretty_title(key: str) -> str:
    return key.replace("_", " ").title()


def panel_plot(ax, bundle, title=None, square=None,
               background="speed", gt_Z=None,
               arrow_source="inferred", Ugt=None, Vgt=None,
               show_stream=False, show_quiver=False, quiver_step=1,
               norm=None, nlevels=16,
               stream_min=None, stream_q=None):
    X, Y = bundle["X"], bundle["Y"]
    U, V = bundle["U"], bundle["V"]
    speed = bundle["speed"]

    scalar = gt_Z if background == "gt_potential" else speed
    levels = np.linspace(norm.vmin, norm.vmax, nlevels)
    cf = ax.contourf(X, Y, scalar, levels=levels, norm=norm,
                     cmap="viridis", extend="both")

    Au, Av = (Ugt, Vgt) if arrow_source == "gt" else (U, V)

    if show_stream:
        Suv = np.hypot(Au, Av)
        cutoff = None
        if stream_q is not None:
            cutoff = float(np.nanquantile(Suv, stream_q))
        if stream_min is not None:
            cutoff = stream_min if cutoff is None else max(cutoff, stream_min)
        if cutoff is not None:
            Au_m = np.ma.masked_where(Suv < cutoff, Au)
            Av_m = np.ma.masked_where(Suv < cutoff, Av)
        else:
            Au_m, Av_m = Au, Av
        ax.streamplot(X, Y, Au_m, Av_m, density=1.2, linewidth=0.9,
                      arrowsize=1.2, color="k")

    if show_quiver:
        step = max(1, int(quiver_step))
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  Au[::step, ::step], Av[::step, ::step],
                  scale=None, width=0.002, alpha=0.7)

    if square is not None:
        ax.set_xlim(*square); ax.set_ylim(*square)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=16, pad=8)
    return cf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="main_experiments")
    p.add_argument("--p0", type=str, default="gmm", choices=["gmm", "gibbs", "unif"])
    p.add_argument("--dataset", type=str,
                   help="Dataset folder name under p0-<p0>/ (no auto-template).")
    p.add_argument("--panels", type=str, default="gt,appex,wot,sbirr,jkonet_star",
                   help="Comma-separated list of panels/subfolders (use 'gt' for ground-truth).")
    p.add_argument("--potential", required=True,
                   choices=["quadratic","styblinski_tang","bohachevsky","wavy_plateau","oakley_ohagan",
                            "watershed","friedman","ishigami","flowers","double_exp","relu","holder_table","zigzag_ridge","flat"])
    p.add_argument("--seed", type=int, default=1000,
                   help="Used to auto-fill dataset name if --dataset is not provided.")
    # optional title overrides "key=Nice Title,other=Another Title"
    p.add_argument("--title-map", type=str, default=None,
                   help="Comma-separated key=value pairs to override panel titles.")

    # figure settings
    p.add_argument("--square", type=float, nargs=2, default=[-4,4])
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--dpi", type=int, default=300)

    # arrows
    p.add_argument("--stream", action="store_true", help="Enable streamline arrows")
    p.add_argument("--quiver", action="store_true", help="Enable quiver arrows")
    p.add_argument("--quiver-step", type=int, default=2, help="Subsample factor for quiver arrows")
    p.add_argument("--arrow-source", choices=["inferred","gt"], default="inferred",
                   help="Which vector field to use for arrows")

    # streamline thresholding
    p.add_argument("--stream-min", type=float, default=None,
                   help="Absolute speed threshold; hide streamlines where ||b|| < stream-min.")
    p.add_argument("--stream-q", type=float, default=None,
                   help="Quantile in [0,1]; hide streamlines below this speed quantile (per panel).")

    # background + color scaling
    p.add_argument("--background", choices=["speed","gt_potential"], default="speed",
                   help="Heatmap background: inferred speed or ground-truth potential")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--levels", type=int, default=16, help="Number of contour levels")

    # diagnostics
    p.add_argument("--print-metrics", action="store_true",
                   help="Print normalized drift MAE and cosine similarity vs GT for each non-GT panel.")
    p.add_argument("--debug-paths", action="store_true",
                   help="Print the resolved npz path for each panel.")

    args = p.parse_args()

    if args.dataset is None:
        if args.potential is None or args.seed is None:
            raise ValueError("If --dataset is not provided, you must pass both --potential and --seed.")
        args.dataset = f"3_margs_langevin_{args.potential}_diff-0.2_seed-{args.seed}"

    root = Path(args.root)
    base_dir = (root / f"p0-{args.p0}" / args.dataset).resolve()

    # parse panels
    panels = [s.strip() for s in args.panels.split(",") if s.strip()]
    # canonical order for display
    PANEL_ORDER = ["gt", "wot", "sbirr", "nn_appex", "jkonet_star", "jkonet"]

    # reorder the requested panels according to the canonical order
    req = panels
    panels = [m for m in PANEL_ORDER if m in req] + [m for m in req if m not in PANEL_ORDER]
    # also drop duplicates while preserving order
    seen = set()
    panels = [m for m in panels if not (m in seen or seen.add(m))]

    # map each panel to its directory (literal names), check duplicates
    panel_dirs = {}
    for m in panels:
        panel_dirs[m] = "gt" if m == "gt" else str((base_dir / m).resolve())

    seen = {}
    dups = []
    for m, pth in panel_dirs.items():
        if m == "gt":
            continue
        if pth in seen:
            dups.append((m, seen[pth], pth))
        else:
            seen[pth] = m
    if dups:
        msg = "\n".join([f"  {a} and {b} both point to {pth}" for (a, b, pth) in dups])
        raise RuntimeError(f"Two or more panels resolve to the same folder:\n{msg}")

    # load bundles
    bundles = {}
    bundle_paths = {}
    for m in panels:
        if m == "gt":
            # borrow a grid from first available non-GT panel
            borrow_from = [x for x in panels if x != "gt"]
            if not borrow_from:
                raise RuntimeError("Need at least one non-GT panel to build GT grid.")
            # find one that actually exists
            picked = None
            for b in borrow_from:
                try:
                    _ = load_bundle(base_dir / b)
                    picked = b
                    break
                except Exception:
                    continue
            if picked is None:
                raise RuntimeError("Could not find any existing non-GT bundle to borrow grid from.")
            b = load_bundle(base_dir / picked)
            X, Y = b["X"], b["Y"]
            Z, Ugt, Vgt = eval_gt_on_grid(args.potential, X, Y)
            if Z is None:
                msg = "Ground-truth requires JAX + utils.functions."
            bundles[m] = dict(X=X, Y=Y, U=Ugt, V=Vgt, speed=np.hypot(Ugt, Vgt), _gtZ=Z)
            bundle_paths[m] = "gt"
        else:
            panel_dir = base_dir / m
            bundles[m] = load_bundle(panel_dir)
            bundle_paths[m] = str((panel_dir / "inference_bundle.npz").resolve())

    if args.debug_paths:
        print("\n[debug] loaded panels from:")
        for m in panels:
            print(f"  {m:>16s}  ->  {bundle_paths.get(m, '?')}")

    # GT eval on each grid if needed
    need_gt = (args.background == "gt_potential") or (args.arrow_source == "gt") or ("gt" in panels)
    gt_grids = {}
    if need_gt:
        for m in panels:
            X, Y = bundles[m]["X"], bundles[m]["Y"]
            if m == "gt" and "_gtZ" in bundles[m]:
                Z, Ugt, Vgt = bundles[m]["_gtZ"], bundles[m]["U"], bundles[m]["V"]
            else:
                Z, Ugt, Vgt = eval_gt_on_grid(args.potential, X, Y)
            gt_grids[m] = (Z, Ugt, Vgt)

    # color range
    mins, maxs = [], []
    for m in panels:
        if args.background == "gt_potential":
            Z = gt_grids[m][0]
            mins.append(np.nanmin(Z)); maxs.append(np.nanmax(Z))
        else:
            mins.append(np.nanmin(bundles[m]["speed"]))
            maxs.append(np.nanmax(bundles[m]["speed"]))
    vmin = args.vmin if args.vmin is not None else min(mins) if mins else 0.0
    vmax = args.vmax if args.vmax is not None else max(maxs) if maxs else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise RuntimeError(f"Invalid color range vmin={vmin}, vmax={vmax}. "
                           "Try supplying --vmin/--vmax explicitly.")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # titles
    title_overrides = parse_title_overrides(args.title_map, panels)

    # plot
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.4 * len(panels), 3.4),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    mappable = None
    for ax, m in zip(axes, panels):
        Z, Ugt, Vgt = (gt_grids[m] if m in gt_grids else (None, None, None))
        # letter prefix based on final panel order
        letter = chr(ord('a') + panels.index(m))
        base_title = BASE_TITLES.get(m, pretty_title(m))
        title_text = title_overrides.get(m, f"({letter}) {base_title}")
        mappable = panel_plot(
            ax, bundles[m], title=title_text,
            square=args.square,
            background=args.background, gt_Z=Z,
            arrow_source=args.arrow_source, Ugt=Ugt, Vgt=Vgt,
            show_stream=args.stream, show_quiver=args.quiver,
            quiver_step=args.quiver_step,
            norm=norm, nlevels=args.levels,
            stream_min=args.stream_min, stream_q=args.stream_q,
        )

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.9, label=None)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.yaxis.set_major_formatter("{:.0f}".format)

    # print-only metrics vs GT
    if args.print_metrics and need_gt:
        for m in panels:
            if m == "gt":
                continue
            Ue, Ve = bundles[m]["U"], bundles[m]["V"]
            _, Ugt, Vgt = gt_grids[m]
            nmae, cos = compute_metrics(Ue, Ve, Ugt, Vgt)
            print(f"[metrics] {m:>16s} :  normalized_MAE={nmae:.4f}   cosine={cos:.4f}")

    if args.save:
        out = Path(args.save); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {out.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    main()