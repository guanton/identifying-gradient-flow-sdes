#!/usr/bin/env python3
# make_boxplot_pdfs.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def draw_legend(fig, methods: list[str], face_colors: dict, hatch_patterns: dict):
    handles = []
    for m in methods:
        patch = Patch(
            facecolor=face_colors.get(m, "#CCCCCC"),
            edgecolor="black",
            hatch=hatch_patterns.get(m, None),
            label=METHOD_LABEL.get(m, m),
            linewidth=1.0
        )
        handles.append(patch)
    # Legend BELOW the entire figure
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(handles),
        fontsize=10,
        frameon=False
    )


# --- ensure UNCOMPRESSED PDFs & editable fonts ---
mpl.rcParams["pdf.compression"] = 0
mpl.rcParams["pdf.fonttype"]   = 42
mpl.rcParams["ps.fonttype"]    = 42

# --- paper font ---
mpl.rcParams["font.family"] = "Times New Roman"

# --- palettes (paper) ---
# Drift error (MAE) colors
ORANGE_WOT    = "#E4936C"
ORANGE_SBIRR  = "#EF975F"
ORANGE_APPEX  = "#F4A261"
ORANGE_JKONET = "#F2A56F"   # JKONet* variants (MAE)


# Cosine colors
BLUE_WOT      = "#6EA8DC"
BLUE_SBIRR    = "#74B1D8"
BLUE_APPEX    = "#7AB8F1"
BLUE_JKONET   = "#82B0F5"   # JKONet* variants (Cos)


PURPLE = "#8E7CC3"

DEFAULT_POT_ORDER = [
    "bohachevsky", "oakley_ohagan", "poly", "styblinski_tang", "wavy_plateau"
]
PRETTY_POT = {
    "bohachevsky":      "Bohachevsky",
    "oakley_ohagan":    "Oakley–O’Hagan",
    "poly":             "Quadratic",
    "styblinski_tang":  "Styblinski–Tang",
    "wavy_plateau":     "Wavy plateau",
}

# canonical order + labels + hatch
# (include both JK variants; you can still choose a subset via --methods)
METHOD_ORDER = ["wot", "SBIRR", "nn_appex", "jkonet_star", "jkonet_10000"]
METHOD_LABEL = {
    "wot": "WOT",
    "SBIRR": "SBIRR",
    "nn_appex": "nn-APPEX",
    "jkonet_star": "JKONet*",
    "jkonet_10000": "JKONet* (10k)",
}
# --- make JKONet boxes use markers only (no hatch) ---
METHOD_HATCH = {
    "wot": None,
    "SBIRR": ".",
    "nn_appex": "//",
    "jkonet_star": None,      # was "*"
    "jkonet_10000": None,     # was "x"
}

# built-in normalization for common variants (case/format tolerant)
DEFAULT_ALIASES = {
    # WOT
    "wot": "wot", "WOT": "wot",

    # SBIRR
    "sirr": "SBIRR", "sbirr": "SBIRR", "SBIRR": "SBIRR",

    # nn-APPEX
    "nn_appex": "nn_appex",
    "appex": "nn_appex", "APPEX": "nn_appex",
    "nn-APPEX": "nn_appex", "NN-APPEX": "nn_appex",
    "nn-appex": "nn_appex", "NN-appex": "nn_appex",

    # JKONet variants
    "jkonet*": "jkonet_star", "JKONET*": "jkonet_star",
    "jkonet-star": "jkonet_star", "JKONET-STAR": "jkonet_star",
}

SUBPLOT_ADJ = dict(left=0.10, right=0.995, bottom=0.18, top=0.995)

# ---------------- IO ----------------
def _load_boxcsv(path: Path, user_alias: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"potential", "method", "lw", "q1", "median", "q3", "uw"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    alias = DEFAULT_ALIASES.copy()
    alias.update(user_alias or {})

    def norm(m):
        mk  = str(m)
        low = mk.lower().replace("-", "_")
        # 1) alias table first (exact or lowercase)
        if mk in alias:        return alias[mk]
        if low in alias:       return alias[low]
        # 2) JKONet variants: keep distinct keys (so they can be compared)
        #    normalize to canonical lowercase tokens
        if low == "jkonet_star":    return "jkonet_star"
        if low == "jkonet_10000":   return "jkonet_10000"
        if low == "jkonet":         return "jkonet"
        if low.startswith("jkonet_"):  # e.g., jkonet_500
            return low
        # 3) default: keep original token
        return mk

    df["method"] = df["method"].map(norm)
    return df

def _draw_grouped_bars(ax,
                       df: pd.DataFrame,
                       *,
                       methods: List[str],
                       face_colors: Dict[str, str],
                       box_width: float = 0.32,
                       group_gap: float = 1.35,
                       offset: float = 0.35,
                       pot_order: List[str] = DEFAULT_POT_ORDER) -> None:
    x0 = 1.0
    offs = _symmetric_offsets(len(methods), offset)

    # build fast lookup: (pot, method) -> (mean, sem)
    key = {(r["potential"], r["method"]): (float(r["mean"]), float(r["sem"]))
           for _, r in df.iterrows()}

    xs = [x0 + i*group_gap for i in range(len(pot_order))]
    for i, pot in enumerate(pot_order):
        base = xs[i]
        for j, m in enumerate(methods):
            pos = base + offs[j]
            if (pot, m) not in key:
                continue
            y, se = key[(pot, m)]
            ax.bar(pos, y, width=box_width,
                   facecolor=face_colors.get(m, "#CCCCCC"),
                   edgecolor="black", linewidth=1.0,
                   zorder=2)
            ax.errorbar(pos, y, yerr=se, fmt="none",
                        ecolor="black", elinewidth=1.0,
                        capsize=3, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY_POT.get(p, p) for p in pot_order], fontsize=12)

    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)


def make_diffusivity_bars(diff_csv: Path, out_dir: Path,
                          *, pot_order=DEFAULT_POT_ORDER, p0 = None):
    # colors/patterns (same color for both bars)
    PURPLE = "#7E57C2"  # tweak if you prefer a different purple
    FACE = {"nn_appex": PURPLE, "jkonet_star": PURPLE}
    HATCH = {"nn_appex": "//", "jkonet_star": None}

    df = pd.read_csv(diff_csv)
    # normalize method tokens like elsewhere
    alias = DEFAULT_ALIASES.copy()
    def norm(m):
        mk = str(m)
        low = mk.lower().replace("-", "_")
        if mk in alias: return alias[mk]
        if low in alias: return alias[low]
        if low in {"jkonet", "jkonet_star"} or low.startswith("jkonet_"):
            return low
        return mk
    df["method"] = df["method"].map(norm)

    # Pivot to (potential, method) → mean / sem
    # allow either 'sem' or 'stderr' or 'se' column names
    err_col = None
    for cand in ["sem", "stderr", "se"]:
        if cand in df.columns:
            err_col = cand
            break
    if err_col is None:
        df["__err__"] = 0.0
        err_col = "__err__"

    # keep only the two methods
    df = df[df["method"].isin(["nn_appex", "jkonet_star"])]

    fig, ax = plt.subplots(figsize=(10.7, 3.0))

    # geometry consistent with boxplots
    box_width = 0.32
    group_gap = 1.35
    offset    = 0.35
    x0 = 1.0
    methods = ["jkonet_star", "nn_appex"]  # order matches your CLI example
    offs = _symmetric_offsets(len(methods), offset)

    # draw bars per method
    for j, m in enumerate(methods):
        block = df[df["method"] == m].set_index("potential")
        # gather means/errors aligned to pot_order
        means = [float(block.loc[p, "mean"]) if p in block.index else np.nan
                 for p in pot_order]
        errs  = [float(block.loc[p, err_col]) if p in block.index else 0.0
                 for p in pot_order]

        # positions
        xs = [x0 + i*group_gap + offs[j] for i, _ in enumerate(pot_order)]

        bars = ax.bar(xs, means, width=box_width,
                      color=FACE[m], edgecolor="black", linewidth=1.0,
                      yerr=errs, capsize=3, ecolor="black")

        # patterns: APPEX hatch, JKONet* stars inside
        if HATCH[m]:
            for b in bars:
                b.set_hatch(HATCH[m])

        if m == "jkonet_star":
            # overlay filled black stars clipped to the bar patch
            for b in bars:
                _overlay_markers(ax, b, marker="*", size=28, color="black",
                                 ncols=3, nrows=2, margin=0.18)

    # x ticks / labels
    ax.set_xticks([x0 + i*group_gap for i in range(len(pot_order))])
    ax.set_xticklabels([PRETTY_POT.get(p, p) for p in pot_order], fontsize=12)

    # y ticks (reuse the same locator style as boxplots)
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=20, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Diffusivity MAE", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)

    fig.subplots_adjust(**SUBPLOT_ADJ)

    if p0 is not None:
        diff_filename = f'diff_mae_{p0}.pdf'
    else:
        diff_filename = f'diff_box_mae.pdf'

    fig.savefig(out_dir / diff_filename, bbox_inches="tight")
    plt.close(fig)


def _load_diffcsv(path: Path, user_alias: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"potential", "method", "mean", "sem"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    alias = DEFAULT_ALIASES.copy()
    alias.update(user_alias or {})

    def norm(m):
        mk  = str(m)
        low = mk.lower().replace("-", "_")
        if mk in alias:  return alias[mk]
        if low in alias: return alias[low]
        if low in {"jkonet", "jkonet_star", "jkonet_10000"} or low.startswith("jkonet_"):
            return low
        return mk

    df["method"] = df["method"].map(norm)
    return df

def _as_bxp_groups(df: pd.DataFrame, pot_order: List[str], methods: List[str]) -> Dict[str, List[dict | None]]:
    out = {m: [] for m in methods}
    for pot in pot_order:
        block = df[df["potential"] == pot]
        look = {str(r["method"]): r for _, r in block.iterrows()}
        for m in methods:
            if m in look:
                r = look[m]
                out[m].append({
                    "label": METHOD_LABEL.get(m, m),
                    "whislo": float(r["lw"]),
                    "q1":     float(r["q1"]),
                    "med":    float(r["median"]),
                    "q3":     float(r["q3"]),
                    "whishi": float(r["uw"]),
                })
            else:
                out[m].append(None)
    return out

def _symmetric_offsets(n: int, step: float) -> list[float]:
    idx = np.arange(n) - (n - 1) / 2.0
    return (idx * step).tolist()

def _overlay_markers(ax, box_patch, *,
                     marker="*", size=28, color="black",
                     ncols=3, nrows=2, margin=0.18):
    """
    Overlay filled markers inside a box patch (clipped to the box).
    - marker: matplotlib marker (e.g., "*", "o", "s")
    - size: marker size in points^2 (matplotlib 's')
    - ncols, nrows: how many markers in a grid
    - margin: fraction of box width/height to keep as padding
    """
    # robust bbox (accounting for the patch transform)
    path = box_patch.get_path()
    trans = box_patch.get_transform()          # path -> display
    bbox_disp = path.get_extents(trans)
    inv = ax.transData.inverted()
    (xmin, ymin) = inv.transform((bbox_disp.xmin, bbox_disp.ymin))
    (xmax, ymax) = inv.transform((bbox_disp.xmax, bbox_disp.ymax))

    w, h = xmax - xmin, ymax - ymin
    if w <= 0 or h <= 0:
        return

    # grid centers with margins
    xs = np.linspace(xmin + margin*w, xmax - margin*w, ncols)
    ys = np.linspace(ymin + margin*h, ymax - margin*h, nrows)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    sc = ax.scatter(XX.ravel(), YY.ravel(),
                    marker=marker, s=size,
                    facecolors=color, edgecolors='none',
                    linewidths=0, zorder=3)
    sc.set_clip_path(box_patch)

# -------------- drawing --------------
def _draw_grouped_bxp(ax,
                      bxp_groups: Dict[str, List[dict | None]],
                      *,
                      methods: List[str],
                      face_colors: Dict[str, str],
                      box_width: float = 0.32,
                      group_gap: float = 1.35,
                      offset: float = 0.35,
                      pot_order: List[str] = DEFAULT_POT_ORDER) -> None:
    x0 = 1.0
    offs = _symmetric_offsets(len(methods), offset)
    for j, m in enumerate(methods):
        stats_all = bxp_groups[m]
        positions = [x0 + i*group_gap + offs[j] for i, s in enumerate(stats_all) if s]
        stats     = [s for s in stats_all if s]
        if not stats:
            continue
        coll = ax.bxp(stats, positions=positions, widths=box_width,
                      showfliers=False, patch_artist=True)
        face  = face_colors.get(m, "#CCCCCC")
        hatch = METHOD_HATCH.get(m, None)
        for patch in coll["boxes"]:
            patch.set_facecolor(face)
            patch.set_edgecolor("black")
            if hatch:  # applies to non-JKONet now
                patch.set_hatch(hatch)

            # --- filled stars for JKONet* variants ---
            if m in {"jkonet_star", "jkonet_10000"}:
                _overlay_markers(ax, patch,
                                 marker="*",
                                 size=28,
                                 color="black",
                                 ncols=3, nrows=2,
                                 margin=0.18)
                # DO NOT call patch.set_hatch(...) again here
        for part in ("medians", "whiskers", "caps"):
            for ln in coll[part]:
                ln.set_color("black")

    ax.set_xticks([x0 + i*group_gap for i in range(len(pot_order))])
    ax.set_xticklabels([PRETTY_POT.get(p, p) for p in pot_order], fontsize=12)

    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10], prune=None))
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)


# -------------- figures --------------
def make_figure(mae_csv: Path, cos_csv: Path, out_pdf: Path,
                *, pot_order=DEFAULT_POT_ORDER, user_alias: Dict[str, str] | None = None,
                mae_yticks=None, cos_yticks=None, methods: List[str] | None = None, legend: bool = False):
    user_alias = user_alias or {}
    df_mae = _load_boxcsv(mae_csv, user_alias)
    df_cos = _load_boxcsv(cos_csv, user_alias)

    present = list(sorted(set(df_mae["method"]).union(df_cos["method"]),
                          key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 1e9))
    active = [m for m in (methods or METHOD_ORDER) if m in present]

    g_mae = _as_bxp_groups(df_mae, pot_order, active)
    g_cos = _as_bxp_groups(df_cos, pot_order, active)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.7, 6.2), sharex=True)

    _draw_grouped_bxp(ax1, g_mae,
                      methods=active,
                      face_colors={
                          "wot": ORANGE_WOT, "SBIRR": ORANGE_SBIRR,
                          "nn_appex": ORANGE_APPEX, "jkonet_star": ORANGE_JKONET,
                          "jkonet_10000": ORANGE_JKONET, "jkonet": ORANGE_JKONET
                      },
                      pot_order=pot_order)
    ax1.set_ylabel(r"Error", fontsize=12)
    ax1.tick_params(axis="x", which="both", labelbottom=True)
    if mae_yticks is not None:
        a, b, ticks = mae_yticks
        ax1.set_yticks(ticks)
        ax1.set_ylim(a - 0.03 * (b - a), b + 0.03 * (b - a) )  # add 2% pad below bottom

    _draw_grouped_bxp(ax2, g_cos,
                      methods=active,
                      face_colors={
                          "wot": BLUE_WOT, "SBIRR": BLUE_SBIRR,
                          "nn_appex": BLUE_APPEX, "jkonet_star": BLUE_JKONET,
                          "jkonet_10000": BLUE_JKONET, "jkonet": BLUE_JKONET
                      },
                      pot_order=pot_order)
    ax2.set_ylabel(r"Cosine similarity", fontsize=12)
    ax2.tick_params(axis="x", which="both", labelbottom=True)
    if cos_yticks is not None:
        a, b, ticks = cos_yticks
        ax2.set_ylim(a, b); ax2.set_yticks(ticks)
        ax2.set_ylim(a - 0.03 * (b - a), b + 0.03 * (b - a) )
    fig.subplots_adjust(**SUBPLOT_ADJ)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def make_separate_panels(mae_csv: Path, cos_csv: Path, out_dir: Path,
                         *, pot_order=DEFAULT_POT_ORDER, user_alias: Dict[str, str] | None = None,
                         mae_yticks=None, cos_yticks=None, methods: List[str] | None = None, p0=None, legend: bool = False):
    user_alias = user_alias or {}
    out_dir.mkdir(parents=True, exist_ok=True)

    # MAE
    df_mae = _load_boxcsv(mae_csv, user_alias)
    present_mae = list(sorted(set(df_mae["method"]),
                              key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 1e9))
    active_mae = [m for m in (methods or METHOD_ORDER) if m in present_mae]
    g = _as_bxp_groups(df_mae, pot_order, active_mae)

    fig, ax = plt.subplots(figsize=(10.7, 3.0))
    _draw_grouped_bxp(ax, g,
        methods=active_mae,
        face_colors={
            "wot": ORANGE_WOT, "SBIRR": ORANGE_SBIRR,
            "nn_appex": ORANGE_APPEX, "jkonet_star": ORANGE_JKONET,
            "jkonet_10000": ORANGE_JKONET, "jkonet": ORANGE_JKONET
        },
        pot_order=pot_order)
    ax.set_ylabel(r"Error", fontsize=12)
    if mae_yticks is not None:
        a, b, ticks = mae_yticks
        ax.set_yticks(ticks)
        ax.set_ylim(a - 0.03 * (b - a), b + 0.03 * (b - a) )  # add 2% pad below bottom

    if legend:
        draw_legend(
            fig,
            active_mae,
            {
                "wot": ORANGE_WOT, "SBIRR": ORANGE_SBIRR,
                "nn_appex": ORANGE_APPEX, "jkonet_star": ORANGE_JKONET,
                "jkonet_10000": ORANGE_JKONET, "jkonet": ORANGE_JKONET
            },
            METHOD_HATCH
        )
        fig.subplots_adjust(bottom=0.35)  # extra space for legend
    fig.subplots_adjust(**SUBPLOT_ADJ)
    if p0 is not None:
        mae_filename = f'grid_box_mae_{p0}.pdf'
    else:
        mae_filename = f'grid_box_mae.pdf'
    fig.savefig(out_dir / mae_filename, bbox_inches="tight")
    plt.close(fig)

    # COS
    df_cos = _load_boxcsv(cos_csv, user_alias)
    present_cos = list(sorted(set(df_cos["method"]),
                              key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 1e9))
    active_cos = [m for m in (methods or METHOD_ORDER) if m in present_cos]
    g = _as_bxp_groups(df_cos, pot_order, active_cos)

    fig, ax = plt.subplots(figsize=(10.7, 3.0))
    _draw_grouped_bxp(ax, g,
        methods=active_cos,
        face_colors={
            "wot": BLUE_WOT, "SBIRR": BLUE_SBIRR,
            "nn_appex": BLUE_APPEX, "jkonet_star": BLUE_JKONET,
            "jkonet_10000": BLUE_JKONET, "jkonet": BLUE_JKONET
        },
        pot_order=pot_order)
    ax.set_ylabel(r"Cosine similarity", fontsize=12)
    if cos_yticks is not None:
        a, b, ticks = cos_yticks
        ax.set_ylim(a, b); ax.set_yticks(ticks)
        ax.set_ylim(a - 0.03 * (b - a), b + 0.03 * (b - a) )
    if legend:
        draw_legend(
            fig,
            active_cos,
            {
                "wot": BLUE_WOT, "SBIRR": BLUE_SBIRR,
                "nn_appex": BLUE_APPEX, "jkonet_star": BLUE_JKONET,
                "jkonet_10000": BLUE_JKONET, "jkonet": BLUE_JKONET
            },
            METHOD_HATCH
        )
        fig.subplots_adjust(bottom=0.35)  # extra space for legend
    fig.subplots_adjust(**SUBPLOT_ADJ)
    if p0 is not None:
        cos_filename = f'grid_box_cos_{p0}.pdf'
    else:
        cos_filename = f'grid_box_cos.pdf'
    fig.savefig(out_dir / cos_filename, bbox_inches="tight")
    plt.close(fig)

# -------------- CLI helpers --------------
def _parse_aliases(s: str | None) -> Dict[str, str]:
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        out[k.strip()] = v.strip()
    return out

def _parse_yticks(s: str | None):
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f'Bad yticks spec: {s!r} (need "start,stop,step")')
    a, b, h = map(float, parts)
    ticks = list(np.arange(a, b + 1e-12, h))
    return (a, b, ticks)

def _normalize_method_name(name: str, alias_map: Dict[str, str]) -> str:
    if not name:
        return name
    # apply user/default aliases
    if name in alias_map:      name = alias_map[name]
    else:
        low = name.lower().replace("-", "_")
        if low in alias_map:   name = alias_map[low]
        else:
            # keep distinct JK variants; normalize formatting only
            if low in {"jkonet", "jkonet_star", "jkonet_10000"} or low.startswith("jkonet_"):
                name = low
    return name

# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae", help="box-agg CSV for MAE (lw,q1,median,q3,uw)")
    ap.add_argument("--cos", help="box-agg CSV for cosine")
    ap.add_argument("--diff", default=None,
                    help="CSV for diffusivity MAE (columns: potential,method,mean,sem[,n])")
    ap.add_argument("--combined", default=None, help="single PDF with both panels")
    ap.add_argument("--outdir", default=None, help="folder for separate MAE/COS PDFs")
    # per-panel yticks
    ap.add_argument("--mae-yticks", default="0,2.0,0.1",
                    help='y ticks for MAE as "start,stop,step"')
    ap.add_argument("--cos-yticks", default="0,1,0.1",
                    help='y ticks for cosine as "start,stop,step"')
    ap.add_argument("--pots", default=",".join(DEFAULT_POT_ORDER),
                    help="comma-separated potential order")
    ap.add_argument("--p0", type=str, default="gmm",
                   choices=["gmm", "unif", "gibbs"],
                   help="Initial distribution mode; used to tag output files.")
    ap.add_argument("--alias", default="", help='method mapping, e.g. "appex:nn_APPEX"')
    ap.add_argument("--methods", default="",
                    help="subset/order of methods (after aliasing), e.g. 'appex,jkonet_star' or 'jkonet_star,jkonet_10000'")
    ap.add_argument("--diff-yticks", default="",
                    help='y ticks for diffusivity as "start,stop,step" (optional)')
    ap.add_argument("--legend", action="store_true", help="Plot the legend")
    args = ap.parse_args()
    def _parse_yticks_opt(s):
        return _parse_yticks(s) if s else None

    diff_ticks = _parse_yticks_opt(args.diff_yticks)

    mae_ticks = _parse_yticks(args.mae_yticks)
    cos_ticks = _parse_yticks(args.cos_yticks)
    pot_order = [p.strip() for p in args.pots.split(",") if p.strip()]
    user_alias = _parse_aliases(args.alias)

    if args.mae and args.cos:
        mae_csv, cos_csv = Path(args.mae), Path(args.cos)



    # normalize requested method names
    requested_methods = None
    if args.methods.strip():
        alias_map = DEFAULT_ALIASES.copy()
        alias_map.update(user_alias)
        requested_methods = [
            _normalize_method_name(m.strip(), alias_map)
            for m in args.methods.split(",") if m.strip()
        ]

    if args.diff:
        make_diffusivity_bars(Path(args.diff), Path(args.outdir), p0=args.p0)

    if args.combined:
        make_figure(mae_csv, cos_csv, Path(args.combined),
                    pot_order=pot_order, user_alias=user_alias,
                    mae_yticks=mae_ticks, cos_yticks=cos_ticks,
                    methods=requested_methods, legend=args.legend)

    if args.outdir and args.mae and args.cos:
        make_separate_panels(mae_csv, cos_csv, Path(args.outdir),
                             pot_order=pot_order, user_alias=user_alias,
                             mae_yticks=mae_ticks, cos_yticks=cos_ticks,
                             methods=requested_methods, p0=args.p0, legend=args.legend)


if __name__ == "__main__":
    main()