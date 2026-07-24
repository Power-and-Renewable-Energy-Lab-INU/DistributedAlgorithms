#!/usr/bin/env python3
# =============================================================================
# plottings.py
# Post-run plotting CLI for the distributed-algorithm service.
#
# For a given --bus dataset it produces three figures:
#   results/<bus>/plots/operation.png    24h stacked operation plot
#   results/<bus>/plots/convergence.png  per-interval convergence iterations
#   results/<bus>/plots/topology.png     single-line diagram of the network
# plus one lambda-convergence figure per timestep that has a saved
# iteration_results CSV:
#   results/<bus>/plots/iteration_plots/timestep_<n>_lambda_convergence.png
#
# operation.png / convergence.png are built from
#   results/<bus>/operation_results.csv
# the per-timestep iteration_plots/ figures are built from
#   results/<bus>/iteration_results/timestep_<n>.csv
# topology.png is built from the network files in
#   Data-v2/<bus>/  (bus.csv, lines.csv, demand.csv, dg.csv, res.csv, ess.csv,
#                     grid.csv, switch.csv) using symbol sizes read from
#   Data-v2/<bus>/plotting_params.json  (R_NODE, R_RES, R_DG, R_ESS, R_GRID,
#                     R_BOLT, LW) -- see plotting_params.sample.json.
#
# Usage:
#   python services/plottings.py --bus 13bus_base
# =============================================================================

# Author:      Talha Rehman                  (Incheon National University)
# Co-authors:  Muhammad Ahsan Khan           (Incheon National University)
#               Woon-Gyu Lee                  (Incheon National University)
#               Hyeong-Jun Yoo                (KERI)
#               Hak-Man Kim (Corresponding)   (Incheon National University)

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style — publication-quality (same as ResultsAnalysis.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "font.family": "serif",
    "font.size": 14,

    "axes.labelsize": 12,
    "axes.titlesize": 12,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,

    "xtick.color": "#333333",
    "ytick.color": "#333333",

    "axes.labelcolor": "#222222",
    "text.color": "#222222",

    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.5,

    "legend.facecolor": "white",
    "legend.edgecolor": "#000000",
    "legend.framealpha": 1.0,
    "legend.fontsize": 10,
})

# Colors - rich dark palette, perceptually distinct (same as ResultsAnalysis.py)
COLORS = {
    "dg":   "#0A4FA3",   # deep royal blue
    "res":  "#1A7A3C",   # dark forest green
    "ess":  "#C45C00",   # burnt dark orange
    "grid": "#5B2D8E",   # deep violet
    "shed": "#A30000",   # dark crimson
    "load": "#111111",   # near-black
    "gold": "#7A6200",   # dark gold / ochre
}

SPINE_COLOR = "#444C56"
SPINE_LW = 1.2

FIGSIZE = (6, 4)


# ===========================================================================
# operation_results.csv  ->  operation.png  +  convergence.png
# ===========================================================================

def _cols_matching(df, pattern):
    rx = re.compile(pattern)
    return [c for c in df.columns if rx.match(c)]


def build_aggregate_frame(df):
    """Collapse the per-unit operation_results.csv columns (DGn_value,
    RESn_value, RESn_curtail, LOADn_value, LOADn_shed, ESSn_value, ...)
    into one row per timestep with system-level totals. Works regardless
    of how many DG/RES/LOAD/ESS units are present in the dataset."""
    dg_cols       = _cols_matching(df, r"^DG\d+_value$")
    res_cols      = _cols_matching(df, r"^RES\d+_value$")
    rescurt_cols  = _cols_matching(df, r"^RES\d+_curtail$")
    ess_cols      = _cols_matching(df, r"^ESS\d+_value$")
    load_cols     = _cols_matching(df, r"^LOAD\d+_value$")
    shed_cols     = _cols_matching(df, r"^LOAD\d+_shed$")
    grid_cols     = _cols_matching(df, r"^(GRID\d*_value|p_grid)$")

    agg = pd.DataFrame()
    agg["timestep"] = df["timestep"].values
    agg["convergence_iteration"] = (
        df["convergence_iteration"].values if "convergence_iteration" in df.columns
        else np.full(len(df), np.nan)
    )

    agg["dg"]   = df[dg_cols].sum(axis=1).values if dg_cols else np.zeros(len(df))
    agg["res"]  = df[res_cols].sum(axis=1).values if res_cols else np.zeros(len(df))
    agg["ess"]  = df[ess_cols].sum(axis=1).values if ess_cols else np.zeros(len(df))
    agg["load"] = df[load_cols].sum(axis=1).values if load_cols else np.zeros(len(df))
    agg["grid"] = df[grid_cols].sum(axis=1).values if grid_cols else np.zeros(len(df))

    # curtailment / shed are magnitudes; normalize sign regardless of how
    # they were stored in the source CSV (curtail plotted below zero,
    # shed stacked above zero, same convention as ResultsAnalysis.py).
    agg["res_curtail"] = (-df[rescurt_cols].abs().sum(axis=1).values
                           if rescurt_cols else np.zeros(len(df)))
    agg["shed"] = (df[shed_cols].abs().sum(axis=1).values
                   if shed_cols else np.zeros(len(df)))

    return agg


def _thinned_xticklabels(values):
    return [str(int(v)) if i % 2 == 0 else "" for i, v in enumerate(values)]


def plot_operation_24h(agg, save_path):
    """24h stacked operation plot — same visual language as
    ResultsAnalysis.py::plot_operation_24h (fixed 7-entry legend, DG/RES/ESS/
    grid/shed stacked positive, ESS-charge/grid-export/curtailment stacked
    negative, dashed load line on top)."""
    hours     = agg["timestep"].values
    bar_width = 0.7

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ess_dchg = np.maximum(agg["ess"].values, 0)
    ess_chg  = np.minimum(agg["ess"].values, 0)
    grid_pos = np.maximum(agg["grid"].values, 0)
    grid_neg = np.minimum(agg["grid"].values, 0)

    bottom_pos = np.zeros(len(agg))

    ax.bar(hours, agg["dg"].values, bar_width, bottom=bottom_pos,
           color=COLORS["dg"], edgecolor="black", linewidth=0.5)
    bottom_pos += agg["dg"].values

    ax.bar(hours, agg["res"].values, bar_width, bottom=bottom_pos,
           color=COLORS["res"], edgecolor="black", linewidth=0.5)
    bottom_pos += agg["res"].values

    ax.bar(hours, ess_dchg, bar_width, bottom=bottom_pos,
           color=COLORS["ess"], edgecolor="black", linewidth=0.5)
    bottom_pos += ess_dchg

    ax.bar(hours, grid_pos, bar_width, bottom=bottom_pos,
           color=COLORS["grid"], edgecolor="black", linewidth=0.5)
    bottom_pos += grid_pos

    if agg["shed"].sum() > 0:
        ax.bar(hours, agg["shed"].values, bar_width, bottom=bottom_pos,
               color=COLORS["shed"], edgecolor="black", linewidth=0.5)

    bottom_neg = np.zeros(len(agg))
    ax.bar(hours, ess_chg, bar_width, bottom=bottom_neg,
           color=COLORS["ess"], edgecolor="black", linewidth=0.5)
    bottom_neg += ess_chg
    ax.bar(hours, grid_neg, bar_width, bottom=bottom_neg,
           color=COLORS["grid"], edgecolor="black", linewidth=0.5)
    bottom_neg += grid_neg

    if np.any(agg["res_curtail"].values != 0):
        ax.bar(hours, agg["res_curtail"].values, bar_width, bottom=bottom_neg,
               color=COLORS["gold"], edgecolor="black", linewidth=0.5)

    ax.plot(hours, agg["load"].values, linestyle="--", marker="D",
            color=COLORS["load"], markersize=3.5, linewidth=1.4)

    # Fixed legend — all 7 entries always present, colors never change
    handles = [
        mpatches.Patch(color=COLORS["dg"],   label="$p^{dg}_t$"),
        mpatches.Patch(color=COLORS["res"],  label="$p^{res}_t$"),
        mpatches.Patch(color=COLORS["ess"],  label=r"$p^{ess +/-}_t$"),
        mpatches.Patch(color=COLORS["grid"], label=r"$p^{grid\ buy/sell}_t$"),
        mpatches.Patch(color=COLORS["shed"], label="$p^{shed}_t$"),
        mpatches.Patch(color=COLORS["gold"], label="$p^{curtail}_t$"),
        Line2D([0], [0], linestyle="--", marker="D", color=COLORS["load"],
               markersize=3.5, linewidth=1.4, label="Load"),
    ]
    labels = [h.get_label() for h in handles]

    ax.axhline(y=0, color="#000000", linewidth=0.8)

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Power [kW]")
    ax.set_xticks(hours)
    ax.set_xticklabels(_thinned_xticklabels(hours))
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_edgecolor(SPINE_COLOR)

    ax.legend(handles, labels,
              loc="center left", bbox_to_anchor=(1.01, 0.7),
              ncol=1, frameon=True,
              borderpad=0.3, handlelength=1.2, columnspacing=0.6,
              handletextpad=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_convergence(agg, save_path):
    """Per-interval convergence iterations — round markers, blue fill,
    black edges. x-axis: Interval, y-axis: Iteration #."""
    x = agg["timestep"].values
    y = agg["convergence_iteration"].values

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.scatter(x, y, s=45, marker="o",
               facecolor=COLORS["dg"], edgecolor="black",
               linewidth=0.8, zorder=3)

    ax.set_xlabel("Interval")
    ax.set_ylabel("Iteration #")
    ax.set_xticks(x)
    ax.set_xticklabels(_thinned_xticklabels(x))
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_edgecolor(SPINE_COLOR)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_operation_plots(results_dir, plot_dir):
    csv_path = os.path.join(results_dir, "operation_results.csv")
    if not os.path.isfile(csv_path):
        print(f"  [skip] operation_results.csv not found at {csv_path}")
        return

    df  = pd.read_csv(csv_path)
    agg = build_aggregate_frame(df)

    plot_operation_24h(agg, os.path.join(plot_dir, "operation.png"))
    print(f"  Saved {os.path.join(plot_dir, 'operation.png')}")

    plot_convergence(agg, os.path.join(plot_dir, "convergence.png"))
    print(f"  Saved {os.path.join(plot_dir, 'convergence.png')}")


# ===========================================================================
# iteration_results/timestep_<n>.csv  ->  plots/iteration_plots/*.png
#
# One lambda-convergence figure per timestep CSV. All agents of the same
# category (DG, RES, ESS, LOAD, ...) share one color and one legend entry,
# regardless of how many individual agents of that category exist.
# ===========================================================================

# Category -> color, reusing the same palette as the operation plot so the
# same agent type always reads as the same color across every figure in the
# report.
ITERATION_CATEGORY_COLORS = {
    "DG":   COLORS["dg"],
    "RES":  COLORS["res"],
    "ESS":  COLORS["ess"],
    "LOAD": COLORS["load"],
    "GRID": COLORS["grid"],
}

# Fallback palette for any agent category not covered above (e.g. a new
# dataset introduces a "STORAGE" or "EV" agent type) -- colors are handed
# out in the order new categories are first encountered and then reused
# consistently for the rest of the run.
_ITERATION_FALLBACK_PALETTE = [
    "#7A6200", "#A30000", "#5B2D8E", "#00695C", "#8E5B2D", "#2D4A8E", "#B5006D",
]
_iteration_fallback_cache = {}


def _agent_category(column_name):
    """'DG1_lambda' -> 'DG', 'RES12_lambda' -> 'RES'. Returns None for
    columns that don't match the '<Category><index>_lambda' pattern."""
    m = re.match(r"^([A-Za-z]+)\d+_lambda$", column_name)
    return m.group(1) if m else None


def _category_color(category):
    if category in ITERATION_CATEGORY_COLORS:
        return ITERATION_CATEGORY_COLORS[category]
    if category not in _iteration_fallback_cache:
        idx = len(_iteration_fallback_cache) % len(_ITERATION_FALLBACK_PALETTE)
        _iteration_fallback_cache[category] = _ITERATION_FALLBACK_PALETTE[idx]
    return _iteration_fallback_cache[category]


def plot_lambda_convergence(iteration_csv_path, save_path, timestep_label):
    """Plot lambda (consensus price) vs. iteration for every agent found in
    a single timestep_<n>.csv, colored by agent category with one legend
    entry per category, legend placed above the axes."""
    df = pd.read_csv(iteration_csv_path)
    lambda_cols = [c for c in df.columns if c.endswith("_lambda")]
    if not lambda_cols or "iteration" not in df.columns:
        return False

    iterations = df["iteration"].values

    # Font size 12, serif family, applied to title/labels/ticks/legend
    with plt.rc_context({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        seen_categories = {}
        for col in lambda_cols:
            category = _agent_category(col)
            if category is None:
                continue

            color = _category_color(category)

            # Keep only positive values for log-log plotting
           # mask = (iterations > 0) & (df[col].values > 0)

            ax.plot(
                iterations,
                df[col].values,
                color=color,
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )

            seen_categories.setdefault(category, color)

        legend_handles = [
            Line2D([0], [0], color=color, linewidth=2.2, label=category)
            for category, color in sorted(seen_categories.items())
        ]

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\lambda$")

          # Set log-log scale
        #ax.set_xscale("log")
        #ax.set_yscale("log")

        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
            spine.set_edgecolor(SPINE_COLOR)

        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            handlelength=1.4,
            columnspacing=1.0,
            handletextpad=0.5,
        )

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    return True


def run_iteration_convergence_plots(results_dir, plot_dir):
    """Generate one lambda-convergence figure per timestep_<n>.csv found in
    results/<bus>/iteration_results/, saved to plots/iteration_plots/."""
    iter_dir = os.path.join(results_dir, "iteration_results")
    if not os.path.isdir(iter_dir):
        print(f"  [skip] iteration_results not found at {iter_dir}")
        return

    out_dir = os.path.join(plot_dir, "iteration_plots")
    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(r"^timestep_(\d+)\.csv$")
    timestep_files = []
    for fname in os.listdir(iter_dir):
        m = pattern.match(fname)
        if m:
            timestep_files.append((int(m.group(1)), fname))
    timestep_files.sort(key=lambda t: t[0])

    if not timestep_files:
        print(f"  [skip] no timestep_*.csv files found in {iter_dir}")
        return

    for n, fname in timestep_files:
        csv_path  = os.path.join(iter_dir, fname)
        save_path = os.path.join(out_dir, f"timestep_{n}_lambda_convergence.png")
        ok = plot_lambda_convergence(csv_path, save_path, timestep_label=n)
        if ok:
            print(f"  Saved {save_path}")
        else:
            print(f"  [skip] {csv_path}: no '*_lambda' columns found")


# ===========================================================================
# Topology single-line diagram  ->  topology.png
# ===========================================================================

DEFAULT_TOPOLOGY_PARAMS = {
    "R_NODE": 0.19,
    "R_RES": 0.27,
    "R_DG": 0.33,
    "R_ESS": 0.40,
    "R_GRID": 0.34,
    "R_BOLT": 0.20,
    "LW": 1.7,
}

_BOLT_VERTS = [
    (0.10, 1.00),
    (-0.50, 0.10),
    (-0.10, 0.10),
    (-0.30, -1.00),
    (0.50, 0.20),
    (0.05, 0.20),
    (0.10, 1.00),
]
_BOLT_CODES = [Path.MOVETO] + [Path.LINETO] * 5 + [Path.CLOSEPOLY]
BOLT_PATH = Path(_BOLT_VERTS, _BOLT_CODES)

TOPO_COLOR_LINE = "#3a3a3a"
TOPO_COLOR_NODE = "black"
TOPO_COLOR_LOAD = "purple"
TOPO_COLOR_DG = "#d62728"
TOPO_COLOR_RES = "#1a8a3c"
TOPO_COLOR_ESS = "#1f5fd6"
TOPO_COLOR_BOLT_FACE = "red"
TOPO_COLOR_BOLT_EDGE = "black"

REQUIRED_TOPOLOGY_FILES = [
    "bus.csv", "lines.csv", "demand.csv", "dg.csv",
    "res.csv", "ess.csv", "grid.csv", "switch.csv",
]


def load_topology_params(dataset_dir, params_override=None):
    params = dict(DEFAULT_TOPOLOGY_PARAMS)
    params_path = params_override or os.path.join(dataset_dir, "plotting_params.json")
    if os.path.isfile(params_path):
        with open(params_path, "r") as f:
            params.update(json.load(f))
        print(f"  Using topology symbol sizes from {params_path}")
    else:
        print(f"  No plotting_params.json found at {params_path}; using defaults.")
    return params


def topology_files_present(dataset_dir):
    missing = [f for f in REQUIRED_TOPOLOGY_FILES
               if not os.path.isfile(os.path.join(dataset_dir, f))]
    return (len(missing) == 0), missing


def _resolve_endpoint(label, pos, grid_pos):
    kind, num = label.split("_")
    num = int(num)
    if kind == "BUS":
        return kind, num, pos[num]
    elif kind == "GRID":
        return kind, num, grid_pos[num]
    raise ValueError(f"Unrecognized switch endpoint: {label}")


def _add_bolt(ax, x, y, r, zorder=4):
    transform = Affine2D().scale(r).translate(x, y) + ax.transData
    patch = mpatches.PathPatch(
        BOLT_PATH, transform=transform, facecolor=TOPO_COLOR_BOLT_FACE,
        edgecolor=TOPO_COLOR_BOLT_EDGE, linewidth=0.9, zorder=zorder
    )
    ax.add_patch(patch)


def plot_topology(dataset_dir, save_path, params_override=None):
    """Single-line diagram of the network, ported from the 33-bus /
    123-bus reference scripts and generalized to any --bus dataset via
    plotting_params.json (R_NODE, R_RES, R_DG, R_ESS, R_GRID, R_BOLT, LW)."""
    ok, missing = topology_files_present(dataset_dir)
    if not ok:
        print(f"  [skip] topology: missing files in {dataset_dir}: {missing}")
        return

    params = load_topology_params(dataset_dir, params_override)
    R_NODE = params["R_NODE"]
    R_RES  = params["R_RES"]
    R_DG   = params["R_DG"]
    R_ESS  = params["R_ESS"]
    R_GRID = params["R_GRID"]
    R_BOLT = params["R_BOLT"]
    LW     = params["LW"]

    bus     = pd.read_csv(os.path.join(dataset_dir, "bus.csv"))
    lines   = pd.read_csv(os.path.join(dataset_dir, "lines.csv"))
    demand  = pd.read_csv(os.path.join(dataset_dir, "demand.csv"))
    dg      = pd.read_csv(os.path.join(dataset_dir, "dg.csv"))
    ess     = pd.read_csv(os.path.join(dataset_dir, "ess.csv"))
    res     = pd.read_csv(os.path.join(dataset_dir, "res.csv"))
    grid    = pd.read_csv(os.path.join(dataset_dir, "grid.csv"))
    switch  = pd.read_csv(os.path.join(dataset_dir, "switch.csv"))

    pos = {int(r.bus_id): (float(r.geo_x), float(r.geo_y)) for r in bus.itertuples()}
    grid_pos = {int(r.grid_id): (float(r.geo_x), float(r.geo_y)) for r in grid.itertuples()}

    load_buses = set(demand["bus_id"].astype(int))
    dg_buses   = set(dg["bus_id"].astype(int))
    ess_buses  = set(ess["bus_id"].astype(int))
    res_buses  = set(res["bus_id"].astype(int))

    existing_line_pairs = {frozenset({int(r.from_bus), int(r.to_bus)}) for r in lines.itertuples()}

    xs = [p[0] for p in pos.values()] + [p[0] for p in grid_pos.values()] + list(switch["geo_x"])
    ys = [p[1] for p in pos.values()] + [p[1] for p in grid_pos.values()] + list(switch["geo_y"])
    margin = max(R_GRID, R_RES, R_DG, R_ESS, R_BOLT) * 2.5
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    fig_w = 16
    fig_h = fig_w * (y_max - y_min) / (x_max - x_min)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    LABEL_OFFSET = 0.34
    LABEL_FONTSIZE = 10

    for r in lines.itertuples():
        x1, y1 = pos[int(r.from_bus)]
        x2, y2 = pos[int(r.to_bus)]
        ax.plot([x1, x2], [y1, y2], color=TOPO_COLOR_LINE, linewidth=1.6, zorder=1,
                 solid_capstyle="round")

        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        norm = (dx ** 2 + dy ** 2) ** 0.5
        px, py = -dy / norm, dx / norm
        if abs(py) < 1e-9:
            if px < 0:
                px, py = -px, -py
        elif py < 0:
            px, py = -px, -py
        lx, ly = mx + px * LABEL_OFFSET, my + py * LABEL_OFFSET
        ax.text(
            lx, ly, rf"$L_{{{int(r.line_id)}}}$",
            ha="center", va="center", fontsize=LABEL_FONTSIZE, color="#222222",
            zorder=8, bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.85),
        )

    for gid, (x, y) in grid_pos.items():
        sq = mpatches.RegularPolygon(
            (x, y), numVertices=4, radius=R_GRID, orientation=0.785398163,
            facecolor="white", edgecolor="black", hatch="////", linewidth=LW,
            zorder=6,
        )
        ax.add_patch(sq)
        ax.text(x, y, "G", ha="center", va="center", fontsize=8.5,
                 fontweight="bold", color="black", zorder=7)

    for s in switch.itertuples():
        kind_a, id_a, pa = _resolve_endpoint(s.a, pos, grid_pos)
        kind_b, id_b, pb = _resolve_endpoint(s.b, pos, grid_pos)

        already_drawn = (
            kind_a == "BUS" and kind_b == "BUS"
            and frozenset({id_a, id_b}) in existing_line_pairs
        )
        if not already_drawn:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=TOPO_COLOR_LINE,
                     linewidth=1.6, zorder=1, solid_capstyle="round")

        _add_bolt(ax, float(s.geo_x), float(s.geo_y), R_BOLT, zorder=4)

    for bid, (x, y) in pos.items():
        if bid in ess_buses:
            tri = mpatches.RegularPolygon(
                (x, y), numVertices=3, radius=R_ESS, orientation=0,
                facecolor="none", edgecolor=TOPO_COLOR_ESS, linewidth=LW, zorder=3
            )
            ax.add_patch(tri)
        if bid in dg_buses:
            circ = mpatches.Circle(
                (x, y), radius=R_DG, facecolor="none", edgecolor=TOPO_COLOR_DG,
                linewidth=LW, zorder=4
            )
            ax.add_patch(circ)
        if bid in res_buses:
            pent = mpatches.RegularPolygon(
                (x, y), numVertices=5, radius=R_RES, orientation=0,
                facecolor="none", edgecolor=TOPO_COLOR_RES, linewidth=LW, zorder=5
            )
            ax.add_patch(pent)

        face = TOPO_COLOR_LOAD if bid in load_buses else TOPO_COLOR_NODE
        node = mpatches.Circle(
            (x, y), radius=R_NODE, facecolor=face, edgecolor="black",
            linewidth=0.9, zorder=6
        )
        ax.add_patch(node)
        ax.text(x, y, str(bid), ha="center", va="center", fontsize=10,
                 color="white", fontweight="bold", zorder=7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=TOPO_COLOR_NODE,
               markeredgecolor="black", markersize=11, label="Bus"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=TOPO_COLOR_LOAD,
               markeredgecolor="black", markersize=11, label="Bus with LOAD"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="////",
                        label="Grid"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
               markeredgecolor=TOPO_COLOR_DG, markeredgewidth=2, markersize=13,
               label="DG"),
        Line2D([0], [0], marker="p", linestyle="None", markerfacecolor="none",
               markeredgecolor=TOPO_COLOR_RES, markeredgewidth=2, markersize=15,
               label="RES"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="none",
               markeredgecolor=TOPO_COLOR_ESS, markeredgewidth=2, markersize=15,
               label="ESS"),
        Line2D([0], [0], color=TOPO_COLOR_LINE, linewidth=2.0,
               label="Line"),
        Line2D([0], [0], marker=BOLT_PATH, linestyle="None",
               markerfacecolor=TOPO_COLOR_BOLT_FACE, markeredgecolor=TOPO_COLOR_BOLT_EDGE,
               markeredgewidth=1.0, markersize=15,
               label="Potential fault locations"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="best",
        frameon=True,
        ncol=9,
        fontsize=12,
        handletextpad=0.8,
        labelspacing=1.1,
        borderaxespad=0, edgecolor="black", fancybox=False,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot operation results, convergence, and topology for a bus system."
    )
    parser.add_argument("--bus", "-b", required=True,
                         help="Bus/dataset folder name (matches the --bus used in distributed.py).")
    parser.add_argument("--data-root", default="Data-v2",
                         help="Root folder containing per-bus dataset folders (default: Data-v2).")
    parser.add_argument("--results-root", default="results",
                         help="Root folder containing per-bus result folders (default: results).")
    parser.add_argument("--operation-csv", default=None,
                         help="Override path to operation_results.csv "
                              "(default: <results-root>/<bus>/operation_results.csv).")
    parser.add_argument("--output-dir", default=None,
                         help="Override output folder for the plots "
                              "(default: <results-root>/<bus>/plots).")
    parser.add_argument("--topology-params", default=None,
                         help="Override path to plotting_params.json "
                              "(default: <data-root>/<bus>/plotting_params.json).")
    parser.add_argument("--skip-topology", action="store_true",
                         help="Skip generating topology.png.")
    parser.add_argument("--skip-iteration-plots", action="store_true",
                         help="Skip generating the per-timestep plots/iteration_plots/ figures.")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.bus)
    results_dir = os.path.join(args.results_root, args.bus)
    plot_dir    = args.output_dir or os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Bus system  : {args.bus}")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Results dir : {results_dir}")
    print(f"Plots dir   : {plot_dir}")

    print("Operation + convergence plots:")
    if args.operation_csv:
        # allow a fully custom CSV location
        if os.path.isfile(args.operation_csv):
            df  = pd.read_csv(args.operation_csv)
            agg = build_aggregate_frame(df)
            plot_operation_24h(agg, os.path.join(plot_dir, "operation.png"))
            print(f"  Saved {os.path.join(plot_dir, 'operation.png')}")
            plot_convergence(agg, os.path.join(plot_dir, "convergence.png"))
            print(f"  Saved {os.path.join(plot_dir, 'convergence.png')}")
        else:
            print(f"  [skip] {args.operation_csv} not found.")
    else:
        run_operation_plots(results_dir, plot_dir)

    if not args.skip_iteration_plots:
        print("Iteration (lambda) convergence plots:")
        run_iteration_convergence_plots(results_dir, plot_dir)

    if not args.skip_topology:
        print("Topology plot:")
        plot_topology(dataset_dir, os.path.join(plot_dir, "topology.png"),
                      params_override=args.topology_params)

    print("Done.")


if __name__ == "__main__":
    main()
