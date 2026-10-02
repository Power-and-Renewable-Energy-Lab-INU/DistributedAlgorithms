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
#                     R_BOLT, LW).
#
# Usage:
#   python services/plottings.py --bus 13bus_base
#
# Each of the four figure types can be skipped independently with
# --skip-operation-plot, --skip-convergence-plot, --skip-topology-plot, and
# --skip-iteration-plots.
# =============================================================================

# Author:      Talha Rehman                  (Incheon National University)
# Co-authors:  Muhammad Ahsan Khan           (Incheon National University)
#               Woon-Gyu Lee                  (Incheon National University)
#               Hyeong-Jun Yoo                (KERI)
#               Hak-Man Kim (Corresponding)   (Incheon National University)

import argparse
import json
import re
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D

plt.switch_backend("Agg")
plt.rcParams.update(
    {
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
        "axes.edgecolor": "#444C56",
        "axes.linewidth": 1.2,
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
    }
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGSIZE = (6, 4)
COLORS = {
    "dg": "#0A4FA3",
    "res": "#1A7A3C",
    "ess": "#C45C00",
    "grid": "#5B2D8E",
    "shed": "#A30000",
    "load": "#111111",
    "gold": "#7A6200",
}

_FALLBACK_COLORS = cycle(
    ["#7A6200", "#A30000", "#5B2D8E", "#00695C", "#8E5B2D", "#2D4A8E", "#B5006D"]
)
CATEGORY_COLORS = defaultdict(
    lambda: next(_FALLBACK_COLORS),
    {
        "DG": COLORS["dg"],
        "RES": COLORS["res"],
        "ESS": COLORS["ess"],
        "LOAD": COLORS["load"],
        "GRID": COLORS["grid"],
    },
)
LAMBDA_COLUMN = re.compile(r"^([A-Za-z]+)\d+_lambda$")
TIMESTEP_FILE = re.compile(r"^timestep_(\d+)\.csv$")

DEFAULT_TOPOLOGY_PARAMS = {
    "R_NODE": 0.19,
    "R_RES": 0.27,
    "R_DG": 0.33,
    "R_ESS": 0.40,
    "R_GRID": 0.34,
    "R_BOLT": 0.20,
    "LW": 1.7,
}
TOPOLOGY_TABLES = ("bus", "lines", "demand", "dg", "res", "ess", "grid", "switch")
TOPO_LABEL_OFFSET = 0.34
TOPO_COLOR_LINE = "#3a3a3a"
TOPO_COLOR_NODE = "black"
TOPO_COLOR_LOAD = "purple"
TOPO_COLOR_DG = "#d62728"
TOPO_COLOR_RES = "#1a8a3c"
TOPO_COLOR_ESS = "#1f5fd6"
TOPO_COLOR_BOLT_FACE = "red"
TOPO_COLOR_BOLT_EDGE = "black"
BOLT_PATH = mpath.Path(
    [
        (0.10, 1.00),
        (-0.50, 0.10),
        (-0.10, 0.10),
        (-0.30, -1.00),
        (0.50, 0.20),
        (0.05, 0.20),
        (0.10, 1.00),
    ],
    [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 5 + [mpath.Path.CLOSEPOLY],
)


def save_figure(fig, save_path, **savefig_kwargs):
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", **savefig_kwargs)
    plt.close(fig)
    print(f"  Saved {save_path}")


def set_thinned_xticks(ax, values):
    ax.set_xticks(
        values, [str(int(v)) if i % 2 == 0 else "" for i, v in enumerate(values)]
    )


def build_aggregate_frame(df):
    def total(pattern):
        return df.filter(regex=pattern).sum(axis=1)

    return pd.DataFrame(
        {
            "timestep": df["timestep"],
            "convergence_iteration": df.get("convergence_iteration", np.nan),
            "dg": total(r"^DG\d+_value$"),
            "res": total(r"^RES\d+_value$"),
            "ess": total(r"^ESS\d+_value$"),
            "load": total(r"^LOAD\d+_value$"),
            "grid": total(r"^(GRID\d*_value|p_grid)$"),
            "res_curtail": -df.filter(regex=r"^RES\d+_curtail$").abs().sum(axis=1),
            "shed": df.filter(regex=r"^LOAD\d+_shed$").abs().sum(axis=1),
        }
    )


def plot_operation_24h(agg, save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    hours = agg["timestep"]

    above = {
        "dg": agg["dg"],
        "res": agg["res"],
        "ess": agg["ess"].clip(lower=0),
        "grid": agg["grid"].clip(lower=0),
    }
    below = {"ess": agg["ess"].clip(upper=0), "grid": agg["grid"].clip(upper=0)}
    if agg["shed"].sum() > 0:
        above["shed"] = agg["shed"]
    if agg["res_curtail"].ne(0).any():
        below["gold"] = agg["res_curtail"]

    for layers in (above, below):
        bottom = np.zeros(len(agg))
        for color, values in layers.items():
            ax.bar(
                hours,
                values,
                0.7,
                bottom=bottom,
                color=COLORS[color],
                edgecolor="black",
                linewidth=0.5,
            )
            bottom = bottom + values.to_numpy()

    ax.plot(
        hours,
        agg["load"],
        linestyle="--",
        marker="D",
        color=COLORS["load"],
        markersize=3.5,
        linewidth=1.4,
    )
    ax.axhline(y=0, color="#000000", linewidth=0.8)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Power [kW]")
    set_thinned_xticks(ax, hours)
    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS["dg"], label="$p^{dg}_t$"),
            mpatches.Patch(color=COLORS["res"], label="$p^{res}_t$"),
            mpatches.Patch(color=COLORS["ess"], label=r"$p^{ess +/-}_t$"),
            mpatches.Patch(color=COLORS["grid"], label=r"$p^{grid\ buy/sell}_t$"),
            mpatches.Patch(color=COLORS["shed"], label="$p^{shed}_t$"),
            mpatches.Patch(color=COLORS["gold"], label="$p^{curtail}_t$"),
            Line2D(
                [0],
                [0],
                linestyle="--",
                marker="D",
                color=COLORS["load"],
                markersize=3.5,
                linewidth=1.4,
                label="Load",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.7),
        borderpad=0.3,
        handlelength=1.2,
        handletextpad=0.3,
    )
    save_figure(fig, save_path)


def plot_convergence(agg, save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        agg["timestep"],
        agg["convergence_iteration"],
        s=45,
        facecolor=COLORS["dg"],
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xlabel("Interval")
    ax.set_ylabel("Iteration #")
    set_thinned_xticks(ax, agg["timestep"])
    save_figure(fig, save_path)


def plot_lambda_convergence(csv_path, save_path):
    df = pd.read_csv(csv_path)
    categories = {
        col: m.group(1) for col in df.columns if (m := LAMBDA_COLUMN.match(col))
    }
    if not categories or "iteration" not in df.columns:
        return False

    with plt.rc_context({"font.size": 12, "legend.fontsize": 12}):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for col, category in categories.items():
            ax.plot(
                df["iteration"],
                df[col],
                color=CATEGORY_COLORS[category],
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\lambda$")
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=CATEGORY_COLORS[category],
                    linewidth=2.2,
                    label=category,
                )
                for category in sorted(set(categories.values()))
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fancybox=False,
            handlelength=1.4,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        save_figure(fig, save_path)
    return True


def run_iteration_convergence_plots(results_dir, plot_dir):
    iteration_dir = results_dir / "iteration_results"
    if not iteration_dir.is_dir():
        print(f"  [skip] iteration_results not found at {iteration_dir}")
        return

    timestep_files = sorted(
        (int(m.group(1)), path)
        for path in iteration_dir.iterdir()
        if (m := TIMESTEP_FILE.match(path.name))
    )
    if not timestep_files:
        print(f"  [skip] no timestep_*.csv files found in {iteration_dir}")
        return

    out_dir = plot_dir / "iteration_plots"
    out_dir.mkdir(exist_ok=True)
    for timestep, csv_path in timestep_files:
        save_path = out_dir / f"timestep_{timestep}_lambda_convergence.png"
        if not plot_lambda_convergence(csv_path, save_path):
            print(f"  [skip] {csv_path}: no '*_lambda' columns found")


def load_topology_params(dataset_dir, params_path=None):
    params = dict(DEFAULT_TOPOLOGY_PARAMS)
    params_path = params_path or dataset_dir / "plotting_params.json"
    if params_path.is_file():
        params.update(json.loads(params_path.read_text()))
        print(f"  Using topology symbol sizes from {params_path}")
    else:
        print(f"  No plotting_params.json found at {params_path}; using defaults.")
    return params


def _resolve_endpoint(label, pos, grid_pos):
    kind, num = label.split("_")
    num = int(num)
    if kind == "BUS":
        return kind, num, pos[num]
    if kind == "GRID":
        return kind, num, grid_pos[num]
    raise ValueError(f"Unrecognized switch endpoint: {label}")


def plot_topology(dataset_dir, save_path, params_path=None):
    paths = {name: dataset_dir / f"{name}.csv" for name in TOPOLOGY_TABLES}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        print(f"  [skip] topology: missing files in {dataset_dir}: {missing}")
        return

    params = load_topology_params(dataset_dir, params_path)
    tables = {name: pd.read_csv(path) for name, path in paths.items()}
    switch = tables["switch"]

    pos = {
        int(r.bus_id): (float(r.geo_x), float(r.geo_y))
        for r in tables["bus"].itertuples()
    }
    grid_pos = {
        int(r.grid_id): (float(r.geo_x), float(r.geo_y))
        for r in tables["grid"].itertuples()
    }
    buses_with = {
        name: set(tables[name]["bus_id"].astype(int))
        for name in ("demand", "dg", "res", "ess")
    }
    line_pairs = {
        frozenset({int(r.from_bus), int(r.to_bus)})
        for r in tables["lines"].itertuples()
    }

    xs, ys = zip(
        *pos.values(), *grid_pos.values(), *zip(switch["geo_x"], switch["geo_y"])
    )
    margin = (
        max(params[key] for key in ("R_GRID", "R_RES", "R_DG", "R_ESS", "R_BOLT")) * 2.5
    )
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    fig, ax = plt.subplots(figsize=(16, 16 * (y_max - y_min) / (x_max - x_min)))

    for r in tables["lines"].itertuples():
        (x1, y1), (x2, y2) = pos[int(r.from_bus)], pos[int(r.to_bus)]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=TOPO_COLOR_LINE,
            linewidth=1.6,
            zorder=1,
            solid_capstyle="round",
        )
        dx, dy = x2 - x1, y2 - y1
        norm = (dx**2 + dy**2) ** 0.5
        px, py = -dy / norm, dx / norm
        if (py if abs(py) >= 1e-9 else px) < 0:
            px, py = -px, -py
        ax.text(
            (x1 + x2) / 2.0 + px * TOPO_LABEL_OFFSET,
            (y1 + y2) / 2.0 + py * TOPO_LABEL_OFFSET,
            rf"$L_{{{int(r.line_id)}}}$",
            ha="center",
            va="center",
            fontsize=10,
            zorder=8,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    for x, y in grid_pos.values():
        ax.add_patch(
            mpatches.RegularPolygon(
                (x, y),
                numVertices=4,
                radius=params["R_GRID"],
                orientation=np.pi / 4,
                facecolor="white",
                edgecolor="black",
                hatch="////",
                linewidth=params["LW"],
                zorder=6,
            )
        )
        ax.text(
            x,
            y,
            "G",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="black",
            zorder=7,
        )

    for s in switch.itertuples():
        kind_a, id_a, pa = _resolve_endpoint(s.a, pos, grid_pos)
        kind_b, id_b, pb = _resolve_endpoint(s.b, pos, grid_pos)
        if not (kind_a == kind_b == "BUS" and frozenset({id_a, id_b}) in line_pairs):
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                color=TOPO_COLOR_LINE,
                linewidth=1.6,
                zorder=1,
                solid_capstyle="round",
            )
        ax.add_patch(
            mpatches.PathPatch(
                BOLT_PATH,
                transform=Affine2D().scale(params["R_BOLT"]).translate(s.geo_x, s.geo_y)
                + ax.transData,
                facecolor=TOPO_COLOR_BOLT_FACE,
                edgecolor=TOPO_COLOR_BOLT_EDGE,
                linewidth=0.9,
                zorder=4,
            )
        )

    for bid, (x, y) in pos.items():
        if bid in buses_with["ess"]:
            ax.add_patch(
                mpatches.RegularPolygon(
                    (x, y),
                    numVertices=3,
                    radius=params["R_ESS"],
                    orientation=0,
                    facecolor="none",
                    edgecolor=TOPO_COLOR_ESS,
                    linewidth=params["LW"],
                    zorder=3,
                )
            )
        if bid in buses_with["dg"]:
            ax.add_patch(
                mpatches.Circle(
                    (x, y),
                    radius=params["R_DG"],
                    facecolor="none",
                    edgecolor=TOPO_COLOR_DG,
                    linewidth=params["LW"],
                    zorder=4,
                )
            )
        if bid in buses_with["res"]:
            ax.add_patch(
                mpatches.RegularPolygon(
                    (x, y),
                    numVertices=5,
                    radius=params["R_RES"],
                    orientation=0,
                    facecolor="none",
                    edgecolor=TOPO_COLOR_RES,
                    linewidth=params["LW"],
                    zorder=5,
                )
            )
        ax.add_patch(
            mpatches.Circle(
                (x, y),
                radius=params["R_NODE"],
                facecolor=TOPO_COLOR_LOAD
                if bid in buses_with["demand"]
                else TOPO_COLOR_NODE,
                edgecolor="black",
                linewidth=0.9,
                zorder=6,
            )
        )
        ax.text(
            x,
            y,
            str(bid),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
            zorder=7,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=TOPO_COLOR_NODE,
                markeredgecolor="black",
                markersize=11,
                label="Bus",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=TOPO_COLOR_LOAD,
                markeredgecolor="black",
                markersize=11,
                label="Bus with LOAD",
            ),
            mpatches.Patch(
                facecolor="white", edgecolor="black", hatch="////", label="Grid"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=TOPO_COLOR_DG,
                markeredgewidth=2,
                markersize=13,
                label="DG",
            ),
            Line2D(
                [0],
                [0],
                marker="p",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=TOPO_COLOR_RES,
                markeredgewidth=2,
                markersize=15,
                label="RES",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=TOPO_COLOR_ESS,
                markeredgewidth=2,
                markersize=15,
                label="ESS",
            ),
            Line2D([0], [0], color=TOPO_COLOR_LINE, linewidth=2.0, label="Line"),
            Line2D(
                [0],
                [0],
                marker=BOLT_PATH,
                linestyle="None",
                markerfacecolor=TOPO_COLOR_BOLT_FACE,
                markeredgecolor=TOPO_COLOR_BOLT_EDGE,
                markeredgewidth=1.0,
                markersize=15,
                label="Potential fault locations",
            ),
        ],
        ncol=9,
        fontsize=12,
        handletextpad=0.8,
        labelspacing=1.1,
        borderaxespad=0,
        fancybox=False,
    )
    save_figure(fig, save_path, dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot operation results, convergence, and topology for a bus system."
    )
    parser.add_argument(
        "--bus",
        "-b",
        required=True,
        help="Bus/dataset folder name (matches the --bus used in distributed.py).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "Data-v2",
        help="Root folder containing per-bus dataset folders (default: Data-v2).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root folder containing per-bus result folders (default: results).",
    )
    parser.add_argument(
        "--operation-csv",
        type=Path,
        help="Override path to operation_results.csv "
        "(default: <results-root>/<bus>/operation_results.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output folder for the plots (default: <results-root>/<bus>/plots).",
    )
    parser.add_argument(
        "--topology-params",
        type=Path,
        help="Override path to plotting_params.json "
        "(default: <data-root>/<bus>/plotting_params.json).",
    )
    parser.add_argument(
        "--skip-operation-plot",
        action="store_true",
        help="Skip generating operation.png.",
    )
    parser.add_argument(
        "--skip-convergence-plot",
        action="store_true",
        help="Skip generating convergence.png.",
    )
    parser.add_argument(
        "--skip-topology-plot",
        action="store_true",
        help="Skip generating topology.png.",
    )
    parser.add_argument(
        "--skip-iteration-plots",
        action="store_true",
        help="Skip generating the per-timestep plots/iteration_plots/ figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.data_root / args.bus
    results_dir = args.results_root / args.bus
    plot_dir = args.output_dir or results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bus system  : {args.bus}")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Results dir : {results_dir}")
    print(f"Plots dir   : {plot_dir}")

    csv_path = args.operation_csv or results_dir / "operation_results.csv"
    agg = None
    if (
        not (args.skip_operation_plot and args.skip_convergence_plot)
        and csv_path.is_file()
    ):
        agg = build_aggregate_frame(pd.read_csv(csv_path))

    for name, skip, plot in (
        ("operation", args.skip_operation_plot, plot_operation_24h),
        ("convergence", args.skip_convergence_plot, plot_convergence),
    ):
        if skip:
            print(f"  [skip] {name}.png (--skip-{name}-plot)")
            continue
        print(f"{name.capitalize()} plot:")
        if agg is None:
            print(f"  [skip] {csv_path} not found.")
        else:
            plot(agg, plot_dir / f"{name}.png")

    if args.skip_iteration_plots:
        print("  [skip] iteration_plots (--skip-iteration-plots)")
    else:
        print("Iteration (lambda) convergence plots:")
        run_iteration_convergence_plots(results_dir, plot_dir)

    if args.skip_topology_plot:
        print("  [skip] topology.png (--skip-topology-plot)")
    else:
        print("Topology plot:")
        plot_topology(dataset_dir, plot_dir / "topology.png", args.topology_params)

    print("Done.")


if __name__ == "__main__":
    main()
