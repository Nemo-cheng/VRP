from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from evaluate_results import parse_solution
from run_unified_experiment import DEFAULT_INSTANCE, ROOT, parse_instance


MODEL_ORDER = ("vrp_spd", "vrp_tw_spd", "evrp_tw_spd")
MODEL_LABELS = {
    "vrp_spd": "VRP-SPD",
    "vrp_tw_spd": "VRPTW-SPD",
    "evrp_tw_spd": "EVRPTW-SPD",
}
MODEL_COLORS = {
    "vrp_spd": "#7A7A7A",
    "vrp_tw_spd": "#3A7D8C",
    "evrp_tw_spd": "#D06B47",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: mpl.figure.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(rows: list[dict[str, str]], output_dir: Path) -> None:
    panels = (
        ("total_cost", "Total cost (thousand)", 1_000.0),
        ("route_count", "Vehicles", 1.0),
        ("total_travel_time", "Travel time", 1.0),
        ("charging_visits", "Charging visits", 1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    offsets = (-0.10, -0.05, 0.0, 0.05, 0.10)

    for panel_index, (axis, (metric, label, scale)) in enumerate(zip(axes.flat, panels)):
        for model_index, model in enumerate(MODEL_ORDER):
            values = [
                float(row[metric]) / scale for row in rows if row["model"] == model
            ]
            mean = statistics.mean(values)
            standard_deviation = statistics.pstdev(values)
            axis.bar(
                model_index,
                mean,
                width=0.62,
                color=MODEL_COLORS[model],
                alpha=0.82,
                zorder=1,
            )
            axis.errorbar(
                model_index,
                mean,
                yerr=standard_deviation,
                color="#202020",
                capsize=3,
                linewidth=0.9,
                zorder=3,
            )
            for offset, value in zip(offsets, values):
                axis.scatter(
                    model_index + offset,
                    value,
                    s=13,
                    facecolor="white",
                    edgecolor="#202020",
                    linewidth=0.6,
                    zorder=4,
                )

        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=15)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
        axis.text(
            -0.14,
            1.04,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Constraint effects on routing outcomes", fontsize=10, fontweight="bold")
    save_figure(fig, output_dir / "model_comparison")


def select_representative_evrp(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [row for row in rows if row["model"] == "evrp_tw_spd"]
    return min(candidates, key=lambda row: (int(row["route_count"]), float(row["total_cost"])))


def plot_representative_routes(
    rows: list[dict[str, str]], instance_path: Path, output_dir: Path
) -> None:
    selected = select_representative_evrp(rows)
    instance = parse_instance(instance_path)
    solution_path = ROOT / selected["solution_file"]
    routes, total_cost, _ = parse_solution(solution_path)
    used_stations = {
        node_id
        for route in routes
        for node_id, _, _ in route
        if instance.nodes[node_id]["type"] == "f"
    }

    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    route_colors = plt.get_cmap("tab20")
    for route_index, route in enumerate(routes):
        coordinates = [
            (float(instance.nodes[node_id]["x"]), float(instance.nodes[node_id]["y"]))
            for node_id, _, _ in route
        ]
        x_values, y_values = zip(*coordinates)
        axis.plot(
            x_values,
            y_values,
            color=route_colors(route_index % 20),
            linewidth=0.75,
            alpha=0.72,
            zorder=1,
        )

    customers = [node for node in instance.nodes.values() if node["type"] == "c"]
    stations = [node for node in instance.nodes.values() if node["type"] == "f"]
    axis.scatter(
        [float(node["x"]) for node in stations],
        [float(node["y"]) for node in stations],
        marker="+",
        s=16,
        color="#B5B5B5",
        linewidth=0.6,
        label="Available charging station",
        zorder=2,
    )
    axis.scatter(
        [float(node["x"]) for node in customers],
        [float(node["y"]) for node in customers],
        s=8,
        color="#252525",
        linewidth=0,
        label="Customer",
        zorder=3,
    )
    axis.scatter(
        [float(instance.nodes[node_id]["x"]) for node_id in used_stations],
        [float(instance.nodes[node_id]["y"]) for node_id in used_stations],
        marker="X",
        s=35,
        color="#D06B47",
        edgecolor="white",
        linewidth=0.5,
        label="Used charging station",
        zorder=4,
    )
    depot = instance.nodes[instance.depot]
    axis.scatter(
        [float(depot["x"])],
        [float(depot["y"])],
        marker="s",
        s=45,
        color="#2B6F66",
        edgecolor="white",
        linewidth=0.6,
        label="Depot",
        zorder=5,
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("X coordinate")
    axis.set_ylabel("Y coordinate")
    axis.set_title(
        f"Representative EVRPTW-SPD solution | seed {selected['seed']} | "
        f"{len(routes)} vehicles | cost {total_cost:,.2f}"
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, fontsize=7)
    axis.grid(color="#E6E6E6", linewidth=0.4, alpha=0.65)
    axis.set_axisbelow(True)
    save_figure(fig, output_dir / "representative_evrp_routes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", type=Path, default=ROOT / "results" / "batch_metrics.csv"
    )
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "figures"
    )
    args = parser.parse_args()

    configure_style()
    rows = load_metrics(args.metrics.resolve())
    plot_model_comparison(rows, args.output_dir.resolve())
    plot_representative_routes(rows, args.instance.resolve(), args.output_dir.resolve())
    print(f"Figures written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
