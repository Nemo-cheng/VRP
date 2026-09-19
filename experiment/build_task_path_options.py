#!/usr/bin/env python3
"""Build feasible loaded path options using only the observed transport network."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "networkx>=3.4",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from vrp_time_utils import evaluate_time_dependent_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成载货任务候选路径。")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("processed/company/vrp")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    parser.add_argument("--paths-per-objective", type=int, default=3)
    return parser.parse_args()


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in edges[edges["high_confidence"]].itertuples(index=False):
        graph.add_edge(
            row.origin_site_id,
            row.destination_site_id,
            distance_km=float(row.distance_km_p50),
            duration_hours=float(row.duration_hours_p50),
        )
    return graph


def k_paths(
    graph: nx.DiGraph, origin: str, destination: str, weight: str, count: int
) -> list[list[str]]:
    try:
        generator = nx.shortest_simple_paths(
            graph, origin, destination, weight=weight
        )
        paths = []
        for _ in range(count):
            try:
                paths.append(next(generator))
            except StopIteration:
                break
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def build_path_options(
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    periods: pd.DataFrame,
    paths_per_objective: int,
) -> pd.DataFrame:
    graph = build_graph(edges)
    edge_duration = {
        (row.origin_site_id, row.destination_site_id): float(row.duration_hours_p50)
        for row in edges[edges["high_confidence"]].itertuples(index=False)
    }
    reliable_periods = periods[periods["period_estimate_available"]]
    period_duration = {
        (row.origin_site_id, row.destination_site_id, row.departure_period): float(
            row.duration_hours_p50
        )
        for row in reliable_periods.itertuples(index=False)
    }
    edge_types = tasks.groupby(
        ["origin_site_id", "destination_site_id"]
    )["vehicle_type_name"].agg(lambda values: set(values.dropna()))

    lane_paths: dict[tuple[str, str], list[list[str]]] = {}
    unique_lanes = tasks.loc[
        tasks["network_covered"], ["origin_site_id", "destination_site_id"]
    ].drop_duplicates()
    for lane in unique_lanes.itertuples(index=False):
        paths = k_paths(
            graph,
            lane.origin_site_id,
            lane.destination_site_id,
            "distance_km",
            paths_per_objective,
        )
        paths += k_paths(
            graph,
            lane.origin_site_id,
            lane.destination_site_id,
            "duration_hours",
            paths_per_objective,
        )
        deduplicated: dict[tuple[str, ...], list[str]] = {}
        for path in paths:
            deduplicated.setdefault(tuple(path), path)
        lane_paths[(lane.origin_site_id, lane.destination_site_id)] = list(
            deduplicated.values()
        )

    rows: list[dict[str, object]] = []
    for task in tasks[tasks["network_covered"]].itertuples(index=False):
        departed_at = pd.Timestamp(task.departed_at)
        deadline = pd.Timestamp(task.arrived_at)
        rows.append(
            {
                "task_id": task.task_id,
                "path_option_id": f"{task.task_id}_historical",
                "path_source": "historical_observed",
                "path": f"{task.origin_site_id}>{task.destination_site_id}",
                "path_edge_count": 1,
                "distance_km": float(task.distance_km),
                "duration_hours": float(task.duration_hours),
                "arrival_at": deadline,
                "compatible_vehicle_types": str(task.vehicle_type_name),
                "compatible_vehicle_type_count": 1,
            }
        )
        for option_number, path in enumerate(
            lane_paths[(task.origin_site_id, task.destination_site_id)], start=1
        ):
            edges_in_path = list(zip(path, path[1:]))
            compatible = set(edge_types[edges_in_path[0]])
            for edge in edges_in_path[1:]:
                compatible &= set(edge_types[edge])
            if not compatible:
                continue
            duration, _, _ = evaluate_time_dependent_path(
                path, departed_at, edge_duration, period_duration
            )
            arrival_at = departed_at + pd.to_timedelta(duration, unit="h")
            if arrival_at > deadline:
                continue
            distance = sum(float(graph.edges[edge]["distance_km"]) for edge in edges_in_path)
            rows.append(
                {
                    "task_id": task.task_id,
                    "path_option_id": f"{task.task_id}_network_{option_number:02d}",
                    "path_source": "high_confidence_network",
                    "path": ">".join(path),
                    "path_edge_count": len(edges_in_path),
                    "distance_km": distance,
                    "duration_hours": duration,
                    "arrival_at": arrival_at,
                    "compatible_vehicle_types": "|".join(sorted(compatible)),
                    "compatible_vehicle_type_count": len(compatible),
                }
            )
    return pd.DataFrame(rows)


def summarize_options(
    options: pd.DataFrame, tasks: pd.DataFrame, instances: pd.DataFrame
) -> dict[str, object]:
    topology_counts = options.groupby("task_id")["path"].nunique()
    alternative_tasks = set(topology_counts[topology_counts > 1].index)
    instances = instances.copy()
    instances["service_date"] = instances["service_date"].astype("string")
    tasks = tasks.copy()
    tasks["service_date"] = tasks["service_date"].astype("string")
    qualified = tasks.merge(
        instances.loc[
            instances["qualifies_for_vrp"],
            ["instance_id", "service_date", "component_id"],
        ],
        on=["service_date", "component_id"],
        how="inner",
    )
    qualified["has_alternative_path"] = qualified["task_id"].isin(alternative_tasks)
    instance_alternatives = qualified.groupby("instance_id")[
        "has_alternative_path"
    ].sum()
    return {
        "source_only": "订单数据.xlsx",
        "network_covered_tasks": int(tasks["network_covered"].sum()),
        "tasks_with_feasible_alternative_paths": int(len(alternative_tasks)),
        "alternative_path_task_share": float(
            len(alternative_tasks) / tasks["network_covered"].sum()
        ),
        "path_options": int(len(options)),
        "qualified_instances": int(qualified["instance_id"].nunique()),
        "qualified_instances_with_alternative_paths": int(
            instance_alternatives.gt(0).sum()
        ),
        "qualified_instances_with_at_least_10_alternative_tasks": int(
            instance_alternatives.ge(10).sum()
        ),
        "historical_option_coverage_rate": float(
            options[options["path_source"] == "historical_observed"]["task_id"].nunique()
            / tasks["network_covered"].sum()
        ),
    }


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    edges = pd.read_csv(args.data_dir / "network_edges.csv")
    periods = pd.read_csv(args.data_dir / "edge_period_stats.csv")
    instances = pd.read_csv(args.data_dir / "instances.csv")
    options = build_path_options(
        tasks, edges, periods, args.paths_per_objective
    )
    summary = summarize_options(options, tasks, instances)
    options.to_csv(
        args.data_dir / "task_path_options.csv", index=False, encoding="utf-8-sig"
    )
    (args.result_dir / "task_path_option_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
