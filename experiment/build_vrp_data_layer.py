#!/usr/bin/env python3
"""Build the task, network and instance data layer for the Excel-only VRP."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "networkx>=3.4",
#   "openpyxl>=3.1",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from prepare_excel_transport_data import build_transport_events, load_data

PERIOD_BINS = [-1, 5, 9, 15, 19, 23]
PERIOD_LABELS = ["night", "morning_peak", "daytime", "evening_peak", "evening"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造订单 Excel 的 VRP 数据层。")
    parser.add_argument("--input", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--min-edge-observations", type=int, default=10)
    parser.add_argument("--min-period-observations", type=int, default=5)
    parser.add_argument("--min-instance-tasks", type=int, default=10)
    parser.add_argument("--required-instances", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("processed/company/vrp"))
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    return parser.parse_args()


def build_tasks(events: pd.DataFrame) -> pd.DataFrame:
    tasks = events[events["analysis_eligible"]].copy()
    tasks = tasks[
        [
            "event_id",
            "origin_site_id",
            "destination_site_id",
            "service_date",
            "departed_at",
            "arrived_at",
            "distance_km",
            "duration_hours",
            "vehicle_type_name",
            "vehicle",
            "order_count",
        ]
    ].rename(columns={"event_id": "task_id", "vehicle": "historical_vehicle"})
    tasks["departed_at"] = pd.to_datetime(tasks["departed_at"])
    tasks["arrived_at"] = pd.to_datetime(tasks["arrived_at"])
    tasks["departure_period"] = pd.cut(
        tasks["departed_at"].dt.hour,
        bins=PERIOD_BINS,
        labels=PERIOD_LABELS,
    ).astype("string")
    return tasks.sort_values(["service_date", "departed_at", "task_id"]).reset_index(
        drop=True
    )


def build_network_tables(
    tasks: pd.DataFrame, min_edge_observations: int, min_period_observations: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_keys = ["origin_site_id", "destination_site_id"]
    edges = tasks.groupby(edge_keys, as_index=False).agg(
        observations=("task_id", "size"),
        active_days=("service_date", "nunique"),
        distance_km_p50=("distance_km", "median"),
        duration_hours_p50=("duration_hours", "median"),
        duration_hours_p90=("duration_hours", lambda values: values.quantile(0.9)),
        vehicle_type_count=("vehicle_type_name", "nunique"),
    )
    edges["high_confidence"] = edges["observations"].ge(min_edge_observations)

    periods = tasks.groupby(edge_keys + ["departure_period"], as_index=False).agg(
        observations=("task_id", "size"),
        duration_hours_p50=("duration_hours", "median"),
        duration_hours_p90=("duration_hours", lambda values: values.quantile(0.9)),
    )
    periods["period_estimate_available"] = periods["observations"].ge(
        min_period_observations
    )
    return edges, periods


def build_high_confidence_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    selected = edges[edges["high_confidence"]]
    for row in selected.itertuples(index=False):
        graph.add_edge(
            row.origin_site_id,
            row.destination_site_id,
            distance_km=float(row.distance_km_p50),
            duration_hours=float(row.duration_hours_p50),
        )
    return graph


def assign_components(tasks: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    component_map: dict[str, str] = {}
    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=lambda nodes: min(nodes))
    for index, nodes in enumerate(components, start=1):
        component_id = f"component_{index:04d}"
        component_map.update({node: component_id for node in nodes})

    assigned = tasks.copy()
    assigned["origin_component"] = assigned["origin_site_id"].map(component_map)
    assigned["destination_component"] = assigned["destination_site_id"].map(component_map)
    assigned["network_covered"] = (
        assigned["origin_component"].notna()
        & assigned["origin_component"].eq(assigned["destination_component"])
        & assigned.apply(
            lambda row: graph.has_edge(
                row["origin_site_id"], row["destination_site_id"]
            ),
            axis=1,
        )
    )
    assigned["component_id"] = assigned["origin_component"].where(
        assigned["network_covered"]
    )
    return assigned


def shortest_duration_lookup(graph: nx.DiGraph, origins: set[str]) -> dict[str, dict[str, float]]:
    return {
        origin: nx.single_source_dijkstra_path_length(
            graph, origin, weight="duration_hours"
        )
        for origin in origins
        if origin in graph
    }


def build_instances(
    tasks: pd.DataFrame,
    graph: nx.DiGraph,
    min_instance_tasks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    covered = tasks[tasks["network_covered"]].copy()
    durations = shortest_duration_lookup(
        graph, set(covered["destination_site_id"].dropna())
    )
    instance_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []

    grouped = covered.groupby(["service_date", "component_id"], sort=True)
    for index, ((service_date, component_id), group) in enumerate(grouped, start=1):
        instance_id = f"instance_{index:05d}"
        records = list(group.sort_values("departed_at").itertuples(index=False))
        linkable_tasks: set[str] = set()
        link_count_before = len(link_rows)
        for predecessor in records:
            destination_lengths = durations.get(predecessor.destination_site_id, {})
            for successor in records:
                if predecessor.task_id == successor.task_id:
                    continue
                deadhead_hours = destination_lengths.get(successor.origin_site_id)
                if deadhead_hours is None:
                    continue
                ready_at = predecessor.arrived_at + pd.to_timedelta(
                    deadhead_hours, unit="h"
                )
                if ready_at <= successor.departed_at:
                    linkable_tasks.update([predecessor.task_id, successor.task_id])
                    link_rows.append(
                        {
                            "instance_id": instance_id,
                            "from_task_id": predecessor.task_id,
                            "to_task_id": successor.task_id,
                            "deadhead_duration_hours_p50": deadhead_hours,
                            "available_slack_hours": (
                                successor.departed_at - ready_at
                            ).total_seconds()
                            / 3600,
                        }
                    )
        instance_rows.append(
            {
                "instance_id": instance_id,
                "service_date": service_date,
                "component_id": component_id,
                "task_count": len(records),
                "linkable_task_count": len(linkable_tasks),
                "candidate_link_count": len(link_rows) - link_count_before,
                "origin_site_count": group["origin_site_id"].nunique(),
                "destination_site_count": group["destination_site_id"].nunique(),
                "historical_vehicle_count": group["historical_vehicle"].nunique(),
                "qualifies_for_vrp": len(linkable_tasks) >= min_instance_tasks,
            }
        )

    return pd.DataFrame(instance_rows), pd.DataFrame(link_rows)


def build_readiness_report(
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    periods: pd.DataFrame,
    instances: pd.DataFrame,
    links: pd.DataFrame,
    min_edge_observations: int,
    min_period_observations: int,
    min_instance_tasks: int,
    required_instances: int,
) -> dict[str, object]:
    qualified = instances[instances["qualifies_for_vrp"]]
    required_columns = [
        "origin_site_id",
        "destination_site_id",
        "departed_at",
        "arrived_at",
        "distance_km",
        "vehicle_type_name",
    ]
    complete_share = float(tasks[required_columns].notna().all(axis=1).mean())
    report = {
        "source_only": "订单数据.xlsx",
        "thresholds": {
            "min_edge_observations": min_edge_observations,
            "min_period_observations": min_period_observations,
            "min_linkable_tasks_per_instance": min_instance_tasks,
            "required_independent_instances": required_instances,
        },
        "tasks": {
            "eligible_tasks": int(len(tasks)),
            "required_field_complete_share": complete_share,
            "network_covered_tasks": int(tasks["network_covered"].sum()),
            "network_coverage_share": float(tasks["network_covered"].mean()),
        },
        "network": {
            "directed_edges": int(len(edges)),
            "high_confidence_edges": int(edges["high_confidence"].sum()),
            "period_cells": int(len(periods)),
            "reliable_period_cells": int(periods["period_estimate_available"].sum()),
        },
        "instances": {
            "date_component_instances": int(len(instances)),
            "qualified_instances": int(len(qualified)),
            "qualified_tasks": int(qualified["task_count"].sum()),
            "qualified_linkable_tasks": int(qualified["linkable_task_count"].sum()),
            "candidate_task_links": int(len(links)),
            "largest_instance_tasks": int(instances["task_count"].max())
            if not instances.empty
            else 0,
            "largest_instance_linkable_tasks": int(
                instances["linkable_task_count"].max()
            )
            if not instances.empty
            else 0,
        },
    }
    checks = {
        "task_fields_complete": complete_share == 1.0,
        "enough_independent_instances": len(qualified) >= required_instances,
        "every_qualified_instance_meets_task_threshold": bool(
            not qualified.empty
            and qualified["linkable_task_count"].ge(min_instance_tasks).all()
        ),
    }
    report["gate_checks"] = checks
    report["ready_for_basic_vrp_comparison"] = all(checks.values())
    return report


def main() -> None:
    args = parse_args()
    orders, waybills = load_data(args.input)
    events, _ = build_transport_events(orders, waybills)
    tasks = build_tasks(events)
    edges, periods = build_network_tables(
        tasks, args.min_edge_observations, args.min_period_observations
    )
    graph = build_high_confidence_graph(edges)
    tasks = assign_components(tasks, graph)
    instances, links = build_instances(tasks, graph, args.min_instance_tasks)
    report = build_readiness_report(
        tasks,
        edges,
        periods,
        instances,
        links,
        args.min_edge_observations,
        args.min_period_observations,
        args.min_instance_tasks,
        args.required_instances,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(args.output_dir / "tasks.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(args.output_dir / "network_edges.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(
        args.output_dir / "edge_period_stats.csv", index=False, encoding="utf-8-sig"
    )
    instances.to_csv(args.output_dir / "instances.csv", index=False, encoding="utf-8-sig")
    links.to_csv(
        args.output_dir / "candidate_task_links.csv", index=False, encoding="utf-8-sig"
    )
    public_instances = instances.drop(columns=["service_date"], errors="ignore")
    public_instances.to_csv(
        args.result_dir / "vrp_instance_summary.csv", index=False, encoding="utf-8-sig"
    )
    (args.result_dir / "vrp_data_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
