#!/usr/bin/env python3
"""Compare independent task routing with a minimum-vehicle task VRP."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比独立任务路径与基础 VRP。")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("processed/company/vrp")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    return parser.parse_args()


def minimum_vehicle_path_cover(
    task_ids: list[str], links: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matching_graph = nx.Graph()
    left_nodes = [("left", task_id) for task_id in task_ids]
    right_nodes = [("right", task_id) for task_id in task_ids]
    matching_graph.add_nodes_from(left_nodes, bipartite=0)
    matching_graph.add_nodes_from(right_nodes, bipartite=1)
    link_index: dict[tuple[str, str], pd.Series] = {}
    for _, link in links.iterrows():
        from_task = str(link["from_task_id"])
        to_task = str(link["to_task_id"])
        link_index[(from_task, to_task)] = link
        matching_graph.add_edge(
            ("left", from_task),
            ("right", to_task),
            weight=-float(link["deadhead_distance_km"]),
        )

    matching = nx.algorithms.matching.max_weight_matching(
        matching_graph, maxcardinality=True, weight="weight"
    )
    selected_pairs: list[tuple[str, str]] = []
    for first, second in matching:
        left, right = (first, second) if first[0] == "left" else (second, first)
        selected_pairs.append((left[1], right[1]))

    successor = {from_task: to_task for from_task, to_task in selected_pairs}
    predecessor = {to_task: from_task for from_task, to_task in selected_pairs}
    starts = sorted(set(task_ids) - set(predecessor))
    assignment_rows: list[dict[str, object]] = []
    visited: set[str] = set()
    for vehicle_number, start in enumerate(starts, start=1):
        task_id = start
        sequence = 1
        while task_id not in visited:
            visited.add(task_id)
            assignment_rows.append(
                {
                    "vehicle_id": f"vehicle_{vehicle_number:04d}",
                    "sequence": sequence,
                    "task_id": task_id,
                }
            )
            if task_id not in successor:
                break
            task_id = successor[task_id]
            sequence += 1

    if visited != set(task_ids):
        raise ValueError("Task path cover contains a cycle or omits tasks")

    selected_rows = [link_index[pair].to_dict() for pair in selected_pairs]
    selected_links = pd.DataFrame(selected_rows, columns=links.columns)
    assignments = pd.DataFrame(assignment_rows)
    return assignments, selected_links


def compare_instances(
    tasks: pd.DataFrame, instances: pd.DataFrame, links: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tasks = tasks.copy()
    tasks["service_date"] = tasks["service_date"].astype("string")
    instances = instances.copy()
    instances["service_date"] = instances["service_date"].astype("string")
    task_instances = tasks.merge(
        instances[["instance_id", "service_date", "component_id", "qualifies_for_vrp"]],
        on=["service_date", "component_id"],
        how="left",
        validate="many_to_one",
    )
    qualified_tasks = task_instances[task_instances["qualifies_for_vrp"] == True]  # noqa: E712
    metric_rows: list[dict[str, object]] = []
    schedule_parts: list[pd.DataFrame] = []

    for instance_id, group in qualified_tasks.groupby("instance_id", sort=True):
        instance_links = links[links["instance_id"] == instance_id]
        task_ids = group["task_id"].astype(str).tolist()
        assignments, selected_links = minimum_vehicle_path_cover(task_ids, instance_links)
        assignments["instance_id"] = instance_id
        assignments = assignments.merge(
            group[
                [
                    "task_id",
                    "origin_site_id",
                    "destination_site_id",
                    "departed_at",
                    "arrived_at",
                    "distance_km",
                ]
            ],
            on="task_id",
            how="left",
            validate="one_to_one",
        )
        selected_lookup = selected_links.set_index("from_task_id") if not selected_links.empty else None
        assignments["deadhead_to_next_distance_km"] = assignments["task_id"].map(
            selected_lookup["deadhead_distance_km"] if selected_lookup is not None else {}
        ).fillna(0.0)
        assignments["deadhead_to_next_path"] = assignments["task_id"].map(
            selected_lookup["deadhead_path"] if selected_lookup is not None else {}
        )
        schedule_parts.append(assignments)

        task_count = len(group)
        vehicle_count = assignments["vehicle_id"].nunique()
        loaded_distance = float(group["distance_km"].sum())
        deadhead_distance = float(selected_links["deadhead_distance_km"].sum())
        task_times = group.set_index("task_id")
        time_overlap_violations = 0
        for link in selected_links.itertuples(index=False):
            ready_at = pd.to_datetime(task_times.loc[link.from_task_id, "arrived_at"])
            ready_at += pd.to_timedelta(link.deadhead_duration_hours_p50, unit="h")
            next_departure = pd.to_datetime(
                task_times.loc[link.to_task_id, "departed_at"]
            )
            time_overlap_violations += int(ready_at > next_departure)
        duplicate_task_assignments = int(assignments["task_id"].duplicated().sum())
        missing_task_assignments = int(
            len(set(task_ids) - set(assignments["task_id"].astype(str)))
        )
        metric_rows.append(
            {
                "instance_id": instance_id,
                "task_count": task_count,
                "independent_vehicle_count": task_count,
                "vrp_vehicle_count": vehicle_count,
                "vehicle_reduction": task_count - vehicle_count,
                "vehicle_reduction_rate": (task_count - vehicle_count) / task_count,
                "loaded_distance_km": loaded_distance,
                "vrp_internal_deadhead_distance_km": deadhead_distance,
                "vrp_total_observed_scope_distance_km": loaded_distance
                + deadhead_distance,
                "average_tasks_per_vrp_vehicle": task_count / vehicle_count,
                "task_service_rate": (task_count - missing_task_assignments)
                / task_count,
                "time_overlap_violations": time_overlap_violations,
                "duplicate_task_assignments": duplicate_task_assignments,
                "missing_task_assignments": missing_task_assignments,
                "missing_selected_link_paths": int(
                    selected_links["deadhead_path"].isna().sum()
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    schedules = pd.concat(schedule_parts, ignore_index=True) if schedule_parts else pd.DataFrame()
    total_tasks = int(metrics["task_count"].sum())
    independent_vehicles = int(metrics["independent_vehicle_count"].sum())
    vrp_vehicles = int(metrics["vrp_vehicle_count"].sum())
    summary = {
        "comparison_scope": "Qualified date-component instances from 订单数据.xlsx",
        "interpretation": "Independent routing assumes one vehicle appears at each task origin. Basic VRP explicitly connects consecutive tasks and counts internal deadhead only.",
        "instances": int(len(metrics)),
        "tasks": total_tasks,
        "independent_vehicle_count": independent_vehicles,
        "vrp_vehicle_count": vrp_vehicles,
        "vehicle_reduction": independent_vehicles - vrp_vehicles,
        "vehicle_reduction_rate": (independent_vehicles - vrp_vehicles)
        / independent_vehicles,
        "loaded_distance_km": float(metrics["loaded_distance_km"].sum()),
        "vrp_internal_deadhead_distance_km": float(
            metrics["vrp_internal_deadhead_distance_km"].sum()
        ),
        "task_service_rate": float(
            (metrics["task_service_rate"] * metrics["task_count"]).sum() / total_tasks
        ),
        "time_overlap_violations": int(metrics["time_overlap_violations"].sum()),
        "duplicate_task_assignments": int(
            metrics["duplicate_task_assignments"].sum()
        ),
        "missing_task_assignments": int(metrics["missing_task_assignments"].sum()),
        "missing_selected_link_paths": int(
            metrics["missing_selected_link_paths"].sum()
        ),
        "result_scope_note": "Vehicle counts measure consolidation within the observed task sample and do not estimate the company's complete fleet.",
    }
    return metrics, schedules, summary


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "candidate_task_links.csv")
    metrics, schedules, summary = compare_instances(tasks, instances, links)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        args.result_dir / "basic_vrp_instance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    schedules.to_csv(
        args.data_dir / "basic_vrp_schedules.csv", index=False, encoding="utf-8-sig"
    )
    (args.result_dir / "basic_vrp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
