#!/usr/bin/env python3
"""Jointly choose empirical vehicle types and task chains."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "scipy>=1.14",
# ]
# ///

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="联合优化经验兼容车型和任务链。")
    parser.add_argument("--data-dir", type=Path, default=Path("processed/company/vrp"))
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    return parser.parse_args()


def solve_type_compatible_path_cover(
    task_ids: list[str],
    compatible_types: dict[str, set[str]],
    links: pd.DataFrame,
    time_limit: float = 60.0,
    vehicle_cost_equivalent_km: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    y_keys = [
        (task_id, vehicle_type)
        for task_id in task_ids
        for vehicle_type in sorted(compatible_types[task_id])
    ]
    x_keys: list[tuple[str, str, str]] = []
    link_lookup: dict[tuple[str, str], pd.Series] = {}
    for _, link in links.iterrows():
        start = str(link["from_task_id"])
        end = str(link["to_task_id"])
        link_lookup[(start, end)] = link
        common_types = compatible_types[start] & compatible_types[end]
        x_keys.extend(
            (start, end, vehicle_type) for vehicle_type in sorted(common_types)
        )

    variable_keys = [("y", *key) for key in y_keys] + [("x", *key) for key in x_keys]
    variable_index = {key: index for index, key in enumerate(variable_keys)}
    total_link_distance = float(links["deadhead_distance_km"].sum())
    cardinality_priority = (
        total_link_distance + 1.0
        if vehicle_cost_equivalent_km is None
        else vehicle_cost_equivalent_km
    )
    objective = np.zeros(len(variable_keys))
    for start, end, vehicle_type in x_keys:
        distance = float(link_lookup[(start, end)]["deadhead_distance_km"])
        objective[variable_index[("x", start, end, vehicle_type)]] = (
            -cardinality_priority + distance
        )

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(values: dict[int, float], lower: float, upper: float) -> None:
        row = len(lower_bounds)
        for column, value in values.items():
            row_indices.append(row)
            column_indices.append(column)
            coefficients.append(value)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    for task_id in task_ids:
        add_constraint(
            {
                variable_index[("y", task_id, vehicle_type)]: 1.0
                for vehicle_type in compatible_types[task_id]
            },
            1.0,
            1.0,
        )

    outgoing: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    incoming: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in x_keys:
        start, end, vehicle_type = key
        outgoing[(start, vehicle_type)].append(key)
        incoming[(end, vehicle_type)].append(key)

    for task_id, vehicle_type in y_keys:
        y_index = variable_index[("y", task_id, vehicle_type)]
        outgoing_values = {y_index: -1.0}
        outgoing_values.update(
            {
                variable_index[("x", *key)]: 1.0
                for key in outgoing[(task_id, vehicle_type)]
            }
        )
        add_constraint(outgoing_values, -np.inf, 0.0)
        incoming_values = {y_index: -1.0}
        incoming_values.update(
            {
                variable_index[("x", *key)]: 1.0
                for key in incoming[(task_id, vehicle_type)]
            }
        )
        add_constraint(incoming_values, -np.inf, 0.0)

    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(lower_bounds), len(variable_keys)),
    ).tocsr()
    result = milp(
        c=objective,
        integrality=np.ones(len(variable_keys)),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower_bounds, upper_bounds),
        options={"time_limit": time_limit},
    )
    if result.x is None:
        raise RuntimeError(f"Type-compatible VRP has no solution: {result.message}")

    task_type = {
        task_id: vehicle_type
        for task_id, vehicle_type in y_keys
        if result.x[variable_index[("y", task_id, vehicle_type)]] > 0.5
    }
    selected_keys = [
        key for key in x_keys if result.x[variable_index[("x", *key)]] > 0.5
    ]
    successor = {start: end for start, end, _ in selected_keys}
    predecessor = {end: start for start, end, _ in selected_keys}
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
                    "vehicle_type_name": task_type[task_id],
                    "sequence": sequence,
                    "task_id": task_id,
                }
            )
            if task_id not in successor:
                break
            task_id = successor[task_id]
            sequence += 1
    if visited != set(task_ids):
        raise ValueError("Type-compatible task cover contains a cycle or omits tasks")

    selected_rows = []
    for start, end, vehicle_type in selected_keys:
        row = link_lookup[(start, end)].to_dict()
        row["vehicle_type_name"] = vehicle_type
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    diagnostics = {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": result.message,
        "optimality_gap": float(result.mip_gap) if result.mip_gap is not None else None,
        "objective_mode": (
            "minimum_vehicles_then_deadhead"
            if vehicle_cost_equivalent_km is None
            else "vehicle_deadhead_weighted_cost"
        ),
        "vehicle_cost_equivalent_km": vehicle_cost_equivalent_km,
    }
    return pd.DataFrame(assignment_rows), selected, diagnostics


def attach_route_details(
    assignments: pd.DataFrame,
    selected: pd.DataFrame,
    tasks: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    route = assignments.merge(
        tasks[
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
    route["loaded_path"] = route["origin_site_id"] + ">" + route["destination_site_id"]
    detail_columns = {
        "to_task_id": "next_task_id",
        "deadhead_path": "deadhead_to_next_path",
        "deadhead_distance_km": "deadhead_to_next_distance_km",
        "deadhead_duration_hours_p50": "deadhead_to_next_duration_hours",
    }
    if selected.empty:
        for output_column in detail_columns.values():
            route[output_column] = pd.NA
    else:
        selected_lookup = selected.set_index("from_task_id")
        for input_column, output_column in detail_columns.items():
            route[output_column] = route["task_id"].map(selected_lookup[input_column])
    route["deadhead_to_next_distance_km"] = pd.to_numeric(
        route["deadhead_to_next_distance_km"], errors="coerce"
    ).fillna(0.0)
    route["deadhead_to_next_duration_hours"] = pd.to_numeric(
        route["deadhead_to_next_duration_hours"], errors="coerce"
    ).fillna(0.0)

    task_lookup = tasks.set_index("task_id")
    time_violations = 0
    endpoint_violations = 0
    missing_paths = 0
    for link in selected.itertuples(index=False):
        ready_at = pd.to_datetime(task_lookup.loc[link.from_task_id, "arrived_at"])
        ready_at += pd.to_timedelta(link.deadhead_duration_hours_p50, unit="h")
        next_departure = pd.to_datetime(task_lookup.loc[link.to_task_id, "departed_at"])
        time_violations += int(ready_at > next_departure)
        if pd.isna(link.deadhead_path):
            missing_paths += 1
            continue
        path_nodes = str(link.deadhead_path).split(">")
        expected_origin = task_lookup.loc[link.from_task_id, "destination_site_id"]
        expected_destination = task_lookup.loc[link.to_task_id, "origin_site_id"]
        endpoint_violations += int(
            path_nodes[0] != expected_origin or path_nodes[-1] != expected_destination
        )
    validation = {
        "time_overlap_violations": time_violations,
        "deadhead_endpoint_violations": endpoint_violations,
        "missing_selected_link_paths": missing_paths,
    }
    return route, validation


def optimize_instances(
    tasks: pd.DataFrame,
    instances: pd.DataFrame,
    links: pd.DataFrame,
    basic_metrics: pd.DataFrame,
    time_limit: float,
    baseline_vehicle_column: str = "vrp_vehicle_count",
    lane_type_compatibility: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tasks = tasks.copy()
    tasks["service_date"] = tasks["service_date"].astype("string")
    instances = instances.copy()
    instances["service_date"] = instances["service_date"].astype("string")
    lane_types = lane_type_compatibility
    if lane_types is None:
        lane_types = tasks.groupby(["origin_site_id", "destination_site_id"])[
            "vehicle_type_name"
        ].agg(lambda values: set(values.dropna()))
    task_instances = tasks.merge(
        instances[["instance_id", "service_date", "component_id", "qualifies_for_vrp"]],
        on=["service_date", "component_id"],
        how="left",
        validate="many_to_one",
    )
    qualified = task_instances[task_instances["qualifies_for_vrp"] == True]
    task_types: dict[str, set[str]] = {}
    for row in qualified.itertuples(index=False):
        lane = (row.origin_site_id, row.destination_site_id)
        if lane not in lane_types.index:
            raise ValueError(
                f"Qualified task {row.task_id} has no vehicle-type evidence for lane {lane}"
            )
        task_types[row.task_id] = lane_types[lane]
    basic_lookup = basic_metrics.set_index("instance_id")
    metric_rows: list[dict[str, object]] = []
    schedule_parts: list[pd.DataFrame] = []

    for instance_id, group in qualified.groupby("instance_id", sort=True):
        instance_links = links[links["instance_id"] == instance_id]
        task_ids = group["task_id"].astype(str).tolist()
        assignments, selected, diagnostics = solve_type_compatible_path_cover(
            task_ids, task_types, instance_links, time_limit
        )
        assignments, validation = attach_route_details(assignments, selected, group)
        assignments["instance_id"] = instance_id
        schedule_parts.append(assignments)
        vehicle_count = assignments["vehicle_id"].nunique()
        metric_rows.append(
            {
                "instance_id": instance_id,
                "task_count": len(group),
                "basic_vrp_vehicle_count": int(
                    basic_lookup.loc[instance_id, baseline_vehicle_column]
                ),
                "type_compatible_vehicle_count": vehicle_count,
                "vehicles_added_by_type_compatibility": vehicle_count
                - int(basic_lookup.loc[instance_id, baseline_vehicle_column]),
                "selected_task_links": len(selected),
                "internal_deadhead_distance_km": float(
                    selected["deadhead_distance_km"].sum()
                )
                if not selected.empty
                else 0.0,
                "task_service_rate": len(assignments) / len(group),
                "duplicate_task_assignments": int(
                    assignments["task_id"].duplicated().sum()
                ),
                "route_type_violations": int(
                    (
                        assignments.groupby("vehicle_id")["vehicle_type_name"].nunique()
                        > 1
                    ).sum()
                ),
                **validation,
                "solver_success": diagnostics["solver_success"],
                "optimality_gap": diagnostics["optimality_gap"],
            }
        )

    metrics = pd.DataFrame(metric_rows)
    schedules = pd.concat(schedule_parts, ignore_index=True)
    summary = {
        "source_only": "订单数据.xlsx",
        "vehicle_type_interpretation": "A vehicle type is eligible for a task only when that type appears historically on the same directed lane.",
        "instances": len(metrics),
        "tasks": int(metrics["task_count"].sum()),
        "basic_vrp_vehicle_count": int(metrics["basic_vrp_vehicle_count"].sum()),
        "type_compatible_vehicle_count": int(
            metrics["type_compatible_vehicle_count"].sum()
        ),
        "vehicles_added_by_type_compatibility": int(
            metrics["vehicles_added_by_type_compatibility"].sum()
        ),
        "internal_deadhead_distance_km": float(
            metrics["internal_deadhead_distance_km"].sum()
        ),
        "task_service_rate": float(
            (metrics["task_service_rate"] * metrics["task_count"]).sum()
            / metrics["task_count"].sum()
        ),
        "duplicate_task_assignments": int(metrics["duplicate_task_assignments"].sum()),
        "route_type_violations": int(metrics["route_type_violations"].sum()),
        "time_overlap_violations": int(metrics["time_overlap_violations"].sum()),
        "deadhead_endpoint_violations": int(
            metrics["deadhead_endpoint_violations"].sum()
        ),
        "missing_selected_link_paths": int(
            metrics["missing_selected_link_paths"].sum()
        ),
        "all_instances_solved": bool(metrics["solver_success"].all()),
    }
    return metrics, schedules, summary


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "candidate_task_links.csv")
    basic_metrics = pd.read_csv(args.result_dir / "basic_vrp_instance_comparison.csv")
    lane_type_path = args.data_dir / "lane_vehicle_types.csv"
    lane_type_compatibility = None
    if lane_type_path.exists():
        lane_type_rows = pd.read_csv(lane_type_path)
        lane_type_compatibility = lane_type_rows.groupby(
            ["origin_site_id", "destination_site_id"]
        )["vehicle_type_name"].agg(lambda values: set(values.dropna()))
    metrics, schedules, summary = optimize_instances(
        tasks,
        instances,
        links,
        basic_metrics,
        args.time_limit,
        lane_type_compatibility=lane_type_compatibility,
    )
    metrics.to_csv(
        args.result_dir / "type_compatible_vrp_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    schedules.to_csv(
        args.data_dir / "type_compatible_vrp_schedules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "type_compatible_vrp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
