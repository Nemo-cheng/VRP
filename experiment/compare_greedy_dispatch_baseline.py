#!/usr/bin/env python3
"""Compare the final P90 VRP with a chronological greedy dispatcher."""

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
from pathlib import Path

import pandas as pd
from optimize_type_compatible_vrp import attach_route_details
from optimize_vehicle_deadhead_tradeoff import prepare_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较顺序贪心调度与最终 P90 VRP。")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("processed/company/vrp_final_test")
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    return parser.parse_args()


def preferred_task_types(
    tasks: pd.DataFrame, lane_vehicle_types: pd.DataFrame
) -> dict[str, str]:
    ranked = lane_vehicle_types.sort_values(
        ["origin_site_id", "destination_site_id", "observations", "vehicle_type_name"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["origin_site_id", "destination_site_id"])
    preference = ranked.set_index(["origin_site_id", "destination_site_id"])[
        "vehicle_type_name"
    ]
    result: dict[str, str] = {}
    for row in tasks.itertuples(index=False):
        result[str(row.task_id)] = str(
            preference.loc[(row.origin_site_id, row.destination_site_id)]
        )
    return result


def greedy_dispatch_instance(
    tasks: pd.DataFrame,
    compatible_types: dict[str, set[str]],
    preferred_types: dict[str, str],
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    link_lookup = {
        (str(row.from_task_id), str(row.to_task_id)): row
        for row in links.itertuples(index=False)
    }
    vehicles: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    ordered = tasks.sort_values(["departed_at", "task_id"])

    for task in ordered.itertuples(index=False):
        task_id = str(task.task_id)
        candidates: list[tuple[float, str, int, object]] = []
        for index, vehicle in enumerate(vehicles):
            if vehicle["vehicle_type_name"] not in compatible_types[task_id]:
                continue
            link = link_lookup.get((str(vehicle["last_task_id"]), task_id))
            if link is None:
                continue
            candidates.append(
                (
                    float(link.deadhead_distance_km),
                    str(vehicle["vehicle_id"]),
                    index,
                    link,
                )
            )

        if candidates:
            _, _, vehicle_index, selected_link = min(candidates)
            vehicle = vehicles[vehicle_index]
            vehicle["last_task_id"] = task_id
            vehicle["sequence"] = int(vehicle["sequence"]) + 1
            selected_rows.append(selected_link._asdict())
        else:
            vehicle_index = len(vehicles)
            selected_type = preferred_types[task_id]
            if selected_type not in compatible_types[task_id]:
                selected_type = min(compatible_types[task_id])
            vehicle = {
                "vehicle_id": f"greedy_vehicle_{vehicle_index + 1:04d}",
                "vehicle_type_name": selected_type,
                "last_task_id": task_id,
                "sequence": 1,
            }
            vehicles.append(vehicle)

        assignment_rows.append(
            {
                "vehicle_id": vehicle["vehicle_id"],
                "vehicle_type_name": vehicle["vehicle_type_name"],
                "sequence": vehicle["sequence"],
                "task_id": task_id,
            }
        )

    return pd.DataFrame(assignment_rows), pd.DataFrame(
        selected_rows, columns=links.columns
    )


def evaluate_greedy_dispatch(
    qualified_tasks: pd.DataFrame,
    compatible_types: dict[str, set[str]],
    preferred_types: dict[str, str],
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    schedule_parts: list[pd.DataFrame] = []
    for instance_id, group in qualified_tasks.groupby("instance_id", sort=True):
        instance_links = links[links["instance_id"] == instance_id]
        assignments, selected = greedy_dispatch_instance(
            group, compatible_types, preferred_types, instance_links
        )
        route, validation = attach_route_details(assignments, selected, group)
        route["instance_id"] = instance_id
        schedule_parts.append(route)
        metric_rows.append(
            {
                "instance_id": instance_id,
                "task_count": len(group),
                "greedy_vehicle_count": int(route["vehicle_id"].nunique()),
                "greedy_internal_deadhead_distance_km": float(
                    selected["deadhead_distance_km"].sum()
                )
                if not selected.empty
                else 0.0,
                "task_service_rate": len(route) / len(group),
                "duplicate_task_assignments": int(route["task_id"].duplicated().sum()),
                "route_type_violations": int(
                    (
                        route.groupby("vehicle_id")["vehicle_type_name"].nunique() > 1
                    ).sum()
                ),
                **validation,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    tasks = int(metrics["task_count"].sum())
    summary = {
        "source_only": "订单数据.xlsx",
        "method": "Chronological nearest-feasible-vehicle greedy dispatch",
        "travel_time_stat": "p90",
        "instances": len(metrics),
        "tasks": tasks,
        "greedy_vehicle_count": int(metrics["greedy_vehicle_count"].sum()),
        "greedy_internal_deadhead_distance_km": float(
            metrics["greedy_internal_deadhead_distance_km"].sum()
        ),
        "task_service_rate": float(
            (metrics["task_service_rate"] * metrics["task_count"]).sum() / tasks
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
    }
    return metrics, pd.concat(schedule_parts, ignore_index=True), summary


def compare_with_final_vrp(
    greedy_metrics: pd.DataFrame,
    greedy_summary: dict[str, object],
    final_metrics: pd.DataFrame,
    final_summary: dict[str, object],
    minimum_vehicle_summary: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    comparison = greedy_metrics.merge(
        final_metrics[
            ["instance_id", "vehicle_count", "internal_deadhead_distance_km"]
        ].rename(
            columns={
                "vehicle_count": "vrp_vehicle_count",
                "internal_deadhead_distance_km": "vrp_internal_deadhead_distance_km",
            }
        ),
        on="instance_id",
        validate="one_to_one",
    )
    comparison["vrp_vehicle_reduction_vs_greedy"] = (
        comparison["greedy_vehicle_count"] - comparison["vrp_vehicle_count"]
    )
    comparison["vrp_deadhead_reduction_vs_greedy_km"] = (
        comparison["greedy_internal_deadhead_distance_km"]
        - comparison["vrp_internal_deadhead_distance_km"]
    )
    greedy_vehicles = int(greedy_summary["greedy_vehicle_count"])
    vrp_vehicles = int(final_summary["recommended_vehicle_count"])
    greedy_deadhead = float(greedy_summary["greedy_internal_deadhead_distance_km"])
    vrp_deadhead = float(final_summary["recommended_deadhead_distance_km"])
    vehicle_cost = float(final_summary["vehicle_cost_equivalent_km"])
    greedy_proxy_cost = greedy_vehicles * vehicle_cost + greedy_deadhead
    vrp_proxy_cost = vrp_vehicles * vehicle_cost + vrp_deadhead
    minimum_vehicles = int(minimum_vehicle_summary["p90_robust_vehicle_count"])
    minimum_vehicle_deadhead = float(
        minimum_vehicle_summary["internal_deadhead_distance_km"]
    )
    summary = {
        "source_only": "订单数据.xlsx",
        "comparison": "Chronological greedy dispatch versus frozen P90 VRP on the independent December final test.",
        "instances": int(greedy_summary["instances"]),
        "tasks": int(greedy_summary["tasks"]),
        "greedy_vehicle_count": greedy_vehicles,
        "vrp_vehicle_count": vrp_vehicles,
        "vrp_vehicle_reduction_vs_greedy": greedy_vehicles - vrp_vehicles,
        "vrp_vehicle_reduction_rate_vs_greedy": (greedy_vehicles - vrp_vehicles)
        / greedy_vehicles,
        "greedy_internal_deadhead_distance_km": greedy_deadhead,
        "vrp_internal_deadhead_distance_km": vrp_deadhead,
        "vrp_deadhead_reduction_vs_greedy_km": greedy_deadhead - vrp_deadhead,
        "vehicle_cost_equivalent_km": vehicle_cost,
        "greedy_weighted_proxy_cost": greedy_proxy_cost,
        "vrp_weighted_proxy_cost": vrp_proxy_cost,
        "vrp_weighted_proxy_cost_reduction_rate": (greedy_proxy_cost - vrp_proxy_cost)
        / greedy_proxy_cost,
        "minimum_vehicle_p90_vrp_vehicle_count": minimum_vehicles,
        "minimum_vehicle_p90_vrp_deadhead_distance_km": minimum_vehicle_deadhead,
        "minimum_vehicle_p90_vrp_vehicle_reduction_vs_greedy": greedy_vehicles
        - minimum_vehicles,
        "minimum_vehicle_p90_vrp_deadhead_reduction_vs_greedy_km": greedy_deadhead
        - minimum_vehicle_deadhead,
        "instances_where_vrp_uses_fewer_vehicles": int(
            comparison["vrp_vehicle_reduction_vs_greedy"].gt(0).sum()
        ),
        "instances_where_vehicle_count_is_equal": int(
            comparison["vrp_vehicle_reduction_vs_greedy"].eq(0).sum()
        ),
        "instances_where_vrp_uses_more_vehicles": int(
            comparison["vrp_vehicle_reduction_vs_greedy"].lt(0).sum()
        ),
        "both_methods_task_service_rate": [
            greedy_summary["task_service_rate"],
            final_summary["task_service_rate"],
        ],
        "greedy_constraint_violations": int(
            greedy_summary["route_type_violations"]
            + greedy_summary["time_overlap_violations"]
            + greedy_summary["deadhead_endpoint_violations"]
            + greedy_summary["missing_selected_link_paths"]
        ),
        "vrp_constraint_violations": int(
            final_summary["route_type_violations"]
            + final_summary["time_overlap_violations"]
            + final_summary["deadhead_endpoint_violations"]
            + final_summary["missing_selected_link_paths"]
        ),
    }
    return comparison, summary


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "robust_p90_candidate_task_links.csv")
    lane_types = pd.read_csv(args.data_dir / "lane_vehicle_types.csv")
    qualified, compatible_types = prepare_problem(tasks, instances, lane_types)
    preferences = preferred_task_types(qualified, lane_types)
    greedy_metrics, greedy_schedule, greedy_summary = evaluate_greedy_dispatch(
        qualified, compatible_types, preferences, links
    )
    final_metrics = pd.read_csv(
        args.result_dir / "independently_selected_p90_solution_instances.csv"
    )
    final_summary = json.loads(
        (
            args.result_dir / "independently_selected_p90_solution_summary.json"
        ).read_text(encoding="utf-8")
    )
    minimum_vehicle_summary = json.loads(
        (args.result_dir / "robust_p90_vrp_summary.json").read_text(encoding="utf-8")
    )
    comparison, summary = compare_with_final_vrp(
        greedy_metrics,
        greedy_summary,
        final_metrics,
        final_summary,
        minimum_vehicle_summary,
    )
    greedy_schedule.to_csv(
        args.data_dir / "greedy_p90_dispatch_schedules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        args.result_dir / "greedy_vs_vrp_instance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "greedy_vs_vrp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
