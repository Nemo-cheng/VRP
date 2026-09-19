#!/usr/bin/env python3
"""Re-evaluate task links with time-dependent travel and optimize the VRP."""

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

from optimize_type_compatible_vrp import optimize_instances
from vrp_time_utils import evaluate_time_dependent_path, period_for_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="优化分时段车型兼容 VRP。")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("processed/company/vrp")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument(
        "--travel-time-stat", choices=["p50", "p90"], default="p50"
    )
    return parser.parse_args()


def build_time_dependent_links(
    links: pd.DataFrame,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    periods: pd.DataFrame,
    travel_time_stat: str = "p50",
) -> tuple[pd.DataFrame, dict[str, int]]:
    duration_column = f"duration_hours_{travel_time_stat}"
    if duration_column not in edges.columns or duration_column not in periods.columns:
        raise ValueError(f"Missing travel-time column: {duration_column}")
    task_time = tasks.set_index("task_id")[["arrived_at", "departed_at"]].copy()
    task_time["arrived_at"] = pd.to_datetime(task_time["arrived_at"])
    task_time["departed_at"] = pd.to_datetime(task_time["departed_at"])
    edge_lookup = {
        (row.origin_site_id, row.destination_site_id): float(
            getattr(row, duration_column)
        )
        for row in edges[edges["high_confidence"]].itertuples(index=False)
    }
    reliable_periods = periods[periods["period_estimate_available"]]
    period_lookup = {
        (row.origin_site_id, row.destination_site_id, row.departure_period): float(
            getattr(row, duration_column)
        )
        for row in reliable_periods.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    period_edge_uses = 0
    total_edge_uses = 0
    for link in links.itertuples(index=False):
        path = str(link.deadhead_path).split(">")
        start_time = task_time.loc[link.from_task_id, "arrived_at"]
        duration, reliable_edges, total_edges = evaluate_time_dependent_path(
            path, start_time, edge_lookup, period_lookup
        )
        period_edge_uses += reliable_edges
        total_edge_uses += total_edges
        ready_at = start_time + pd.to_timedelta(duration, unit="h")
        next_departure = task_time.loc[link.to_task_id, "departed_at"]
        if ready_at <= next_departure:
            row = link._asdict()
            row["static_deadhead_duration_hours_p50"] = row[
                "deadhead_duration_hours_p50"
            ]
            row["deadhead_duration_hours_p50"] = duration
            row["available_slack_hours"] = (
                next_departure - ready_at
            ).total_seconds() / 3600
            row["period_edge_uses"] = reliable_edges
            row["path_edge_count"] = total_edges
            rows.append(row)
    diagnostics = {
        "static_candidate_links": int(len(links)),
        "time_feasible_candidate_links": int(len(rows)),
        "links_removed_by_time_dependence": int(len(links) - len(rows)),
        "deadhead_edge_traversals": total_edge_uses,
        "reliable_period_edge_uses": period_edge_uses,
        "travel_time_stat": travel_time_stat,
    }
    return pd.DataFrame(rows), diagnostics


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "candidate_task_links.csv")
    edges = pd.read_csv(args.data_dir / "network_edges.csv")
    periods = pd.read_csv(args.data_dir / "edge_period_stats.csv")
    is_robust = args.travel_time_stat == "p90"
    baseline_filename = (
        "time_dependent_vrp_comparison.csv"
        if is_robust
        else "type_compatible_vrp_comparison.csv"
    )
    baseline_vehicle_column = (
        "time_dependent_vehicle_count"
        if is_robust
        else "type_compatible_vehicle_count"
    )
    static_metrics = pd.read_csv(args.result_dir / baseline_filename)
    lane_type_path = args.data_dir / "lane_vehicle_types.csv"
    lane_type_compatibility = None
    if lane_type_path.exists():
        lane_type_rows = pd.read_csv(lane_type_path)
        lane_type_compatibility = lane_type_rows.groupby(
            ["origin_site_id", "destination_site_id"]
        )["vehicle_type_name"].agg(lambda values: set(values.dropna()))
    time_links, diagnostics = build_time_dependent_links(
        links, tasks, edges, periods, args.travel_time_stat
    )
    metrics, schedules, summary = optimize_instances(
        tasks,
        instances,
        time_links,
        static_metrics,
        args.time_limit,
        baseline_vehicle_column=baseline_vehicle_column,
        lane_type_compatibility=lane_type_compatibility,
    )
    if is_robust:
        renamed_columns = {
            "basic_vrp_vehicle_count": "p50_time_dependent_vehicle_count",
            "type_compatible_vehicle_count": "p90_robust_vehicle_count",
            "vehicles_added_by_type_compatibility": "vehicles_added_by_p90_robustness",
        }
        output_prefix = "robust_p90"
    else:
        renamed_columns = {
            "basic_vrp_vehicle_count": "static_type_compatible_vehicle_count",
            "type_compatible_vehicle_count": "time_dependent_vehicle_count",
            "vehicles_added_by_type_compatibility": "vehicles_added_by_time_dependence",
        }
        output_prefix = "time_dependent"
    metrics = metrics.rename(columns=renamed_columns)
    baseline_output_column = (
        "p50_time_dependent_vehicle_count"
        if is_robust
        else "static_type_compatible_vehicle_count"
    )
    optimized_output_column = (
        "p90_robust_vehicle_count"
        if is_robust
        else "time_dependent_vehicle_count"
    )
    increase_output_column = (
        "vehicles_added_by_p90_robustness"
        if is_robust
        else "vehicles_added_by_time_dependence"
    )
    summary = {
        "source_only": "订单数据.xlsx",
        "comparison": (
            "P50 versus P90 time-dependent type-compatible VRP on the same candidate paths."
            if is_robust
            else "Static type-compatible VRP versus time-dependent type-compatible VRP on the same candidate paths."
        ),
        **diagnostics,
        "instances": int(len(metrics)),
        "tasks": int(metrics["task_count"].sum()),
        baseline_output_column: int(metrics[baseline_output_column].sum()),
        optimized_output_column: int(metrics[optimized_output_column].sum()),
        increase_output_column: int(metrics[increase_output_column].sum()),
        "internal_deadhead_distance_km": float(
            metrics["internal_deadhead_distance_km"].sum()
        ),
        "task_service_rate": float(
            (metrics["task_service_rate"] * metrics["task_count"]).sum()
            / metrics["task_count"].sum()
        ),
        "route_type_violations": int(metrics["route_type_violations"].sum()),
        "time_overlap_violations": int(
            metrics["time_overlap_violations"].sum()
        ),
        "deadhead_endpoint_violations": int(
            metrics["deadhead_endpoint_violations"].sum()
        ),
        "missing_selected_link_paths": int(
            metrics["missing_selected_link_paths"].sum()
        ),
        "all_instances_solved": bool(metrics["solver_success"].all()),
    }
    time_links.to_csv(
        args.data_dir / f"{output_prefix}_candidate_task_links.csv",
        index=False,
        encoding="utf-8-sig",
    )
    schedules.to_csv(
        args.data_dir / f"{output_prefix}_vrp_schedules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        args.result_dir / f"{output_prefix}_vrp_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / f"{output_prefix}_vrp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
