#!/usr/bin/env python3
"""Compare fixed historical vehicle types with joint type-route optimization."""

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
from optimize_vehicle_deadhead_tradeoff import evaluate_scenario, prepare_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较固定历史车型与车型路线联合优化。")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("processed/company/vrp_final_test")
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    parser.add_argument("--vehicle-cost-equivalent-km", type=float, default=75.0)
    parser.add_argument("--time-limit", type=float, default=60.0)
    return parser.parse_args()


def fixed_historical_task_types(tasks: pd.DataFrame) -> dict[str, set[str]]:
    if tasks["vehicle_type_name"].isna().any():
        raise ValueError("Qualified tasks contain missing historical vehicle types")
    return {
        str(row.task_id): {str(row.vehicle_type_name)}
        for row in tasks.itertuples(index=False)
    }


def compare_scenarios(
    fixed_metrics: pd.DataFrame,
    fixed_summary: dict[str, object],
    joint_metrics: pd.DataFrame,
    joint_summary: dict[str, object],
    objective_label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fixed = fixed_metrics[
        ["instance_id", "vehicle_count", "internal_deadhead_distance_km"]
    ].rename(
        columns={
            "vehicle_count": "fixed_vehicle_count",
            "internal_deadhead_distance_km": "fixed_deadhead_distance_km",
        }
    )
    joint = joint_metrics[
        ["instance_id", "vehicle_count", "internal_deadhead_distance_km"]
    ].rename(
        columns={
            "vehicle_count": "joint_vehicle_count",
            "internal_deadhead_distance_km": "joint_deadhead_distance_km",
        }
    )
    comparison = fixed.merge(joint, on="instance_id", validate="one_to_one")
    comparison["joint_vehicle_reduction"] = (
        comparison["fixed_vehicle_count"] - comparison["joint_vehicle_count"]
    )
    comparison["joint_deadhead_reduction_km"] = (
        comparison["fixed_deadhead_distance_km"]
        - comparison["joint_deadhead_distance_km"]
    )
    fixed_vehicles = int(fixed_summary["vehicle_count"])
    joint_vehicles = int(joint_summary["vehicle_count"])
    fixed_deadhead = float(fixed_summary["internal_deadhead_distance_km"])
    joint_deadhead = float(joint_summary["internal_deadhead_distance_km"])
    summary = {
        "objective": objective_label,
        "tasks": int(fixed_summary["tasks"]),
        "instances": len(comparison),
        "fixed_historical_type_vehicle_count": fixed_vehicles,
        "joint_type_route_vehicle_count": joint_vehicles,
        "joint_vehicle_reduction": fixed_vehicles - joint_vehicles,
        "fixed_historical_type_deadhead_distance_km": fixed_deadhead,
        "joint_type_route_deadhead_distance_km": joint_deadhead,
        "joint_deadhead_reduction_km": fixed_deadhead - joint_deadhead,
        "instances_where_joint_uses_fewer_vehicles": int(
            comparison["joint_vehicle_reduction"].gt(0).sum()
        ),
        "instances_where_vehicle_count_is_equal": int(
            comparison["joint_vehicle_reduction"].eq(0).sum()
        ),
        "instances_where_joint_uses_more_vehicles": int(
            comparison["joint_vehicle_reduction"].lt(0).sum()
        ),
        "both_solutions_feasible": bool(
            fixed_summary["task_service_rate"] == 1.0
            and joint_summary["task_service_rate"] == 1.0
            and fixed_summary["route_type_violations"] == 0
            and joint_summary["route_type_violations"] == 0
            and fixed_summary["time_overlap_violations"] == 0
            and joint_summary["time_overlap_violations"] == 0
        ),
    }
    if fixed_summary["weighted_proxy_cost"] is not None:
        fixed_cost = float(fixed_summary["weighted_proxy_cost"])
        joint_cost = float(joint_summary["weighted_proxy_cost"])
        summary.update(
            {
                "fixed_weighted_proxy_cost": fixed_cost,
                "joint_weighted_proxy_cost": joint_cost,
                "joint_weighted_proxy_cost_reduction_rate": (fixed_cost - joint_cost)
                / fixed_cost,
            }
        )
    return comparison, summary


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "robust_p90_candidate_task_links.csv")
    lane_types = pd.read_csv(args.data_dir / "lane_vehicle_types.csv")
    qualified, joint_types = prepare_problem(tasks, instances, lane_types)
    fixed_types = fixed_historical_task_types(qualified)

    summaries: dict[str, object] = {
        "source_only": "订单数据.xlsx",
        "comparison": (
            "Fixed observed task vehicle types versus joint empirical-compatible "
            "vehicle-type and route optimization on the independent December test."
        ),
        "travel_time_stat": "p90",
        "vehicle_type_scope": (
            "Joint choices are restricted to vehicle types observed on the same "
            "directed lane in training data."
        ),
    }
    comparison_parts: list[pd.DataFrame] = []
    for objective_label, vehicle_cost in [
        ("minimum_vehicles_then_deadhead", None),
        (
            f"vehicle_cost_{args.vehicle_cost_equivalent_km:g}_km",
            args.vehicle_cost_equivalent_km,
        ),
    ]:
        fixed_metrics, fixed_summary = evaluate_scenario(
            qualified, fixed_types, links, vehicle_cost, args.time_limit
        )
        joint_metrics, joint_summary = evaluate_scenario(
            qualified, joint_types, links, vehicle_cost, args.time_limit
        )
        comparison, summary = compare_scenarios(
            fixed_metrics,
            fixed_summary,
            joint_metrics,
            joint_summary,
            objective_label,
        )
        comparison["objective"] = objective_label
        comparison_parts.append(comparison)
        summaries[objective_label] = summary

    output = pd.concat(comparison_parts, ignore_index=True)
    output.to_csv(
        args.result_dir / "fixed_vs_joint_vehicle_routing_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "fixed_vs_joint_vehicle_routing_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
