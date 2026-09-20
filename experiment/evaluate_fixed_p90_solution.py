#!/usr/bin/env python3
"""Evaluate one frozen P90 vehicle-cost setting without test-set selection."""

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
from optimize_vehicle_deadhead_tradeoff import (
    build_scenario_schedule,
    evaluate_scenario,
    historical_chain_metrics,
    prepare_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估冻结参数的 P90 VRP 方案。")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("processed/company/vrp_final_test")
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    parser.add_argument("--vehicle-cost", type=float, required=True)
    parser.add_argument("--selection-period", required=True)
    parser.add_argument("--output-prefix", default="fixed_p90_solution")
    parser.add_argument("--time-limit", type=float, default=60.0)
    return parser.parse_args()


def evaluate_fixed_solution(
    qualified_tasks: pd.DataFrame,
    task_types: dict[str, set[str]],
    links: pd.DataFrame,
    chains: pd.DataFrame,
    vehicle_cost: float,
    time_limit: float,
    selection_period: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    instance_metrics, scenario = evaluate_scenario(
        qualified_tasks, task_types, links, vehicle_cost, time_limit
    )
    schedule = build_scenario_schedule(
        qualified_tasks, task_types, links, vehicle_cost, time_limit
    )
    historical = historical_chain_metrics(chains, links, "p90")
    historical_vehicles = int(historical["vehicle_count"])
    recommended_vehicles = int(scenario["vehicle_count"])
    historical_deadhead = float(historical["internal_deadhead_distance_km"])
    recommended_deadhead = float(scenario["internal_deadhead_distance_km"])
    summary = {
        "source_only": "订单数据.xlsx",
        "protocol": "Frozen parameter selected before final-test evaluation.",
        "travel_time_stat": "p90",
        "vehicle_cost_equivalent_km": vehicle_cost,
        "parameter_selection_period": selection_period,
        "instances": int(qualified_tasks["instance_id"].nunique()),
        "tasks": len(qualified_tasks),
        "historical_p90_chain_vehicle_count": historical_vehicles,
        "recommended_vehicle_count": recommended_vehicles,
        "vehicle_reduction": historical_vehicles - recommended_vehicles,
        "vehicle_reduction_rate": (historical_vehicles - recommended_vehicles)
        / historical_vehicles,
        "historical_p90_chain_deadhead_distance_km": historical_deadhead,
        "recommended_deadhead_distance_km": recommended_deadhead,
        "deadhead_reduction_km": historical_deadhead - recommended_deadhead,
        "task_service_rate": scenario["task_service_rate"],
        "route_type_violations": scenario["route_type_violations"],
        "time_overlap_violations": scenario["time_overlap_violations"],
        "deadhead_endpoint_violations": scenario["deadhead_endpoint_violations"],
        "missing_selected_link_paths": scenario["missing_selected_link_paths"],
        "all_instances_solved": scenario["all_instances_solved"],
        "no_final_test_parameter_reselection": True,
    }
    return instance_metrics, schedule, summary


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "robust_p90_candidate_task_links.csv")
    lane_types = pd.read_csv(args.data_dir / "lane_vehicle_types.csv")
    chains = pd.read_csv(args.data_dir / "historical_vehicle_chains.csv")
    qualified, task_types = prepare_problem(tasks, instances, lane_types)
    metrics, schedule, summary = evaluate_fixed_solution(
        qualified,
        task_types,
        links,
        chains,
        args.vehicle_cost,
        args.time_limit,
        args.selection_period,
    )
    metrics.to_csv(
        args.result_dir / f"{args.output_prefix}_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    schedule.to_csv(
        args.data_dir / f"{args.output_prefix}_schedules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
