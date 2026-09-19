#!/usr/bin/env python3
"""Evaluate the vehicle-count and internal-deadhead tradeoff on holdout tasks."""

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
from itertools import pairwise
from pathlib import Path

import pandas as pd
from optimize_type_compatible_vrp import (
    attach_route_details,
    solve_type_compatible_path_cover,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算车辆数与空驶距离权衡前沿。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed/company/vrp_holdout"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/company_transport/holdout"),
    )
    parser.add_argument(
        "--vehicle-costs",
        type=str,
        default="1,10,25,50,75,80,85,89,90,100,250,500,1000",
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    return parser.parse_args()


def prepare_problem(
    tasks: pd.DataFrame,
    instances: pd.DataFrame,
    lane_vehicle_types: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    prepared_tasks = tasks.copy()
    prepared_tasks["service_date"] = prepared_tasks["service_date"].astype("string")
    prepared_instances = instances.copy()
    prepared_instances["service_date"] = prepared_instances["service_date"].astype(
        "string"
    )
    qualified = prepared_tasks.merge(
        prepared_instances[
            ["instance_id", "service_date", "component_id", "qualifies_for_vrp"]
        ],
        on=["service_date", "component_id"],
        how="left",
        validate="many_to_one",
    )
    qualified = qualified[qualified["qualifies_for_vrp"].eq(True)].copy()
    lane_types = lane_vehicle_types.groupby(["origin_site_id", "destination_site_id"])[
        "vehicle_type_name"
    ].agg(lambda values: set(values.dropna()))
    task_types: dict[str, set[str]] = {}
    for row in qualified.itertuples(index=False):
        lane = (row.origin_site_id, row.destination_site_id)
        if lane not in lane_types.index:
            raise ValueError(
                f"Task {row.task_id} has no training vehicle-type evidence"
            )
        task_types[str(row.task_id)] = lane_types[lane]
    return qualified, task_types


def evaluate_scenario(
    qualified_tasks: pd.DataFrame,
    task_types: dict[str, set[str]],
    links: pd.DataFrame,
    vehicle_cost_equivalent_km: float | None,
    time_limit: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for instance_id, group in qualified_tasks.groupby("instance_id", sort=True):
        instance_links = links[links["instance_id"] == instance_id]
        task_ids = group["task_id"].astype(str).tolist()
        assignments, selected, diagnostics = solve_type_compatible_path_cover(
            task_ids,
            task_types,
            instance_links,
            time_limit,
            vehicle_cost_equivalent_km,
        )
        route, validation = attach_route_details(assignments, selected, group)
        vehicle_count = int(route["vehicle_id"].nunique())
        deadhead_distance = (
            float(selected["deadhead_distance_km"].sum()) if not selected.empty else 0.0
        )
        rows.append(
            {
                "instance_id": instance_id,
                "task_count": len(group),
                "vehicle_count": vehicle_count,
                "selected_task_links": len(selected),
                "internal_deadhead_distance_km": deadhead_distance,
                "task_service_rate": len(route) / len(group),
                "duplicate_task_assignments": int(route["task_id"].duplicated().sum()),
                "route_type_violations": int(
                    (
                        route.groupby("vehicle_id")["vehicle_type_name"].nunique() > 1
                    ).sum()
                ),
                **validation,
                "solver_success": diagnostics["solver_success"],
                "optimality_gap": diagnostics["optimality_gap"],
            }
        )
    metrics = pd.DataFrame(rows)
    tasks = int(metrics["task_count"].sum())
    vehicles = int(metrics["vehicle_count"].sum())
    deadhead = float(metrics["internal_deadhead_distance_km"].sum())
    summary = {
        "scenario": (
            "minimum_vehicles"
            if vehicle_cost_equivalent_km is None
            else f"vehicle_cost_{vehicle_cost_equivalent_km:g}_km"
        ),
        "vehicle_cost_equivalent_km": vehicle_cost_equivalent_km,
        "tasks": tasks,
        "vehicle_count": vehicles,
        "vehicles_reduced_vs_one_task_one_vehicle": tasks - vehicles,
        "selected_task_links": tasks - vehicles,
        "internal_deadhead_distance_km": deadhead,
        "weighted_proxy_cost": (
            None
            if vehicle_cost_equivalent_km is None
            else vehicle_cost_equivalent_km * vehicles + deadhead
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
        "all_instances_solved": bool(metrics["solver_success"].all()),
    }
    return metrics, summary


def historical_chain_metrics(
    chains: pd.DataFrame, links: pd.DataFrame
) -> dict[str, object]:
    p50_chains = chains[chains["baseline_variant"] == "p50"].sort_values(
        ["historical_chain_id", "sequence"]
    )
    link_distance = {
        (row.instance_id, row.from_task_id, row.to_task_id): float(
            row.deadhead_distance_km
        )
        for row in links.itertuples(index=False)
    }
    distance = 0.0
    transition_count = 0
    for _, group in p50_chains.groupby("historical_chain_id", sort=False):
        records = list(group.itertuples(index=False))
        for first, second in pairwise(records):
            distance += link_distance[
                (first.instance_id, first.task_id, second.task_id)
            ]
            transition_count += 1
    return {
        "scenario": "historical_p50_verifiable_chains",
        "vehicle_cost_equivalent_km": None,
        "tasks": len(p50_chains),
        "vehicle_count": int(p50_chains["historical_chain_id"].nunique()),
        "vehicles_reduced_vs_one_task_one_vehicle": int(
            len(p50_chains) - p50_chains["historical_chain_id"].nunique()
        ),
        "selected_task_links": transition_count,
        "internal_deadhead_distance_km": distance,
        "weighted_proxy_cost": None,
    }


def mark_pareto_frontier(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["pareto_efficient"] = False
    scenario_rows = result[result["scenario"] != "historical_p50_verifiable_chains"]
    for index, row in scenario_rows.iterrows():
        dominated = (
            (scenario_rows["vehicle_count"] <= row["vehicle_count"])
            & (
                scenario_rows["internal_deadhead_distance_km"]
                <= row["internal_deadhead_distance_km"]
            )
            & (
                (scenario_rows["vehicle_count"] < row["vehicle_count"])
                | (
                    scenario_rows["internal_deadhead_distance_km"]
                    < row["internal_deadhead_distance_km"]
                )
            )
        ).any()
        result.loc[index, "pareto_efficient"] = not dominated
    return result


def run_tradeoff(
    qualified_tasks: pd.DataFrame,
    task_types: dict[str, set[str]],
    links: pd.DataFrame,
    chains: pd.DataFrame,
    vehicle_costs: list[float],
    time_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    instance_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for cost in [*vehicle_costs, None]:
        metrics, summary = evaluate_scenario(
            qualified_tasks, task_types, links, cost, time_limit
        )
        metrics["scenario"] = summary["scenario"]
        metrics["vehicle_cost_equivalent_km"] = cost
        instance_parts.append(metrics)
        summaries.append(summary)
    summaries.append(historical_chain_metrics(chains, links))
    frontier = mark_pareto_frontier(pd.DataFrame(summaries))
    scenario_rows = frontier[frontier["scenario"] != "historical_p50_verifiable_chains"]
    historical = frontier[
        frontier["scenario"] == "historical_p50_verifiable_chains"
    ].iloc[0]
    dominates_history = scenario_rows[
        (scenario_rows["vehicle_count"] <= historical["vehicle_count"])
        & (
            scenario_rows["internal_deadhead_distance_km"]
            <= historical["internal_deadhead_distance_km"]
        )
    ].sort_values(["vehicle_count", "internal_deadhead_distance_km"])
    best_no_more_deadhead = dominates_history.iloc[0]
    report = {
        "source_only": "订单数据.xlsx",
        "interpretation": (
            "Vehicle cost is expressed as an equivalent number of deadhead "
            "kilometres because the source has no monetary vehicle cost."
        ),
        "instances": int(qualified_tasks["instance_id"].nunique()),
        "tasks": len(qualified_tasks),
        "vehicle_cost_scenarios_km": vehicle_costs,
        "pareto_scenarios": scenario_rows.loc[
            scenario_rows["pareto_efficient"], "scenario"
        ].tolist(),
        "minimum_observed_vehicle_count": int(scenario_rows["vehicle_count"].min()),
        "maximum_observed_vehicle_count": int(scenario_rows["vehicle_count"].max()),
        "minimum_observed_deadhead_distance_km": float(
            scenario_rows["internal_deadhead_distance_km"].min()
        ),
        "minimum_vehicle_deadhead_distance_km": float(
            scenario_rows.loc[
                scenario_rows["scenario"] == "minimum_vehicles",
                "internal_deadhead_distance_km",
            ].iloc[0]
        ),
        "historical_p50_chain_vehicle_count": int(historical["vehicle_count"]),
        "historical_p50_chain_deadhead_distance_km": float(
            historical["internal_deadhead_distance_km"]
        ),
        "scenarios_dominating_historical_p50_chains": dominates_history[
            "scenario"
        ].tolist(),
        "best_scenario_with_no_more_deadhead_than_history": {
            "scenario": best_no_more_deadhead["scenario"],
            "vehicle_count": int(best_no_more_deadhead["vehicle_count"]),
            "vehicle_reduction": int(
                historical["vehicle_count"] - best_no_more_deadhead["vehicle_count"]
            ),
            "vehicle_reduction_rate": float(
                (historical["vehicle_count"] - best_no_more_deadhead["vehicle_count"])
                / historical["vehicle_count"]
            ),
            "internal_deadhead_distance_km": float(
                best_no_more_deadhead["internal_deadhead_distance_km"]
            ),
            "deadhead_reduction_km": float(
                historical["internal_deadhead_distance_km"]
                - best_no_more_deadhead["internal_deadhead_distance_km"]
            ),
        },
        "all_scenarios_feasible": bool(
            scenario_rows["task_service_rate"].eq(1.0).all()
            and scenario_rows["route_type_violations"].eq(0).all()
            and scenario_rows["time_overlap_violations"].eq(0).all()
            and scenario_rows["deadhead_endpoint_violations"].eq(0).all()
            and scenario_rows["all_instances_solved"].all()
        ),
        "selection_rule": (
            "Choose the scenario matching an independently supplied vehicle "
            "cost per dispatch; no unique monetary optimum is identifiable "
            "from the order data alone."
        ),
    }
    return frontier, pd.concat(instance_parts, ignore_index=True), report


def main() -> None:
    args = parse_args()
    costs = [float(value) for value in args.vehicle_costs.split(",")]
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    links = pd.read_csv(args.data_dir / "time_dependent_candidate_task_links.csv")
    lane_types = pd.read_csv(args.data_dir / "lane_vehicle_types.csv")
    chains = pd.read_csv(args.data_dir / "historical_vehicle_chains.csv")
    qualified, task_types = prepare_problem(tasks, instances, lane_types)
    frontier, instance_metrics, report = run_tradeoff(
        qualified, task_types, links, chains, costs, args.time_limit
    )

    frontier.to_csv(
        args.result_dir / "vehicle_deadhead_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    instance_metrics.to_csv(
        args.result_dir / "vehicle_deadhead_tradeoff_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "vehicle_deadhead_tradeoff_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frontier.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
