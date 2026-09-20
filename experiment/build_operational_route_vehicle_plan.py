#!/usr/bin/env python3
"""Assign the lowest operating-energy-cost feasible vehicle to each final route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.build_carbon_baseline import interval_values, load_vehicle_parameters
from experiment.validate_parameter_registry import load_registry

CHAIN_KEY = ["instance_id", "vehicle_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最终路线的稳健车型配置方案。")
    parser.add_argument(
        "--candidate-detail",
        type=Path,
        default=Path(
            "results/company_transport/final_test/"
            "candidate_vehicle_feasibility_detail.csv"
        ),
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument("--vehicles", type=Path, default=Path("车辆数据.xlsx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    parser.add_argument("--scenario", default="lower_bound")
    return parser.parse_args()


def candidate_cost_table(registry: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": f"candidate_{index:02d}",
                "candidate_energy_cost_cny_per_km": float(
                    candidate["energy_cost_cny_per_km"]
                ),
                "candidate_purchase_cost_cny": float(
                    candidate["purchase_cost_cny"]
                ),
                "candidate_maintenance_cost_cny_per_year": float(
                    candidate["maintenance_cost_cny_per_year"]
                ),
            }
            for index, candidate in enumerate(
                registry["candidate_vehicles"], start=1
            )
        ]
    )


def build_route_vehicle_assignment(
    candidate_detail: pd.DataFrame,
    schedule: pd.DataFrame,
    registry: dict[str, Any],
    scenario: str = "lower_bound",
) -> pd.DataFrame:
    selected = candidate_detail[candidate_detail["scenario"].eq(scenario)].copy()
    selected = selected.merge(
        candidate_cost_table(registry),
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    route_context = schedule.groupby(CHAIN_KEY, as_index=False).agg(
        tasks=("task_id", "size"),
        historical_vehicle_type=("vehicle_type_name", "first"),
        loaded_distance_km=("distance_km", "sum"),
        deadhead_distance_km=("deadhead_to_next_distance_km", "sum"),
    )
    route_context["route_distance_km"] = (
        route_context["loaded_distance_km"]
        + route_context["deadhead_distance_km"]
    )
    rows: list[dict[str, object]] = []
    for chain_values, group in selected.groupby(CHAIN_KEY, sort=True):
        feasible = group[group["technical_feasible"]].sort_values(
            ["candidate_energy_cost_cny_per_km", "candidate_id"]
        )
        operating_window_feasible = bool(
            group["operating_window_feasible"].iloc[0]
        )
        if feasible.empty:
            assignment = {
                "assigned_candidate_id": pd.NA,
                "assigned_model": "retain_existing_vehicle",
                "assigned_energy": "diesel",
                "assigned_energy_cost_cny_per_km": pd.NA,
                "assignment_status": (
                    "retain_pending_operating_plan"
                    if not operating_window_feasible
                    else "retain_no_candidate_feasible"
                ),
            }
        else:
            best = feasible.iloc[0]
            assignment = {
                "assigned_candidate_id": best["candidate_id"],
                "assigned_model": best["candidate_model"],
                "assigned_energy": best["energy"],
                "assigned_energy_cost_cny_per_km": best[
                    "candidate_energy_cost_cny_per_km"
                ],
                "assignment_status": "candidate_selected",
            }
        rows.append(
            {
                "instance_id": chain_values[0],
                "vehicle_id": chain_values[1],
                "scenario": scenario,
                "operating_window_feasible": operating_window_feasible,
                **assignment,
            }
        )
    assignment = pd.DataFrame(rows).merge(
        route_context, on=CHAIN_KEY, how="left", validate="one_to_one"
    )
    return assignment


def summarize_operating_cost(
    assignment: pd.DataFrame,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    diesel_min, diesel_max = interval_values(
        registry, "existing_diesel_energy_cost"
    )
    selected = assignment["assignment_status"].eq("candidate_selected")
    assignment = assignment.copy()
    assignment["assigned_energy_cost_cny_per_km"] = pd.to_numeric(
        assignment["assigned_energy_cost_cny_per_km"], errors="coerce"
    )
    assignment["baseline_energy_cost_cny_min"] = (
        assignment["route_distance_km"] * diesel_min
    )
    assignment["baseline_energy_cost_cny_max"] = (
        assignment["route_distance_km"] * diesel_max
    )
    assignment["plan_energy_cost_cny_min"] = assignment[
        "baseline_energy_cost_cny_min"
    ]
    assignment["plan_energy_cost_cny_max"] = assignment[
        "baseline_energy_cost_cny_max"
    ]
    assignment.loc[selected, "plan_energy_cost_cny_min"] = (
        assignment.loc[selected, "route_distance_km"]
        * assignment.loc[selected, "assigned_energy_cost_cny_per_km"]
    )
    assignment.loc[selected, "plan_energy_cost_cny_max"] = assignment.loc[
        selected, "plan_energy_cost_cny_min"
    ]
    by_assignment = (
        assignment.groupby(
            ["assignment_status", "assigned_energy", "assigned_model"],
            as_index=False,
        )
        .agg(
            route_chains=("vehicle_id", "size"),
            route_distance_km=("route_distance_km", "sum"),
            plan_energy_cost_cny_min=("plan_energy_cost_cny_min", "sum"),
            plan_energy_cost_cny_max=("plan_energy_cost_cny_max", "sum"),
        )
        .sort_values(["assignment_status", "assigned_energy", "assigned_model"])
    )
    baseline_min = float(assignment["baseline_energy_cost_cny_min"].sum())
    baseline_max = float(assignment["baseline_energy_cost_cny_max"].sum())
    plan_min = float(assignment["plan_energy_cost_cny_min"].sum())
    plan_max = float(assignment["plan_energy_cost_cny_max"].sum())
    summary = {
        "scenario": assignment["scenario"].iloc[0],
        "route_chains": len(assignment),
        "candidate_selected_chains": int(selected.sum()),
        "retained_existing_chains": int((~selected).sum()),
        "observed_route_distance_km": float(assignment["route_distance_km"].sum()),
        "existing_diesel_energy_cost_cny_per_km": {
            "minimum": diesel_min,
            "maximum": diesel_max,
            "source": "competition_brief",
        },
        "baseline_energy_cost_cny": {
            "minimum": baseline_min,
            "maximum": baseline_max,
        },
        "plan_energy_cost_cny": {
            "minimum": plan_min,
            "maximum": plan_max,
        },
        "energy_cost_saving_cny": {
            "minimum": baseline_min - plan_min,
            "maximum": baseline_max - plan_max,
        },
        "energy_cost_reduction_rate": {
            "minimum": (baseline_min - plan_min) / baseline_min,
            "maximum": (baseline_max - plan_max) / baseline_max,
        },
        "purchase_and_maintenance_included": False,
        "decision_scope": (
            "仅比较最终测试观测路线的单位里程能耗成本。"
            "购置成本和年维护成本需先确定车辆资产数、使用年限与年行驶里程。"
        ),
    }
    return by_assignment, summary


def build_task_level_plan(
    schedule: pd.DataFrame, assignment: pd.DataFrame
) -> pd.DataFrame:
    assignment_columns = [
        *CHAIN_KEY,
        "scenario",
        "assignment_status",
        "assigned_candidate_id",
        "assigned_model",
        "assigned_energy",
        "assigned_energy_cost_cny_per_km",
    ]
    plan = schedule.merge(
        assignment[assignment_columns],
        on=CHAIN_KEY,
        how="left",
        validate="many_to_one",
    )
    if plan["assignment_status"].isna().any():
        raise ValueError("任务级路线存在无法关联的车型配置")
    return plan.sort_values(["instance_id", "vehicle_id", "sequence"])


def main() -> None:
    args = parse_args()
    registry = load_registry()
    schedule = pd.read_csv(args.schedule)
    vehicles = load_vehicle_parameters(args.vehicles)
    used_types = set(schedule["vehicle_type_name"])
    used_fuels = set(
        vehicles.loc[vehicles["vehicle_type_name"].isin(used_types), "fuel"]
    )
    if used_fuels != {"柴油"}:
        raise ValueError(f"基线单位能耗成本只适用于柴油车，实际车型燃料为: {used_fuels}")
    assignment = build_route_vehicle_assignment(
        pd.read_csv(args.candidate_detail), schedule, registry, args.scenario
    )
    by_assignment, summary = summarize_operating_cost(assignment, registry)
    task_plan = build_task_level_plan(schedule, assignment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(
        args.output_dir / "operational_route_vehicle_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_assignment.to_csv(
        args.output_dir / "operational_route_vehicle_plan_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    task_plan.to_csv(
        args.output_dir / "operational_route_vehicle_task_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "operational_route_vehicle_plan.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
