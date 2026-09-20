"""Build VRP ablation tables from existing independent-test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RESULT_DIR = Path("results/company_transport/final_test")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def constraint_violations(summary: dict[str, Any]) -> int:
    fields = (
        "duplicate_task_assignments",
        "missing_task_assignments",
        "route_type_violations",
        "time_overlap_violations",
        "deadhead_endpoint_violations",
        "missing_selected_link_paths",
    )
    return sum(int(summary.get(field, 0)) for field in fields)


def build_ablation(result_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    basic = read_json(result_dir / "basic_vrp_summary.json")
    typed = read_json(result_dir / "type_compatible_vrp_summary.json")
    timed = read_json(result_dir / "time_dependent_vrp_summary.json")
    robust = read_json(result_dir / "robust_p90_vrp_summary.json")
    greedy = read_json(result_dir / "greedy_vs_vrp_summary.json")
    fixed_joint = read_json(result_dir / "fixed_vs_joint_vehicle_routing_summary.json")
    selected = read_json(result_dir / "independently_selected_p90_solution_summary.json")

    tasks = int(basic["tasks"])
    instances = int(basic["instances"])
    vehicle_cost = float(selected["vehicle_cost_equivalent_km"])
    if any(
        int(summary["tasks"]) != tasks
        for summary in (typed, timed, robust, greedy, selected)
    ):
        raise ValueError("Ablation inputs do not use the same task set")

    strict_rows = [
        {
            "comparison_family": "constraint_ablation",
            "variant": "one_task_one_vehicle",
            "change_from_next_model": "remove_vehicle_reuse",
            "objective": "minimum_vehicles_then_deadhead",
            "vehicle_count": int(basic["independent_vehicle_count"]),
            "internal_deadhead_distance_km": 0.0,
            "task_service_rate": 1.0,
            "constraint_violations": 0,
        },
        {
            "comparison_family": "constraint_ablation",
            "variant": "basic_vrp",
            "change_from_next_model": "remove_vehicle_type_compatibility",
            "objective": "minimum_vehicles_then_deadhead",
            "vehicle_count": int(basic["vrp_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                basic["vrp_internal_deadhead_distance_km"]
            ),
            "task_service_rate": float(basic["task_service_rate"]),
            "constraint_violations": constraint_violations(basic),
        },
        {
            "comparison_family": "constraint_ablation",
            "variant": "static_type_compatible_vrp",
            "change_from_next_model": "remove_time_dependent_travel",
            "objective": "minimum_vehicles_then_deadhead",
            "vehicle_count": int(typed["type_compatible_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                typed["internal_deadhead_distance_km"]
            ),
            "task_service_rate": float(typed["task_service_rate"]),
            "constraint_violations": constraint_violations(typed),
        },
        {
            "comparison_family": "constraint_ablation",
            "variant": "p50_time_dependent_vrp",
            "change_from_next_model": "remove_p90_robustness",
            "objective": "minimum_vehicles_then_deadhead",
            "vehicle_count": int(timed["time_dependent_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                timed["internal_deadhead_distance_km"]
            ),
            "task_service_rate": float(timed["task_service_rate"]),
            "constraint_violations": constraint_violations(timed),
        },
        {
            "comparison_family": "constraint_ablation",
            "variant": "p90_joint_type_vrp",
            "change_from_next_model": "full_strict_model",
            "objective": "minimum_vehicles_then_deadhead",
            "vehicle_count": int(robust["p90_robust_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                robust["internal_deadhead_distance_km"]
            ),
            "task_service_rate": float(robust["task_service_rate"]),
            "constraint_violations": constraint_violations(robust),
        },
    ]

    fixed = fixed_joint[f"vehicle_cost_{vehicle_cost:g}_km"]
    decision_rows = [
        {
            "comparison_family": "decision_ablation",
            "variant": "chronological_greedy",
            "change_from_next_model": "remove_global_optimization",
            "objective": f"vehicle_cost_{vehicle_cost:g}_km",
            "vehicle_count": int(greedy["greedy_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                greedy["greedy_internal_deadhead_distance_km"]
            ),
            "task_service_rate": float(greedy["both_methods_task_service_rate"][0]),
            "constraint_violations": int(greedy["greedy_constraint_violations"]),
        },
        {
            "comparison_family": "decision_ablation",
            "variant": "fixed_type_global_vrp",
            "change_from_next_model": "remove_joint_vehicle_type_choice",
            "objective": f"vehicle_cost_{vehicle_cost:g}_km",
            "vehicle_count": int(fixed["fixed_historical_type_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                fixed["fixed_historical_type_deadhead_distance_km"]
            ),
            "task_service_rate": 1.0,
            "constraint_violations": 0 if fixed["both_solutions_feasible"] else 1,
        },
        {
            "comparison_family": "decision_ablation",
            "variant": "p90_joint_type_balanced_vrp",
            "change_from_next_model": "full_balanced_model",
            "objective": f"vehicle_cost_{vehicle_cost:g}_km",
            "vehicle_count": int(selected["recommended_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                selected["recommended_deadhead_distance_km"]
            ),
            "task_service_rate": float(selected["task_service_rate"]),
            "constraint_violations": constraint_violations(selected),
        },
        {
            "comparison_family": "decision_ablation",
            "variant": "p90_joint_type_minimum_vehicle_vrp",
            "change_from_next_model": "remove_deadhead_control",
            "objective": "minimum_vehicles_then_deadhead",
            "vehicle_count": int(robust["p90_robust_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                robust["internal_deadhead_distance_km"]
            ),
            "task_service_rate": float(robust["task_service_rate"]),
            "constraint_violations": constraint_violations(robust),
        },
    ]

    table = pd.DataFrame([*strict_rows, *decision_rows])
    table.insert(0, "instances", instances)
    table.insert(1, "tasks", tasks)
    table["weighted_proxy_cost"] = (
        table["vehicle_count"] * vehicle_cost
        + table["internal_deadhead_distance_km"]
    )
    table.loc[
        table["objective"].eq("minimum_vehicles_then_deadhead"),
        "weighted_proxy_cost",
    ] = pd.NA

    full = table[table["variant"].eq("p90_joint_type_balanced_vrp")].iloc[0]
    decision_mask = table["comparison_family"].eq("decision_ablation")
    table.loc[decision_mask, "proxy_cost_change_vs_full_rate"] = (
        table.loc[decision_mask, "weighted_proxy_cost"]
        - float(full["weighted_proxy_cost"])
    ) / float(full["weighted_proxy_cost"])

    report = {
        "source_only": "existing independent December final-test results",
        "instances": instances,
        "tasks": tasks,
        "vehicle_cost_equivalent_km": vehicle_cost,
        "vehicle_cost_parameter_source": (
            "selected on November validation data and frozen before December testing"
        ),
        "families": {
            "constraint_ablation": (
                "Uses the minimum-vehicles-then-deadhead objective and adds empirical "
                "feasibility constraints step by step."
            ),
            "decision_ablation": (
                "Uses the frozen 75 km proxy where applicable to isolate global "
                "optimization, joint type choice and deadhead control."
            ),
        },
        "key_effects": {
            "global_optimization_proxy_cost_reduction_rate_vs_greedy": float(
                greedy["vrp_weighted_proxy_cost_reduction_rate"]
            ),
            "joint_type_choice_proxy_cost_reduction_rate_vs_fixed": float(
                fixed["joint_weighted_proxy_cost_reduction_rate"]
            ),
            "p90_additional_vehicle_chains_vs_p50": int(
                robust["vehicles_added_by_p90_robustness"]
            ),
            "deadhead_control_additional_vehicle_chains": int(
                selected["recommended_vehicle_count"]
                - robust["p90_robust_vehicle_count"]
            ),
            "deadhead_control_reduction_km": float(
                robust["internal_deadhead_distance_km"]
                - selected["recommended_deadhead_distance_km"]
            ),
        },
        "all_variants_feasible": bool(
            table["task_service_rate"].eq(1.0).all()
            and table["constraint_violations"].eq(0).all()
        ),
    }
    return table, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总独立测试集VRP消融结果。")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, report = build_ablation(args.result_dir)
    table.to_csv(
        args.result_dir / "vrp_ablation.csv", index=False, encoding="utf-8-sig"
    )
    (args.result_dir / "vrp_ablation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
