#!/usr/bin/env python3
"""Combine the Excel-only VRP experiment stages into one audit summary."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 VRP 实验结果。")
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_threshold(directory: Path, threshold: int) -> dict[str, object]:
    readiness = read_json(directory / "vrp_data_readiness.json")
    basic = read_json(directory / "basic_vrp_summary.json")
    vehicle_type = read_json(directory / "type_compatible_vrp_summary.json")
    time_dependent = read_json(directory / "time_dependent_vrp_summary.json")
    robust_p90 = read_json(directory / "robust_p90_vrp_summary.json")
    paths = read_json(directory / "task_path_option_summary.json")
    basic_vehicles = int(basic["vrp_vehicle_count"])
    type_vehicles = int(vehicle_type["type_compatible_vehicle_count"])
    time_vehicles = int(time_dependent["time_dependent_vehicle_count"])
    return {
        "min_edge_observations": threshold,
        "network_covered_tasks": readiness["tasks"]["network_covered_tasks"],
        "qualified_instances": readiness["instances"]["qualified_instances"],
        "qualified_tasks": readiness["instances"]["qualified_tasks"],
        "basic_vehicle_reduction_rate": basic["vehicle_reduction_rate"],
        "basic_vrp_vehicle_count": basic_vehicles,
        "type_compatible_vehicle_count": type_vehicles,
        "type_constraint_vehicle_increase_rate": (type_vehicles - basic_vehicles)
        / basic_vehicles,
        "time_dependent_vehicle_count": time_vehicles,
        "time_constraint_vehicle_increase_rate": (time_vehicles - type_vehicles)
        / type_vehicles,
        "links_removed_by_time_dependence": time_dependent[
            "links_removed_by_time_dependence"
        ],
        "time_dependent_deadhead_distance_km": time_dependent[
            "internal_deadhead_distance_km"
        ],
        "p90_robust_vehicle_count": robust_p90["p90_robust_vehicle_count"],
        "p90_vehicle_increase_rate": (
            robust_p90["p90_robust_vehicle_count"]
            - robust_p90["p50_time_dependent_vehicle_count"]
        )
        / robust_p90["p50_time_dependent_vehicle_count"],
        "p90_removed_candidate_links": robust_p90["links_removed_by_time_dependence"],
        "p90_task_service_rate": robust_p90["task_service_rate"],
        "p90_time_overlap_violations": robust_p90["time_overlap_violations"],
        "alternative_path_task_share": paths["alternative_path_task_share"],
        "task_service_rate": time_dependent["task_service_rate"],
        "route_type_violations": time_dependent["route_type_violations"],
        "all_instances_solved": time_dependent["all_instances_solved"],
    }


def summarize_holdout(directory: Path) -> dict[str, object]:
    readiness = read_json(directory / "vrp_data_readiness.json")
    basic = read_json(directory / "basic_vrp_summary.json")
    vehicle_type = read_json(directory / "type_compatible_vrp_summary.json")
    time_dependent = read_json(directory / "time_dependent_vrp_summary.json")
    robust_p90 = read_json(directory / "robust_p90_vrp_summary.json")
    historical = read_json(directory / "historical_vehicle_baseline_summary.json")
    tradeoff = read_json(directory / "vehicle_deadhead_tradeoff_summary.json")
    robust_tradeoff = read_json(
        directory / "robust_p90_vehicle_deadhead_tradeoff_summary.json"
    )
    recommended_validation = read_json(
        directory / "recommended_p90_validation_summary.json"
    )
    return {
        "protocol": readiness["validation_protocol"],
        "qualified_instances": readiness["instances"]["qualified_instances"],
        "qualified_tasks": readiness["instances"]["qualified_tasks"],
        "basic_vehicle_reduction_rate": basic["vehicle_reduction_rate"],
        "basic_vrp_vehicle_count": basic["vrp_vehicle_count"],
        "type_compatible_vehicle_count": vehicle_type["type_compatible_vehicle_count"],
        "time_dependent_vehicle_count": time_dependent["time_dependent_vehicle_count"],
        "links_removed_by_time_dependence": time_dependent[
            "links_removed_by_time_dependence"
        ],
        "task_service_rate": time_dependent["task_service_rate"],
        "route_type_violations": time_dependent["route_type_violations"],
        "time_overlap_violations": time_dependent["time_overlap_violations"],
        "deadhead_endpoint_violations": time_dependent["deadhead_endpoint_violations"],
        "all_instances_solved": time_dependent["all_instances_solved"],
        "p90_robust_vehicle_count": robust_p90["p90_robust_vehicle_count"],
        "p90_vehicles_added": robust_p90["vehicles_added_by_p90_robustness"],
        "p90_removed_candidate_links": robust_p90["links_removed_by_time_dependence"],
        "p90_task_service_rate": robust_p90["task_service_rate"],
        "p90_time_overlap_violations": robust_p90["time_overlap_violations"],
        "p90_all_instances_solved": robust_p90["all_instances_solved"],
        "raw_historical_vehicle_count": historical["raw_historical_vehicle_count"],
        "p50_historical_chain_count": historical["p50_historical_chain_count"],
        "p90_historical_chain_count": historical["p90_historical_chain_count"],
        "p50_vehicle_reduction_vs_historical_chains": historical[
            "p50_vehicle_reduction_vs_historical_chains"
        ],
        "p90_vehicle_reduction_vs_historical_chains": historical[
            "p90_vehicle_reduction_vs_historical_chains"
        ],
        "historical_p50_chain_deadhead_distance_km": tradeoff[
            "historical_p50_chain_deadhead_distance_km"
        ],
        "best_no_more_deadhead_scenario": tradeoff[
            "best_scenario_with_no_more_deadhead_than_history"
        ],
        "all_tradeoff_scenarios_feasible": tradeoff["all_scenarios_feasible"],
        "historical_p90_chain_deadhead_distance_km": robust_tradeoff[
            "historical_p90_chain_deadhead_distance_km"
        ],
        "best_p90_no_more_deadhead_scenario": robust_tradeoff[
            "best_scenario_with_no_more_deadhead_than_history"
        ],
        "all_p90_tradeoff_scenarios_feasible": robust_tradeoff[
            "all_scenarios_feasible"
        ],
        "recommended_p90_instance_validation": {
            "instances_with_vehicle_reduction": recommended_validation[
                "instances_with_vehicle_reduction"
            ],
            "instances_with_vehicle_increase": recommended_validation[
                "instances_with_vehicle_increase"
            ],
            "instances_with_no_deadhead_increase": recommended_validation[
                "instances_with_no_deadhead_increase"
            ],
            "instances_with_deadhead_increase": recommended_validation[
                "instances_with_deadhead_increase"
            ],
            "aggregate_vehicle_reduction_rate_95_ci": recommended_validation[
                "aggregate_vehicle_reduction_rate_95_ci"
            ],
            "aggregate_deadhead_reduction_km_95_ci": recommended_validation[
                "aggregate_deadhead_reduction_km_95_ci"
            ],
        },
    }


def summarize_independent_validation(result_dir: Path) -> dict[str, object]:
    validation_readiness = read_json(
        result_dir / "validation" / "vrp_data_readiness.json"
    )
    validation_choice = read_json(
        result_dir / "validation" / "robust_p90_vehicle_deadhead_tradeoff_summary.json"
    )
    final_readiness = read_json(result_dir / "final_test" / "vrp_data_readiness.json")
    final_solution = read_json(
        result_dir / "final_test" / "independently_selected_p90_solution_summary.json"
    )
    final_validation = read_json(
        result_dir / "final_test" / "independently_selected_p90_validation_summary.json"
    )
    greedy_comparison = read_json(
        result_dir / "final_test" / "greedy_vs_vrp_summary.json"
    )
    retained_solutions = read_json(
        result_dir / "final_test" / "retained_p90_solutions.json"
    )
    selected = validation_choice["best_scenario_with_no_more_deadhead_than_history"]
    return {
        "validation_period": validation_readiness["validation_protocol"],
        "validation_instances": validation_readiness["instances"][
            "qualified_instances"
        ],
        "validation_instance_gate_met": validation_readiness["gate_checks"][
            "enough_independent_instances"
        ],
        "selected_vehicle_cost_equivalent_km": selected["vehicle_cost_equivalent_km"],
        "final_test_period": final_readiness["validation_protocol"],
        "final_test_instances": final_readiness["instances"]["qualified_instances"],
        "final_test_instance_gate_met": final_readiness["gate_checks"][
            "enough_independent_instances"
        ],
        "historical_deadhead_controlled_solution": final_solution,
        "instance_validation": {
            "instances_with_vehicle_reduction": final_validation[
                "instances_with_vehicle_reduction"
            ],
            "instances_with_equal_vehicle_count": final_validation[
                "instances_with_equal_vehicle_count"
            ],
            "instances_with_vehicle_increase": final_validation[
                "instances_with_vehicle_increase"
            ],
            "aggregate_vehicle_reduction_rate_95_ci": final_validation[
                "aggregate_vehicle_reduction_rate_95_ci"
            ],
            "aggregate_deadhead_reduction_km_95_ci": final_validation[
                "aggregate_deadhead_reduction_km_95_ci"
            ],
        },
        "greedy_dispatch_comparison": greedy_comparison,
        "retained_solutions": retained_solutions,
    }


def build_summary(result_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    directories = {
        5: result_dir / "sensitivity" / "edge5",
        10: result_dir,
        20: result_dir / "sensitivity" / "edge20",
    }
    rows = [
        summarize_threshold(directory, threshold)
        for threshold, directory in directories.items()
    ]
    table = pd.DataFrame(rows).sort_values("min_edge_observations")
    all_feasible = bool(
        table["task_service_rate"].eq(1.0).all()
        and table["route_type_violations"].eq(0).all()
        and table["all_instances_solved"].all()
        and table["p90_task_service_rate"].eq(1.0).all()
        and table["p90_time_overlap_violations"].eq(0).all()
    )
    stable_basic_effect = bool(
        table["basic_vehicle_reduction_rate"].between(0.20, 0.35).all()
    )
    path_selection_supported = bool(
        (table["alternative_path_task_share"] >= 0.05).all()
    )
    holdout = summarize_holdout(result_dir / "holdout")
    independent = summarize_independent_validation(result_dir)
    holdout_feasible = bool(
        holdout["task_service_rate"] == 1.0
        and holdout["route_type_violations"] == 0
        and holdout["time_overlap_violations"] == 0
        and holdout["deadhead_endpoint_violations"] == 0
        and holdout["all_instances_solved"]
        and holdout["p90_task_service_rate"] == 1.0
        and holdout["p90_time_overlap_violations"] == 0
        and holdout["p90_all_instances_solved"]
    )
    report = {
        "source_only": "订单数据.xlsx",
        "completed_comparisons": [
            "independent tasks versus basic VRP",
            "basic VRP versus empirical vehicle-type-compatible VRP",
            "static versus time-dependent vehicle-type-compatible VRP",
            "network observation threshold sensitivity at 5, 10 and 20",
            "historical vehicle assignments versus P50 and P90 VRP on chronological holdout",
            "vehicle-count versus internal-deadhead tradeoff on chronological holdout",
            "instance-level bootstrap validation of the recommended P90 solution",
            "November parameter selection followed by frozen December final testing",
            "chronological greedy dispatch versus global P90 VRP on the independent final test",
        ],
        "gate_checks": {
            "all_reported_vrp_solutions_feasible": all_feasible,
            "basic_vrp_effect_stable_across_thresholds": stable_basic_effect,
            "loaded_path_selection_has_sufficient_coverage": path_selection_supported,
            "chronological_holdout_feasible": holdout_feasible,
            "historical_baseline_improved_by_p50_vrp": holdout[
                "p50_vehicle_reduction_vs_historical_chains"
            ]
            > 0,
            "historical_baseline_improved_by_p90_vrp": holdout[
                "p90_vehicle_reduction_vs_historical_chains"
            ]
            > 0,
            "vehicle_and_deadhead_both_improved_over_history": (
                holdout["best_no_more_deadhead_scenario"]["vehicle_reduction"] > 0
                and holdout["best_no_more_deadhead_scenario"]["deadhead_reduction_km"]
                > 0
                and holdout["all_tradeoff_scenarios_feasible"]
            ),
            "p90_vehicle_and_deadhead_both_improved_over_history": (
                holdout["best_p90_no_more_deadhead_scenario"]["vehicle_reduction"] > 0
                and holdout["best_p90_no_more_deadhead_scenario"][
                    "deadhead_reduction_km"
                ]
                > 0
                and holdout["all_p90_tradeoff_scenarios_feasible"]
            ),
            "recommended_p90_reduces_vehicles_in_every_instance": (
                holdout["recommended_p90_instance_validation"][
                    "instances_with_vehicle_reduction"
                ]
                == holdout["qualified_instances"]
                and holdout["recommended_p90_instance_validation"][
                    "instances_with_vehicle_increase"
                ]
                == 0
            ),
            "independent_final_test_instance_gate_met": independent[
                "final_test_instance_gate_met"
            ],
            "independent_final_test_vehicle_ci_above_zero": independent[
                "instance_validation"
            ]["aggregate_vehicle_reduction_rate_95_ci"][0]
            > 0,
            "independent_final_test_has_no_vehicle_increase": independent[
                "instance_validation"
            ]["instances_with_vehicle_increase"]
            == 0,
            "minimum_vehicle_p90_vrp_dominates_greedy": (
                independent["greedy_dispatch_comparison"][
                    "minimum_vehicle_p90_vrp_vehicle_reduction_vs_greedy"
                ]
                > 0
                and independent["greedy_dispatch_comparison"][
                    "minimum_vehicle_p90_vrp_deadhead_reduction_vs_greedy_km"
                ]
                > 0
            ),
            "frozen_weighted_p90_vrp_improves_proxy_cost": independent[
                "greedy_dispatch_comparison"
            ]["vrp_weighted_proxy_cost_reduction_rate"]
            > 0,
        },
        "main_findings": {
            "basic_vehicle_reduction_rate_range": [
                float(table["basic_vehicle_reduction_rate"].min()),
                float(table["basic_vehicle_reduction_rate"].max()),
            ],
            "vehicle_type_compatibility_always_increases_vehicle_count": bool(
                table["type_constraint_vehicle_increase_rate"].gt(0).all()
            ),
            "time_dependence_always_removes_static_links": bool(
                table["links_removed_by_time_dependence"].gt(0).all()
            ),
            "p90_always_increases_vehicle_count": bool(
                table["p90_vehicle_increase_rate"].gt(0).all()
            ),
            "p90_vehicle_increase_rate_range": [
                float(table["p90_vehicle_increase_rate"].min()),
                float(table["p90_vehicle_increase_rate"].max()),
            ],
            "alternative_path_share_range": [
                float(table["alternative_path_task_share"].min()),
                float(table["alternative_path_task_share"].max()),
            ],
        },
        "decision": {
            "proceed_with_vrp_comparison": all_feasible
            and stable_basic_effect
            and holdout_feasible,
            "use_loaded_path_choice_as_main_experiment": path_selection_supported,
            "loaded_path_choice_role": "supplementary analysis"
            if not path_selection_supported
            else "main experiment",
            "solution_selection": "both_retained_no_economic_ranking",
            "retained_solutions": independent["retained_solutions"][
                "retained_solutions"
            ],
        },
        "chronological_holdout": holdout,
        "independent_parameter_validation": independent,
        "scope_note": "Vehicle counts and reductions apply to the observed task sample, not the company's complete fleet.",
    }
    return table, report


def main() -> None:
    args = parse_args()
    table, report = build_summary(args.result_dir)
    table.to_csv(
        args.result_dir / "vrp_robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "vrp_experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
