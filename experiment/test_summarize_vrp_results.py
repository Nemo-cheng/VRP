import json
from pathlib import Path

from summarize_vrp_results import build_summary


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_result_set(directory: Path, alternative_share: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "vrp_data_readiness.json",
        {
            "tasks": {"network_covered_tasks": 100},
            "instances": {"qualified_instances": 20, "qualified_tasks": 100},
        },
    )
    write_json(
        directory / "basic_vrp_summary.json",
        {"vrp_vehicle_count": 75, "vehicle_reduction_rate": 0.25},
    )
    write_json(
        directory / "type_compatible_vrp_summary.json",
        {"type_compatible_vehicle_count": 80},
    )
    write_json(
        directory / "time_dependent_vrp_summary.json",
        {
            "time_dependent_vehicle_count": 82,
            "links_removed_by_time_dependence": 3,
            "internal_deadhead_distance_km": 10.0,
            "task_service_rate": 1.0,
            "route_type_violations": 0,
            "all_instances_solved": True,
        },
    )
    write_json(
        directory / "robust_p90_vrp_summary.json",
        {
            "p50_time_dependent_vehicle_count": 82,
            "p90_robust_vehicle_count": 84,
            "vehicles_added_by_p90_robustness": 2,
            "links_removed_by_time_dependence": 5,
            "task_service_rate": 1.0,
            "time_overlap_violations": 0,
            "all_instances_solved": True,
        },
    )
    write_json(
        directory / "task_path_option_summary.json",
        {"alternative_path_task_share": alternative_share},
    )


def test_summary_applies_feasibility_and_path_coverage_gates(tmp_path: Path) -> None:
    make_result_set(tmp_path, 0.01)
    make_result_set(tmp_path / "sensitivity" / "edge5", 0.02)
    make_result_set(tmp_path / "sensitivity" / "edge20", 0.005)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    write_json(
        holdout / "vrp_data_readiness.json",
        {
            "validation_protocol": {"split": "chronological"},
            "instances": {"qualified_instances": 20, "qualified_tasks": 100},
        },
    )
    write_json(
        holdout / "basic_vrp_summary.json",
        {"vehicle_reduction_rate": 0.25, "vrp_vehicle_count": 75},
    )
    write_json(
        holdout / "type_compatible_vrp_summary.json",
        {"type_compatible_vehicle_count": 80},
    )
    write_json(
        holdout / "time_dependent_vrp_summary.json",
        {
            "time_dependent_vehicle_count": 82,
            "links_removed_by_time_dependence": 3,
            "task_service_rate": 1.0,
            "route_type_violations": 0,
            "time_overlap_violations": 0,
            "deadhead_endpoint_violations": 0,
            "all_instances_solved": True,
        },
    )
    write_json(
        holdout / "robust_p90_vrp_summary.json",
        {
            "p90_robust_vehicle_count": 84,
            "vehicles_added_by_p90_robustness": 2,
            "links_removed_by_time_dependence": 5,
            "task_service_rate": 1.0,
            "time_overlap_violations": 0,
            "all_instances_solved": True,
        },
    )

    table, report = build_summary(tmp_path)

    assert len(table) == 3
    assert report["decision"]["proceed_with_vrp_comparison"]
    assert not report["decision"]["use_loaded_path_choice_as_main_experiment"]
    assert report["gate_checks"]["chronological_holdout_feasible"]
    assert report["main_findings"]["p90_always_increases_vehicle_count"]
