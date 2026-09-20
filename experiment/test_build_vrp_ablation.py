import json
from pathlib import Path

from build_vrp_ablation import build_ablation


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builds_ablation_from_existing_results(tmp_path: Path) -> None:
    common = {"instances": 2, "tasks": 10, "task_service_rate": 1.0}
    write_json(
        tmp_path / "basic_vrp_summary.json",
        {
            **common,
            "independent_vehicle_count": 10,
            "vrp_vehicle_count": 7,
            "vrp_internal_deadhead_distance_km": 12.0,
        },
    )
    write_json(
        tmp_path / "type_compatible_vrp_summary.json",
        {
            **common,
            "type_compatible_vehicle_count": 8,
            "internal_deadhead_distance_km": 15.0,
        },
    )
    write_json(
        tmp_path / "time_dependent_vrp_summary.json",
        {
            **common,
            "time_dependent_vehicle_count": 8,
            "internal_deadhead_distance_km": 16.0,
        },
    )
    write_json(
        tmp_path / "robust_p90_vrp_summary.json",
        {
            **common,
            "p90_robust_vehicle_count": 9,
            "vehicles_added_by_p90_robustness": 1,
            "internal_deadhead_distance_km": 14.0,
        },
    )
    write_json(
        tmp_path / "greedy_vs_vrp_summary.json",
        {
            "instances": 2,
            "tasks": 10,
            "greedy_vehicle_count": 9,
            "greedy_internal_deadhead_distance_km": 30.0,
            "both_methods_task_service_rate": [1.0, 1.0],
            "greedy_constraint_violations": 0,
            "vrp_weighted_proxy_cost_reduction_rate": 0.1,
        },
    )
    write_json(
        tmp_path / "fixed_vs_joint_vehicle_routing_summary.json",
        {
            "vehicle_cost_75_km": {
                "fixed_historical_type_vehicle_count": 9,
                "fixed_historical_type_deadhead_distance_km": 18.0,
                "joint_weighted_proxy_cost_reduction_rate": 0.03,
                "both_solutions_feasible": True,
            }
        },
    )
    write_json(
        tmp_path / "independently_selected_p90_solution_summary.json",
        {
            **common,
            "vehicle_cost_equivalent_km": 75.0,
            "recommended_vehicle_count": 9,
            "recommended_deadhead_distance_km": 13.0,
        },
    )

    table, report = build_ablation(tmp_path)

    assert len(table) == 9
    assert set(table["comparison_family"]) == {
        "constraint_ablation",
        "decision_ablation",
    }
    full = table[table["variant"] == "p90_joint_type_balanced_vrp"].iloc[0]
    assert full["weighted_proxy_cost"] == 688.0
    assert report["key_effects"]["p90_additional_vehicle_chains_vs_p50"] == 1
    assert report["all_variants_feasible"] is True


def test_rejects_mixed_task_sets(tmp_path: Path) -> None:
    test_builds_ablation_from_existing_results(tmp_path)
    typed_path = tmp_path / "type_compatible_vrp_summary.json"
    typed = json.loads(typed_path.read_text(encoding="utf-8"))
    typed["tasks"] = 11
    write_json(typed_path, typed)

    try:
        build_ablation(tmp_path)
    except ValueError as error:
        assert "same task set" in str(error)
    else:
        raise AssertionError("mixed task sets must be rejected")
