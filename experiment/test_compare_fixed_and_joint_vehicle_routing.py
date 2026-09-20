import pandas as pd

from compare_fixed_and_joint_vehicle_routing import (
    compare_scenarios,
    fixed_historical_task_types,
)


def test_fixed_types_keep_each_tasks_observed_vehicle_type() -> None:
    tasks = pd.DataFrame(
        {"task_id": ["a", "b"], "vehicle_type_name": ["small", "large"]}
    )

    assert fixed_historical_task_types(tasks) == {
        "a": {"small"},
        "b": {"large"},
    }


def test_comparison_isolates_joint_type_route_effect() -> None:
    fixed_metrics = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_count": [3],
            "internal_deadhead_distance_km": [20.0],
        }
    )
    joint_metrics = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_count": [2],
            "internal_deadhead_distance_km": [15.0],
        }
    )
    common = {
        "tasks": 4,
        "task_service_rate": 1.0,
        "route_type_violations": 0,
        "time_overlap_violations": 0,
    }
    comparison, summary = compare_scenarios(
        fixed_metrics,
        {
            **common,
            "vehicle_count": 3,
            "internal_deadhead_distance_km": 20.0,
            "weighted_proxy_cost": 245.0,
        },
        joint_metrics,
        {
            **common,
            "vehicle_count": 2,
            "internal_deadhead_distance_km": 15.0,
            "weighted_proxy_cost": 165.0,
        },
        "vehicle_cost_75_km",
    )

    assert comparison.loc[0, "joint_vehicle_reduction"] == 1
    assert comparison.loc[0, "joint_deadhead_reduction_km"] == 5.0
    assert summary["joint_weighted_proxy_cost_reduction_rate"] > 0
    assert summary["both_solutions_feasible"]
