import pandas as pd
from compare_greedy_dispatch_baseline import (
    compare_with_final_vrp,
    greedy_dispatch_instance,
)


def test_greedy_local_choice_can_use_more_vehicles_than_global_assignment() -> None:
    tasks = pd.DataFrame(
        {
            "task_id": ["a", "b", "c", "d"],
            "departed_at": pd.to_datetime(
                [
                    "2023-12-01 08:00",
                    "2023-12-01 08:30",
                    "2023-12-01 10:00",
                    "2023-12-01 11:00",
                ]
            ),
        }
    )
    links = pd.DataFrame(
        {
            "from_task_id": ["a", "a", "b"],
            "to_task_id": ["c", "d", "c"],
            "deadhead_distance_km": [1.0, 2.0, 3.0],
        }
    )
    compatible_types = {task: {"van"} for task in ["a", "b", "c", "d"]}
    preferred_types = {task: "van" for task in ["a", "b", "c", "d"]}

    assignments, selected = greedy_dispatch_instance(
        tasks, compatible_types, preferred_types, links
    )

    assert assignments["vehicle_id"].nunique() == 3
    assert set(zip(selected["from_task_id"], selected["to_task_id"], strict=True)) == {
        ("a", "c")
    }


def test_greedy_reuses_only_compatible_vehicle_type() -> None:
    tasks = pd.DataFrame(
        {
            "task_id": ["a", "b"],
            "departed_at": pd.to_datetime(["2023-12-01 08:00", "2023-12-01 10:00"]),
        }
    )
    links = pd.DataFrame(
        {
            "from_task_id": ["a"],
            "to_task_id": ["b"],
            "deadhead_distance_km": [0.0],
        }
    )

    assignments, selected = greedy_dispatch_instance(
        tasks,
        {"a": {"small"}, "b": {"large"}},
        {"a": "small", "b": "large"},
        links,
    )

    assert assignments["vehicle_id"].nunique() == 2
    assert selected.empty


def test_comparison_reports_weighted_and_minimum_vehicle_advantages() -> None:
    greedy_metrics = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "greedy_vehicle_count": [10],
            "greedy_internal_deadhead_distance_km": [100.0],
        }
    )
    final_metrics = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_count": [11],
            "internal_deadhead_distance_km": [10.0],
        }
    )

    _, summary = compare_with_final_vrp(
        greedy_metrics,
        {
            "instances": 1,
            "tasks": 20,
            "greedy_vehicle_count": 10,
            "greedy_internal_deadhead_distance_km": 100.0,
            "task_service_rate": 1.0,
            "route_type_violations": 0,
            "time_overlap_violations": 0,
            "deadhead_endpoint_violations": 0,
            "missing_selected_link_paths": 0,
        },
        final_metrics,
        {
            "recommended_vehicle_count": 11,
            "recommended_deadhead_distance_km": 10.0,
            "vehicle_cost_equivalent_km": 75.0,
            "task_service_rate": 1.0,
            "route_type_violations": 0,
            "time_overlap_violations": 0,
            "deadhead_endpoint_violations": 0,
            "missing_selected_link_paths": 0,
        },
        {"p90_robust_vehicle_count": 9, "internal_deadhead_distance_km": 80.0},
    )

    assert summary["vrp_weighted_proxy_cost_reduction_rate"] > 0
    assert summary["minimum_vehicle_p90_vrp_vehicle_reduction_vs_greedy"] == 1
    assert summary["minimum_vehicle_p90_vrp_deadhead_reduction_vs_greedy_km"] == 20.0
