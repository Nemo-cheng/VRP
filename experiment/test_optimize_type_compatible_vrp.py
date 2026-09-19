import pandas as pd

from optimize_type_compatible_vrp import (
    attach_route_details,
    solve_type_compatible_path_cover,
)


def test_joint_model_keeps_one_type_for_each_vehicle_route() -> None:
    links = pd.DataFrame(
        [
            {
                "instance_id": "i1",
                "from_task_id": "a",
                "to_task_id": "b",
                "deadhead_distance_km": 1.0,
            },
            {
                "instance_id": "i1",
                "from_task_id": "b",
                "to_task_id": "c",
                "deadhead_distance_km": 1.0,
            },
        ]
    )
    compatible_types = {
        "a": {"small"},
        "b": {"small", "large"},
        "c": {"large"},
    }

    assignments, selected, diagnostics = solve_type_compatible_path_cover(
        ["a", "b", "c"], compatible_types, links
    )

    assert diagnostics["solver_success"]
    assert assignments["vehicle_id"].nunique() == 2
    assert len(selected) == 1
    assert (
        assignments.groupby("vehicle_id")["vehicle_type_name"].nunique() == 1
    ).all()


def test_route_details_are_recomputable() -> None:
    assignments = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v1"],
            "vehicle_type_name": ["small", "small"],
            "sequence": [1, 2],
            "task_id": ["a", "b"],
        }
    )
    tasks = pd.DataFrame(
        {
            "task_id": ["a", "b"],
            "origin_site_id": ["A", "C"],
            "destination_site_id": ["B", "D"],
            "departed_at": ["2023-01-01 08:00", "2023-01-01 10:00"],
            "arrived_at": ["2023-01-01 09:00", "2023-01-01 11:00"],
            "distance_km": [10.0, 20.0],
        }
    )
    selected = pd.DataFrame(
        {
            "from_task_id": ["a"],
            "to_task_id": ["b"],
            "deadhead_path": ["B>C"],
            "deadhead_distance_km": [5.0],
            "deadhead_duration_hours_p50": [0.5],
        }
    )

    route, validation = attach_route_details(assignments, selected, tasks)

    assert route.loc[0, "loaded_path"] == "A>B"
    assert route.loc[0, "next_task_id"] == "b"
    assert route.loc[0, "deadhead_to_next_path"] == "B>C"
    assert all(value == 0 for value in validation.values())
