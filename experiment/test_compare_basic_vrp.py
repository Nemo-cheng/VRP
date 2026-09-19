import pandas as pd

from compare_basic_vrp import minimum_vehicle_path_cover


def test_path_cover_minimizes_vehicles_then_deadhead_distance() -> None:
    links = pd.DataFrame(
        [
            {
                "instance_id": "i1",
                "from_task_id": "a",
                "to_task_id": "b",
                "deadhead_distance_km": 10.0,
                "deadhead_duration_hours_p50": 1.0,
                "deadhead_path": "X>Y",
            },
            {
                "instance_id": "i1",
                "from_task_id": "a",
                "to_task_id": "c",
                "deadhead_distance_km": 2.0,
                "deadhead_duration_hours_p50": 0.2,
                "deadhead_path": "X>Z",
            },
        ]
    )

    assignments, selected = minimum_vehicle_path_cover(["a", "b", "c"], links)

    assert assignments["vehicle_id"].nunique() == 2
    assert len(assignments) == 3
    assert selected.iloc[0]["to_task_id"] == "c"


def test_path_cover_keeps_isolated_tasks() -> None:
    empty_links = pd.DataFrame(
        columns=[
            "instance_id",
            "from_task_id",
            "to_task_id",
            "deadhead_distance_km",
            "deadhead_duration_hours_p50",
            "deadhead_path",
        ]
    )

    assignments, selected = minimum_vehicle_path_cover(["a", "b"], empty_links)

    assert assignments["vehicle_id"].nunique() == 2
    assert selected.empty
