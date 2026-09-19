import pandas as pd

from optimize_type_compatible_vrp import solve_type_compatible_path_cover


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
