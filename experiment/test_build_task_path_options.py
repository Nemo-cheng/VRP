import pandas as pd

from build_task_path_options import build_path_options


def test_keeps_historical_path_and_adds_feasible_network_path() -> None:
    tasks = pd.DataFrame(
        {
            "task_id": ["t1", "e_ab", "e_bc", "e_ac"],
            "origin_site_id": ["A", "A", "B", "A"],
            "destination_site_id": ["C", "B", "C", "C"],
            "network_covered": [True, True, True, True],
            "departed_at": pd.to_datetime(["2023-01-01 08:00"] * 4),
            "arrived_at": pd.to_datetime(["2023-01-01 11:00"] * 4),
            "distance_km": [30.0, 10.0, 10.0, 30.0],
            "duration_hours": [3.0, 1.0, 1.0, 3.0],
            "vehicle_type_name": ["small", "small", "small", "small"],
        }
    )
    edges = pd.DataFrame(
        {
            "origin_site_id": ["A", "B", "A"],
            "destination_site_id": ["B", "C", "C"],
            "distance_km_p50": [10.0, 10.0, 30.0],
            "duration_hours_p50": [1.0, 1.0, 3.0],
            "high_confidence": [True, True, True],
        }
    )
    periods = pd.DataFrame(
        columns=[
            "origin_site_id",
            "destination_site_id",
            "departure_period",
            "duration_hours_p50",
            "period_estimate_available",
        ]
    )

    options = build_path_options(tasks, edges, periods, paths_per_objective=3)
    task_options = options[options["task_id"] == "t1"]

    assert "historical_observed" in set(task_options["path_source"])
    assert "A>B>C" in set(task_options["path"])
    assert task_options["compatible_vehicle_type_count"].ge(1).all()
