import pandas as pd

from optimize_time_dependent_vrp import (
    build_time_dependent_links,
    evaluate_time_dependent_path,
    period_for_timestamp,
)


def test_period_boundaries() -> None:
    assert period_for_timestamp(pd.Timestamp("2023-01-01 05:59")) == "night"
    assert period_for_timestamp(pd.Timestamp("2023-01-01 06:00")) == "morning_peak"
    assert period_for_timestamp(pd.Timestamp("2023-01-01 20:00")) == "evening"


def test_path_uses_period_value_then_falls_back_to_edge_value() -> None:
    duration, reliable, total = evaluate_time_dependent_path(
        ["A", "B", "C"],
        pd.Timestamp("2023-01-01 08:00"),
        {("A", "B"): 2.0, ("B", "C"): 3.0},
        {("A", "B", "morning_peak"): 1.0},
    )

    assert duration == 4.0
    assert reliable == 1
    assert total == 2


def test_time_dependent_evaluation_removes_late_link() -> None:
    links = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "from_task_id": ["a"],
            "to_task_id": ["b"],
            "deadhead_duration_hours_p50": [1.0],
            "deadhead_distance_km": [10.0],
            "deadhead_path": ["A>B"],
            "available_slack_hours": [1.0],
        }
    )
    tasks = pd.DataFrame(
        {
            "task_id": ["a", "b"],
            "arrived_at": ["2023-01-01 08:00", "2023-01-01 12:00"],
            "departed_at": ["2023-01-01 07:00", "2023-01-01 09:30"],
        }
    )
    edges = pd.DataFrame(
        {
            "origin_site_id": ["A"],
            "destination_site_id": ["B"],
            "duration_hours_p50": [1.0],
            "high_confidence": [True],
        }
    )
    periods = pd.DataFrame(
        {
            "origin_site_id": ["A"],
            "destination_site_id": ["B"],
            "departure_period": ["morning_peak"],
            "duration_hours_p50": [2.0],
            "period_estimate_available": [True],
        }
    )

    time_links, diagnostics = build_time_dependent_links(
        links, tasks, edges, periods
    )

    assert time_links.empty
    assert diagnostics["links_removed_by_time_dependence"] == 1
