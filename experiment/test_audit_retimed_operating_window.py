import pandas as pd

from experiment.audit_retimed_operating_window import audit_retimed_schedule


def test_historical_waiting_is_removed_but_travel_time_is_preserved() -> None:
    frame = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "vehicle_id": ["v1", "v1"],
            "task_id": ["t1", "t2"],
            "departed_at": ["2023-12-01 01:00", "2023-12-01 20:00"],
            "arrived_at": ["2023-12-01 03:00", "2023-12-01 22:00"],
            "distance_km": [10.0, 10.0],
            "deadhead_to_next_distance_km": [1.0, 0.0],
            "deadhead_to_next_duration_hours": [1.0, 0.0],
        }
    )

    chains, summary = audit_retimed_schedule(frame)

    assert chains.iloc[0]["retimed_shift_hours"] == 5.0
    assert bool(chains.iloc[0]["retimed_operating_window_compliant"])
    assert summary["retimed_compliant_chains"] == 1


def test_overlong_task_remains_infeasible_after_retiming() -> None:
    frame = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_id": ["v1"],
            "task_id": ["t1"],
            "departed_at": ["2023-12-01 01:00"],
            "arrived_at": ["2023-12-01 16:00"],
            "distance_km": [100.0],
            "deadhead_to_next_distance_km": [0.0],
            "deadhead_to_next_duration_hours": [0.0],
        }
    )

    chains, summary = audit_retimed_schedule(frame)

    assert not bool(chains.iloc[0]["retimed_operating_window_compliant"])
    assert summary["individual_tasks_over_14h"] == 1
