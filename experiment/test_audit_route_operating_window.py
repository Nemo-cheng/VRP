import pandas as pd

from experiment.audit_route_operating_window import audit_schedule


def make_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i2"],
            "vehicle_id": ["v1", "v1", "v2"],
            "task_id": ["t1", "t2", "t3"],
            "departed_at": [
                "2023-12-01 06:00:00",
                "2023-12-01 18:00:00",
                "2023-12-01 05:00:00",
            ],
            "arrived_at": [
                "2023-12-01 07:00:00",
                "2023-12-01 21:00:00",
                "2023-12-01 06:00:00",
            ],
            "distance_km": [10.0, 20.0, 5.0],
            "deadhead_to_next_distance_km": [1.0, 0.0, 0.0],
        }
    )


def test_audit_requires_both_clock_and_duration_compliance() -> None:
    chains, summary = audit_schedule(make_schedule())

    assert len(chains) == 2
    assert summary["tasks_outside_clock_window"] == 2
    assert summary["chains_over_14h"] == 1
    assert summary["operating_window_compliant_chains"] == 0


def test_single_task_inside_window_is_compliant() -> None:
    frame = make_schedule().iloc[[0]].copy()

    chains, summary = audit_schedule(frame)

    assert bool(chains.iloc[0]["operating_window_compliant"])
    assert summary["operating_window_compliant_chains"] == 1
