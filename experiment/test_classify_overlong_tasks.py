import pandas as pd

from experiment.classify_overlong_tasks import classify_overlong_tasks


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": ["long", "waiting", "normal"],
            "instance_id": ["i1", "i1", "i1"],
            "vehicle_id": ["v1", "v2", "v3"],
            "origin_site_id": ["a", "c", "e"],
            "destination_site_id": ["b", "d", "f"],
            "departed_at": ["2023-12-01 06:00"] * 3,
            "arrived_at": [
                "2023-12-02 02:00",
                "2023-12-02 02:00",
                "2023-12-01 08:00",
            ],
            "distance_km": [1000.0, 20.0, 30.0],
        }
    )


def _training_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_site_id": ["a", "c", "e"],
            "destination_site_id": ["b", "d", "f"],
            "observations": [20, 20, 20],
            "distance_km_p50": [1000.0, 20.0, 30.0],
            "duration_hours_p50": [17.0, 1.0, 2.0],
            "duration_hours_p90": [20.0, 2.0, 3.0],
            "high_confidence": [True, True, True],
        }
    )


def test_only_training_supported_long_haul_enters_multiday_analysis() -> None:
    detail, summary = classify_overlong_tasks(_schedule(), _training_edges())

    assert set(detail["task_id"]) == {"long", "waiting"}
    long_task = detail.set_index("task_id").loc["long"]
    assert long_task["classification"] == "verified_multiday_long_haul"
    assert bool(long_task["included_in_multiday_analysis"])
    assert long_task["planning_duration_hours"] == 20.0
    assert long_task["required_operating_days"] == 2
    assert long_task["maximum_daily_distance_km"] == 700.0
    assert summary["verified_multiday_long_haul_tasks"] == 1


def test_waiting_anomaly_is_deferred_without_replacing_its_value() -> None:
    detail, summary = classify_overlong_tasks(_schedule(), _training_edges())

    waiting = detail.set_index("task_id").loc["waiting"]
    assert waiting["classification"] == "deferred_time_anomaly"
    assert not bool(waiting["included_in_multiday_analysis"])
    assert pd.isna(waiting["planning_duration_hours"])
    assert pd.isna(waiting["required_operating_days"])
    assert pd.isna(waiting["maximum_daily_distance_km"])
    assert summary["deferred_time_anomaly_tasks"] == 1


def test_low_confidence_lane_is_not_promoted_to_verified_long_haul() -> None:
    edges = _training_edges()
    edges.loc[edges["origin_site_id"].eq("a"), "high_confidence"] = False

    detail, summary = classify_overlong_tasks(_schedule(), edges)

    long_task = detail.set_index("task_id").loc["long"]
    assert long_task["classification"] == "deferred_time_anomaly"
    assert summary["verified_multiday_long_haul_tasks"] == 0
