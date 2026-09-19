import pandas as pd

from build_vrp_holdout_data import build_holdout_layer


def test_holdout_network_and_types_use_training_period_only() -> None:
    events = pd.DataFrame(
        {
            "analysis_eligible": [True, True, True],
            "event_id": ["train1", "train2", "test1"],
            "origin_site_id": ["A", "A", "A"],
            "destination_site_id": ["B", "B", "B"],
            "service_date": ["2023-09-01", "2023-09-02", "2023-10-01"],
            "departed_at": pd.to_datetime(
                ["2023-09-01 08:00", "2023-09-02 08:00", "2023-10-01 08:00"]
            ),
            "arrived_at": pd.to_datetime(
                ["2023-09-01 09:00", "2023-09-02 09:00", "2023-10-01 09:00"]
            ),
            "distance_km": [10.0, 10.0, 100.0],
            "duration_hours": [1.0, 1.0, 1.0],
            "vehicle_type_name": ["train_type", "train_type", "test_only_type"],
            "vehicle": ["v1", "v2", "v3"],
            "order_count": [1, 1, 1],
        }
    )

    layer = build_holdout_layer(
        events,
        pd.Timestamp("2023-10-01"),
        min_edge_observations=2,
        min_period_observations=2,
        min_instance_tasks=1,
        required_instances=1,
    )

    assert layer["edges"].loc[0, "distance_km_p50"] == 10.0
    assert set(layer["lane_vehicle_types"]["vehicle_type_name"]) == {"train_type"}
    assert set(layer["test_tasks"]["task_id"]) == {"test1"}
    assert layer["report"]["validation_protocol"][
        "network_and_vehicle_type_evidence_from_train_only"
    ]
