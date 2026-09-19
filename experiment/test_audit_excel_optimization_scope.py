import pandas as pd

from audit_excel_optimization_scope import audit_scope


def test_audit_separates_route_evidence_from_manifest_coverage() -> None:
    orders = pd.DataFrame(
        {
            "waybill": ["w1", "w2"],
            "volume_cm3": [1000.0, 2000.0],
            "weight_kg": [1.0, 2.0],
            "pieces": [1, 1],
            "shipper_province": ["p", "p"],
            "shipper_city": ["c1", "c1"],
            "receiver_province": ["p", "p"],
            "receiver_city": ["c2", "c2"],
            "completed_at": pd.to_datetime(["2023-01-02", "2023-01-02"]),
        }
    )
    waybills = pd.DataFrame(
        {
            "waybill": ["w1", "w1", "w2"],
            "vehicle": ["v1", "v2", "v3"],
            "vehicle_type_code": [1, 1, 1],
            "vehicle_type_name": ["type", "type", "type"],
            "distance_km": [10.0, 20.0, 25.0],
            "departed_at": pd.to_datetime(
                ["2023-01-01 08:00", "2023-01-01 10:00", "2023-01-01 09:00"]
            ),
            "arrived_at": pd.to_datetime(
                ["2023-01-01 09:00", "2023-01-01 12:00", "2023-01-01 11:00"]
            ),
            "departure_site": ["A", "B", "A"],
            "arrival_site": ["B", "C", "C"],
        }
    )

    report = audit_scope(orders, waybills)

    assert report["route_evidence"]["multi_leg_waybills"] == 1
    assert report["route_evidence"]["valid_sequence_waybills"] == 2
    assert report["route_evidence"]["multi_path_pairs"] == 1
    assert report["route_evidence"]["site_continuity_rate"] == 1.0
    assert report["manifest_coverage_evidence"][
        "events_with_one_observed_order_share"
    ] == 1.0
