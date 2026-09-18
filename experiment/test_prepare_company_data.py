from __future__ import annotations

import pandas as pd
from prepare_company_data import build_physical_legs, build_vehicle_type_summary


def test_physical_leg_aggregation_and_conflict_filtering() -> None:
    orders = pd.DataFrame(
        {
            "waybill": ["A", "B", "C"],
            "volume_cm3": [1000, 2000, 3000],
            "weight_kg": [1.0, 2.0, 3.0],
            "pieces": [1, 1, 1],
            "completed_at": pd.to_datetime(["2023-01-02"] * 3),
        }
    )
    waybills = pd.DataFrame(
        {
            "waybill": ["A", "B", "C"],
            "vehicle": ["plate-1", "plate-1", "plate-2"],
            "vehicle_type_code": [4, 4, 84],
            "vehicle_type_name": ["4.2米厢式货车", "9.6米厢式货车", "(新能源)4.2米厢式货车"],
            "distance_km": [20.0, 20.0, 30.0],
            "departed_at": pd.to_datetime(["2023-01-01 08:00"] * 3),
            "arrived_at": pd.to_datetime(
                ["2023-01-01 09:00", "2023-01-01 09:00", "2023-01-01 10:00"]
            ),
            "departure_site": ["site-a", "site-a", "site-c"],
            "arrival_site": ["site-b", "site-b", "site-d"],
            "duration_hours": [1.0, 1.0, 2.0],
            "average_speed_kmh": [20.0, 20.0, 15.0],
            "energy_observation": ["未标明", "未标明", "新能源"],
        }
    )

    legs, inconsistent = build_physical_legs(orders, waybills)
    summary = build_vehicle_type_summary(legs)

    assert len(legs) == 2
    assert len(inconsistent) == 1

    conflict = legs.loc[legs["vehicle_type_conflict"]].iloc[0]
    assert conflict["waybill_count"] == 2
    assert conflict["total_weight_kg"] == 3.0
    assert conflict["total_volume_cm3"] == 3000
    assert not bool(conflict["analysis_eligible"])

    assert summary["physical_legs"].sum() == 1
    assert summary.iloc[0]["energy_observation"] == "新能源"
    assert set(legs["vehicle_id"]) == {"vehicle_00001", "vehicle_00002"}
    assert "plate-1" not in legs.to_csv(index=False)
