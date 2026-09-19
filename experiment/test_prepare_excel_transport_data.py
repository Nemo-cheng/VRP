import pandas as pd

from prepare_excel_transport_data import build_lane_daily, build_transport_events


def test_build_transport_events_uses_receiver_city_and_anonymizes_lane() -> None:
    orders = pd.DataFrame(
        {
            "waybill": ["w1", "w2"],
            "volume_cm3": [1000.0, 2000.0],
            "weight_kg": [1.0, 2.0],
            "pieces": [1.0, 1.0],
            "shipper_province": ["上海", "上海"],
            "shipper_city": ["上海市", "上海市"],
            "receiver_province": ["江苏", "江苏"],
            "receiver_city": ["苏州市", "苏州市"],
            "completed_at": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        }
    )
    waybills = pd.DataFrame(
        {
            "waybill": ["w1", "w2"],
            "vehicle": ["plate1", "plate1"],
            "vehicle_type_code": [4, 4],
            "vehicle_type_name": ["4.2米厢式货车", "4.2米厢式货车"],
            "distance_km": [100.0, 100.0],
            "departed_at": pd.to_datetime(["2023-01-01 08:00", "2023-01-01 08:00"]),
            "arrived_at": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:00"]),
            "departure_site": ["上海站", "上海站"],
            "arrival_site": ["苏州站", "苏州站"],
        }
    )

    events, conflicts = build_transport_events(orders, waybills)
    daily = build_lane_daily(events)

    assert len(conflicts) == 0
    assert events.loc[0, "destination_city"] == "苏州市"
    assert events.loc[0, "lane_id"].startswith("site_")
    assert daily.loc[0, "total_weight_kg"] == 3.0
    assert daily.loc[0, "observed_trips"] == 1
