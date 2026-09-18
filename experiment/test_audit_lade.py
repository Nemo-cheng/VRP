import pandas as pd

from audit_lade import audit_route_sequence, haversine_km


def test_haversine_returns_expected_equatorial_distance() -> None:
    distance = haversine_km(
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([1.0]),
        pd.Series([0.0]),
    )
    assert 111.1 < distance.iloc[0] < 111.3


def test_route_audit_sorts_events_and_flags_duplicate_and_jump() -> None:
    frame = pd.DataFrame(
        {
            "courier_id": [1, 1, 1],
            "ds": [501, 501, 501],
            "order_id": [10, 10, 11],
            "delivery_dt": pd.to_datetime(
                ["2022-05-01 09:10", "2022-05-01 09:00", "2022-05-01 09:10"]
            ),
            "lng": [121.0, 121.0, 121.0],
            "lat": [31.0, 31.0, 31.0],
            "delivery_gps_lng": [121.0, 121.0, 121.2],
            "delivery_gps_lat": [31.0, 31.0, 31.0],
        }
    )

    result = audit_route_sequence(frame)

    assert result["duplicate_orders_within_courier_day"] == 2
    assert result["duplicate_order_groups_within_courier_day"] == 1
    assert result["recovered_sequence_negative_time_gaps"] == 0
    assert result["equal_time_pairs_over_100m"] == 1
    assert result["adjacent_distance_over_10km"] == 1
    assert result["adjacent_pairs_excluded_for_coordinates"] == 0


def test_route_audit_excludes_out_of_bounds_coordinates() -> None:
    frame = pd.DataFrame(
        {
            "courier_id": [1, 1],
            "ds": [501, 501],
            "order_id": [10, 11],
            "delivery_dt": pd.to_datetime(
                ["2022-05-01 09:00", "2022-05-01 09:10"]
            ),
            "lng": [121.0, 121.0],
            "lat": [31.0, 31.0],
            "delivery_gps_lng": [121.0, 0.0],
            "delivery_gps_lat": [31.0, 0.0],
        }
    )

    result = audit_route_sequence(frame)

    assert result["adjacent_pairs"] == 1
    assert result["adjacent_pairs_with_valid_coordinates"] == 0
    assert result["adjacent_pairs_excluded_for_coordinates"] == 1
    assert result["adjacent_distance_km"]["max"] is None
