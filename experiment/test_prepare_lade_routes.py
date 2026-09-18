import pandas as pd

from prepare_lade_routes import recover_routes


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "courier_id": [1, 1, 1, 1],
            "ds": [501, 501, 501, 501],
            "order_id": [1, 2, 3, 4],
            "region_id": [10, 10, 10, 11],
            "lng": [121.0, 121.001, 121.002, 121.003],
            "lat": [31.0, 31.0, 31.0, 31.0],
            "aoi_id": [100, 101, 102, 103],
            "aoi_type": [1, 1, 1, 1],
            "accept_time": [
                "05-01 08:00:00",
                "05-01 08:10:00",
                "05-01 10:00:00",
                "05-01 10:10:00",
            ],
            "delivery_time": [
                "05-01 08:30:00",
                "05-01 08:40:00",
                "05-01 11:00:00",
                "05-01 11:10:00",
            ],
            "accept_gps_lng": [121.0, 121.0, 121.0, 121.0],
            "accept_gps_lat": [31.0, 31.0, 31.0, 31.0],
            "delivery_gps_lng": [121.0, 121.001, 121.002, 121.003],
            "delivery_gps_lat": [31.0, 31.0, 31.0, 31.0],
        }
    )


def test_recover_routes_splits_long_gap_and_region_change() -> None:
    nodes, routes, report = recover_routes(make_frame())

    assert list(nodes["route_id"]) == [
        "R00000001",
        "R00000001",
        "R00000002",
        "R00000003",
    ]
    assert list(routes["stop_count"]) == [2, 1, 1]
    assert report["break_counts"]["long_gap"] == 1
    assert report["break_counts"]["region_change"] == 1
    assert "courier_id" not in nodes.columns
    assert "order_id" not in nodes.columns


def test_recover_routes_falls_back_from_invalid_accept_coordinate() -> None:
    frame = make_frame().iloc[:1].copy()
    frame.loc[:, "accept_gps_lng"] = 0.0
    frame.loc[:, "accept_gps_lat"] = 0.0

    _, routes, report = recover_routes(frame)

    assert routes.loc[0, "start_proxy_source"] == "first_delivery_gps_fallback"
    assert routes.loc[0, "start_proxy_lng"] == 121.0
    assert report["start_proxy_fallback_routes"] == 1
