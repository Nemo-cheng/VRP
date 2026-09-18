import numpy as np
import pandas as pd

from prepare_road_network import gcj02_to_wgs84, route_points


def test_gcj02_to_wgs84_moves_shanghai_point_by_expected_magnitude() -> None:
    lng, lat = gcj02_to_wgs84(pd.Series([121.4737]), pd.Series([31.2304]))

    assert lng[0] < 121.4737
    assert 0.003 < 121.4737 - lng[0] < 0.01
    assert 0.001 < abs(31.2304 - lat[0]) < 0.01


def test_route_points_adds_start_proxy_and_customers() -> None:
    frame = pd.DataFrame(
        {
            "stop_index": [2, 1],
            "accept_gps_lng": [121.1, 121.0],
            "accept_gps_lat": [31.1, 31.0],
            "delivery_gps_lng": [121.2, 121.1],
            "delivery_gps_lat": [31.2, 31.1],
        }
    )

    points = route_points(frame)

    assert points["node_index"].tolist() == [0, 1, 2]
    assert points["node_type"].tolist() == ["start_proxy", "customer", "customer"]
    assert points.loc[0, "lng"] == 121.0
