import pandas as pd

from analyze_route_scale import analyze_scale


def test_scale_analysis_filters_routes_and_adds_closed_proxy_distance() -> None:
    routes = pd.DataFrame(
        {
            "route_id": ["R1", "R2"],
            "stop_count": [10, 5],
            "complete_coordinate_share": [1.0, 1.0],
            "work_span_minutes": [60.0, 30.0],
            "straight_line_distance_km": [1.0, 0.5],
            "max_adjacent_distance_km": [0.2, 0.1],
            "max_service_radius_km": [0.3, 0.2],
            "start_proxy_lng": [121.0, 121.0],
            "start_proxy_lat": [31.0, 31.0],
        }
    )
    nodes = pd.DataFrame(
        {
            "route_id": ["R1", "R1", "R2"],
            "stop_index": [1, 2, 1],
            "delivery_gps_lng": [121.0, 121.01, 121.0],
            "delivery_gps_lat": [31.0, 31.0, 31.0],
        }
    )

    summary, assessment, report = analyze_scale(routes, nodes)

    assert report["eligible_routes"] == 1
    assert report["eligible_orders"] == 10
    assert len(summary) == 16
    assert set(assessment["vehicle_class"]) == {
        "cargo_two_wheeler",
        "cargo_tricycle",
        "microvan",
        "4.2m_box_truck",
    }
    closed = summary.loc[
        (summary["population"] == "eligible_at_least_10_stops")
        & (summary["metric"] == "closed_proxy_distance_km"),
        "p50",
    ].iloc[0]
    assert closed > 1.0
