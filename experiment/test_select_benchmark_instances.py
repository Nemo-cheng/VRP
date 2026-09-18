import pandas as pd

from select_benchmark_instances import select_routes


def test_select_routes_returns_three_cases_and_one_boundary_case() -> None:
    rows = []
    loads = []
    for scale in [20, 30, 50]:
        for index in range(5):
            route_id = f"R{scale}_{index}"
            rows.append(
                {
                    "route_id": route_id,
                    "stop_count": scale,
                    "complete_coordinate_share": 1.0,
                    "straight_line_distance_km": 5.0 + index,
                    "work_span_minutes": 100.0 + index,
                    "max_service_radius_km": 1.0 + index / 10,
                    "max_adjacent_distance_km": 1.0,
                }
            )
            loads.append(
                {
                    "route_id": route_id,
                    "total_weight_kg": 20.0 + index,
                    "total_volume_m3": 0.2 + index / 100,
                }
            )
    rows.append(
        {
            "route_id": "R100",
            "stop_count": 100,
            "complete_coordinate_share": 1.0,
            "straight_line_distance_km": 40.0,
            "work_span_minutes": 800.0,
            "max_service_radius_km": 5.0,
            "max_adjacent_distance_km": 5.0,
        }
    )
    loads.append(
        {
            "route_id": "R100",
            "total_weight_kg": 100.0,
            "total_volume_m3": 1.0,
        }
    )

    selected = select_routes(pd.DataFrame(rows), pd.DataFrame(loads))

    assert len(selected) == 10
    assert selected[selected["stop_count"] == 100]["case_type"].tolist() == [
        "representative"
    ]
    assert set(selected[selected["stop_count"] == 20]["case_type"]) == {
        "representative",
        "distance_stress",
        "payload_stress",
    }
