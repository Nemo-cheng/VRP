import pandas as pd

from export_representative_route import select_representative_route, summarize_route


def test_exports_longest_route_with_relative_time() -> None:
    schedule = pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i2"],
            "vehicle_id": ["v1", "v1", "v2"],
            "sequence": [1, 2, 1],
            "vehicle_type_name": ["small", "small", "large"],
            "origin_site_id": ["A", "C", "X"],
            "destination_site_id": ["B", "D", "Y"],
            "loaded_path": ["A>B", "C>D", "X>Y"],
            "distance_km": [10.0, 20.0, 30.0],
            "departed_at": [
                "2023-01-01 08:00",
                "2023-01-01 10:00",
                "2023-01-01 09:00",
            ],
            "arrived_at": [
                "2023-01-01 09:00",
                "2023-01-01 11:00",
                "2023-01-01 10:00",
            ],
            "next_task_id": ["b", None, None],
            "deadhead_to_next_path": ["B>C", None, None],
            "deadhead_to_next_distance_km": [5.0, 0.0, 0.0],
            "deadhead_to_next_duration_hours": [0.5, 0.0, 0.0],
        }
    )

    route = select_representative_route(schedule)
    summary = summarize_route(route)

    assert len(route) == 2
    assert route.loc[0, "departure_offset_hours"] == 0.0
    assert route.loc[0, "waiting_after_deadhead_hours"] == 0.5
    assert summary["loaded_distance_km"] == 30.0
    assert summary["all_deadhead_links_time_feasible"]
