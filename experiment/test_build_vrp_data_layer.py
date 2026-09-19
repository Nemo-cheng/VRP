import networkx as nx
import pandas as pd

from build_vrp_data_layer import (
    assign_components,
    build_instances,
    build_network_tables,
    build_tasks,
)


def sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "analysis_eligible": [True, True, True],
            "event_id": ["e1", "e2", "e3"],
            "origin_site_id": ["A", "B", "A"],
            "destination_site_id": ["B", "C", "B"],
            "service_date": ["2023-01-01"] * 3,
            "departed_at": pd.to_datetime(
                ["2023-01-01 08:00", "2023-01-01 10:00", "2023-01-01 13:00"]
            ),
            "arrived_at": pd.to_datetime(
                ["2023-01-01 09:00", "2023-01-01 11:00", "2023-01-01 14:00"]
            ),
            "distance_km": [10.0, 10.0, 12.0],
            "duration_hours": [1.0, 1.0, 1.0],
            "vehicle_type_name": ["type1", "type1", "type2"],
            "vehicle": ["v1", "v2", "v3"],
            "order_count": [1, 1, 1],
        }
    )


def test_builds_high_confidence_edges_and_periods() -> None:
    tasks = build_tasks(sample_events())
    edges, periods = build_network_tables(tasks, 2, 2)

    edge_ab = edges.query("origin_site_id == 'A' and destination_site_id == 'B'").iloc[0]
    assert edge_ab["observations"] == 2
    assert bool(edge_ab["high_confidence"])
    assert periods["period_estimate_available"].sum() == 0


def test_builds_time_feasible_task_links() -> None:
    tasks = build_tasks(sample_events())
    graph = nx.DiGraph()
    graph.add_edge("A", "B", duration_hours=1.0, distance_km=10.0)
    graph.add_edge("B", "C", duration_hours=1.0, distance_km=10.0)
    tasks = assign_components(tasks, graph)

    instances, links = build_instances(tasks, graph, min_instance_tasks=2)

    assert len(instances) == 1
    assert instances.loc[0, "linkable_task_count"] == 2
    assert bool(instances.loc[0, "qualifies_for_vrp"])
    assert ((links["from_task_id"] == "e1") & (links["to_task_id"] == "e2")).any()
