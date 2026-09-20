import pandas as pd

from experiment.audit_longhaul_relay_evidence import (
    audit_relay_paths,
    shortest_relay_path,
)


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_site_id": ["A", "A", "B"],
            "destination_site_id": ["D", "B", "D"],
            "distance_km_p50": [700.0, 300.0, 300.0],
            "duration_hours_p90": [20.0, 5.0, 5.0],
            "observations": [20, 20, 20],
            "high_confidence": [True, True, True],
        }
    )


def test_shortest_relay_path_excludes_direct_edge() -> None:
    from experiment.audit_longhaul_relay_evidence import build_graph

    path = shortest_relay_path(build_graph(_edges()), "A", "D")

    assert path is not None
    assert path["path"] == "A>B>D"
    assert path["maximum_segment_distance_km"] == 300.0


def test_verified_relay_requires_each_segment_to_meet_range_and_time() -> None:
    targets = pd.DataFrame(
        {
            "task_id": ["t1"],
            "instance_id": ["i1"],
            "vehicle_id": ["v1"],
            "origin_site_id": ["A"],
            "destination_site_id": ["D"],
            "distance_km": [700.0],
        }
    )

    feasible = audit_relay_paths(targets, _edges(), maximum_range_km=405.0)
    infeasible = audit_relay_paths(targets, _edges(), maximum_range_km=250.0)

    assert bool(feasible.iloc[0]["verified_relay_feasible"])
    assert feasible.iloc[0]["relay_evidence_status"] == "verified_relay_feasible"
    assert not bool(infeasible.iloc[0]["verified_relay_feasible"])
    assert infeasible.iloc[0]["relay_evidence_status"] == (
        "topology_only_not_segment_feasible"
    )
