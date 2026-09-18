import pandas as pd

from generate_unified_instances import (
    eligible_nodes,
    sample_payloads,
    valid_shanghai_pool,
)


def test_eligible_nodes_and_paired_sampling_are_reproducible() -> None:
    routes = pd.DataFrame(
        {
            "route_id": ["R1", "R2", "R3"],
            "stop_count": [10, 9, 10],
            "complete_coordinate_share": [1.0, 1.0, 0.9],
        }
    )
    nodes = pd.DataFrame(
        {
            "route_id": ["R1", "R1", "R2", "R3"],
            "stop_index": [2, 1, 1, 1],
        }
    )
    pool = pd.DataFrame(
        {
            "region": ["shanghai"] * 1000,
            "weight_kg": range(1, 1001),
            "volume_cm3": [value * 10 for value in range(1, 1001)],
            "pieces": [1] * 1000,
        }
    )

    selected = eligible_nodes(routes, nodes)
    valid_pool = valid_shanghai_pool(pool)
    first = sample_payloads(selected, valid_pool, 42)
    second = sample_payloads(selected, valid_pool, 42)

    assert list(selected["stop_index"]) == [1, 2]
    assert first[["weight_kg", "volume_cm3"]].equals(
        second[["weight_kg", "volume_cm3"]]
    )
    assert (first["volume_cm3"] == first["weight_kg"] * 10).all()
