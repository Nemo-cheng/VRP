import pandas as pd
import pytest

from experiment.reconcile_vrp_carbon_scope import reconcile_scope


def test_scope_reconciliation_uses_matched_subset() -> None:
    baseline = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v2"],
            "departed_at": ["2023-12-01", "2023-12-02"],
            "departure_site": ["a", "b"],
            "arrival_site": ["b", "c"],
            "distance_km": [100.0, 300.0],
            "emissions_kgco2_min": [10.0, 60.0],
            "emissions_kgco2_max": [20.0, 80.0],
        }
    )
    events = baseline[
        [
            "vehicle_id",
            "departed_at",
            "departure_site",
            "arrival_site",
            "distance_km",
        ]
    ].copy()
    events["event_id"] = ["t1", "t2"]
    schedule = pd.DataFrame({"task_id": ["t1"]})

    result = reconcile_scope(baseline, events, schedule)

    assert result["task_scope_share"] == pytest.approx(0.5)
    assert result["distance_scope_share"] == pytest.approx(0.25)
    assert result["emission_scope_share"]["minimum_case"] == pytest.approx(
        10 / 70
    )
    assert result["extrapolation_allowed"] is False
