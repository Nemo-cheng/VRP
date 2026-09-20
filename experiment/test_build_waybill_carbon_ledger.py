import pandas as pd
import pytest

from experiment.build_waybill_carbon_ledger import add_allocation_shares


def test_positive_weights_allocate_by_weight() -> None:
    frame = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v1"],
            "departed_at": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "departure_site": ["a", "a"],
            "arrival_site": ["b", "b"],
            "waybill": ["w1", "w2"],
            "weight_kg": [10.0, 30.0],
        }
    )

    result = add_allocation_shares(frame)

    assert result["allocation_method"].tolist() == ["weight_share", "weight_share"]
    assert result["allocation_share"].tolist() == pytest.approx([0.25, 0.75])


def test_zero_weights_allocate_equally() -> None:
    frame = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v1"],
            "departed_at": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "departure_site": ["a", "a"],
            "arrival_site": ["b", "b"],
            "waybill": ["w1", "w2"],
            "weight_kg": [0.0, None],
        }
    )

    result = add_allocation_shares(frame)

    assert result["allocation_method"].tolist() == ["equal_share", "equal_share"]
    assert result["allocation_share"].tolist() == pytest.approx([0.5, 0.5])
