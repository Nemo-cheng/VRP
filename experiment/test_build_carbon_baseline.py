from pathlib import Path

import pandas as pd
import pytest

from experiment.build_carbon_baseline import (
    classify_consumption_parameter,
    fuel_emission_factor_kg_per_kg,
    load_vehicle_parameters,
)


@pytest.mark.parametrize(
    ("fuel", "payload_t", "expected_parameter", "expected_status"),
    [
        ("汽油", 2.0, "freight_gasoline_le_2t_consumption", "covered"),
        ("柴油", 2.0, None, "diesel_le_2t_no_official_default"),
        ("柴油", 2.01, "freight_diesel_gt_2_le_4t_consumption", "covered"),
        ("柴油", 4.0, "freight_diesel_gt_2_le_4t_consumption", "covered"),
        ("柴油", 4.01, "freight_diesel_gt_4_lt_8t_consumption", "covered"),
        ("柴油", 8.0, "freight_diesel_ge_8_lt_20t_consumption", "covered"),
        ("柴油", 20.0, "freight_diesel_ge_20t_consumption", "covered"),
        ("纯电动", 2.0, None, "electricity_consumption_missing"),
    ],
)
def test_consumption_parameter_boundaries(
    fuel: str,
    payload_t: float,
    expected_parameter: str | None,
    expected_status: str,
) -> None:
    assert classify_consumption_parameter(fuel, payload_t) == (
        expected_parameter,
        expected_status,
    )


def test_diesel_fuel_emission_factor_uses_official_formula() -> None:
    result = fuel_emission_factor_kg_per_kg(43.33, 0.0202, 0.98)

    assert result == pytest.approx(3.1451224933)


def test_vehicle_type_must_be_unique(tmp_path: Path) -> None:
    path = tmp_path / "vehicles.xlsx"
    frame = pd.DataFrame(
        {
            "车型种类": ["重复车型", "重复车型"],
            "燃油类型": ["柴油", "柴油"],
            "车辆载重": [2.0, 2.0],
        }
    )
    frame.to_excel(path, sheet_name="Sheet2", index=False)

    with pytest.raises(ValueError, match="车型种类必须唯一"):
        load_vehicle_parameters(path)
