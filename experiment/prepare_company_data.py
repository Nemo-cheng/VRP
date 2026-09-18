#!/usr/bin/env python3
"""Prepare privacy-safe company transport data for fleet experiments."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openpyxl>=3.1",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ORDER_COLUMNS = {
    "waybill": "waybill_number（运单号）",
    "volume_cm3": "total_volume（cm³）订单体积",
    "weight_kg": "total_weight（kg）订单重量",
    "pieces": "total_pieces_parcel（总包裹数）",
    "completed_at": "complete time（妥投时间）",
}

WAYBILL_COLUMNS = {
    "waybill": "waybill_number（运单号）",
    "vehicle": "vehicle_number（车牌号）",
    "vehicle_type_code": "vehicle_type（车型种类编码）",
    "vehicle_type_name": "vehicle_type_name（车型种类名称）",
    "distance_km": "实际行驶里程（km）",
    "departed_at": "departure_time封车时间",
    "arrived_at": "arrival_time（解封车时间）",
    "departure_site": "departure_site_name（封车网点名称）",
    "arrival_site": "arrival_site_name（解封车网点名称）",
}

LEG_KEY = ["vehicle", "departed_at", "departure_site", "arrival_site"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清洗订单与运单数据，生成车队更新实验的标准化中间表。"
    )
    parser.add_argument("--input", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/company"))
    parser.add_argument(
        "--summary-dir", type=Path, default=Path("results/company_data_audit")
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: dict[str, str], sheet: str) -> None:
    missing = sorted(set(required.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"{sheet} 缺少字段: {', '.join(missing)}")


def energy_label(vehicle_type_name: object) -> str:
    name = str(vehicle_type_name)
    if any(marker in name for marker in ("新能源", "纯电", "电动")):
        return "新能源"
    return "未标明"


def load_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_excel(path, sheet_name="订单数据")
    waybills = pd.read_excel(path, sheet_name="运单数据")
    require_columns(orders, ORDER_COLUMNS, "订单数据")
    require_columns(waybills, WAYBILL_COLUMNS, "运单数据")
    return (
        orders.rename(columns={value: key for key, value in ORDER_COLUMNS.items()}),
        waybills.rename(columns={value: key for key, value in WAYBILL_COLUMNS.items()}),
    )


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    result = orders[list(ORDER_COLUMNS)].copy()
    result["completed_at"] = pd.to_datetime(result["completed_at"], errors="coerce")
    for column in ("volume_cm3", "weight_kg", "pieces"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def prepare_waybills(waybills: pd.DataFrame) -> pd.DataFrame:
    result = waybills[list(WAYBILL_COLUMNS)].copy()
    result["departed_at"] = pd.to_datetime(result["departed_at"], errors="coerce")
    result["arrived_at"] = pd.to_datetime(result["arrived_at"], errors="coerce")
    result["distance_km"] = pd.to_numeric(result["distance_km"], errors="coerce")
    result["duration_hours"] = (
        result["arrived_at"] - result["departed_at"]
    ).dt.total_seconds() / 3600
    result["average_speed_kmh"] = result["distance_km"] / result["duration_hours"]
    result["energy_observation"] = result["vehicle_type_name"].map(energy_label)
    return result


def build_physical_legs(
    orders: pd.DataFrame, waybills: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_payload = orders.set_index("waybill")[["volume_cm3", "weight_kg", "pieces"]]
    linked = waybills.join(order_payload, on="waybill", validate="many_to_one")

    consistency = linked.groupby(LEG_KEY, dropna=False).agg(
        vehicle_type_count=("vehicle_type_name", "nunique"),
        vehicle_type_code_count=("vehicle_type_code", "nunique"),
        distance_count=("distance_km", "nunique"),
        arrival_count=("arrived_at", "nunique"),
    )
    inconsistent = consistency[
        (consistency["vehicle_type_count"] > 1)
        | (consistency["vehicle_type_code_count"] > 1)
        | (consistency["distance_count"] > 1)
        | (consistency["arrival_count"] > 1)
    ].reset_index()

    legs = linked.groupby(LEG_KEY, dropna=False, as_index=False).agg(
        vehicle_type_code=("vehicle_type_code", "first"),
        vehicle_type_name=("vehicle_type_name", "first"),
        energy_observation=("energy_observation", "first"),
        arrived_at=("arrived_at", "first"),
        distance_km=("distance_km", "first"),
        waybill_count=("waybill", "nunique"),
        total_volume_cm3=("volume_cm3", "sum"),
        total_weight_kg=("weight_kg", "sum"),
        total_pieces=("pieces", "sum"),
    )
    legs["duration_hours"] = (
        legs["arrived_at"] - legs["departed_at"]
    ).dt.total_seconds() / 3600
    legs["average_speed_kmh"] = legs["distance_km"] / legs["duration_hours"]
    legs["invalid_time"] = legs["duration_hours"].isna() | (legs["duration_hours"] <= 0)
    legs["invalid_distance"] = legs["distance_km"].isna() | (legs["distance_km"] <= 0)
    legs["duration_over_72h"] = legs["duration_hours"] > 72
    legs["speed_over_120kmh"] = legs["average_speed_kmh"] > 120
    conflict_keys = pd.MultiIndex.from_frame(inconsistent[LEG_KEY])
    leg_keys = pd.MultiIndex.from_frame(legs[LEG_KEY])
    legs["vehicle_type_conflict"] = leg_keys.isin(conflict_keys)
    legs["analysis_eligible"] = ~(
        legs["invalid_time"]
        | legs["invalid_distance"]
        | legs["duration_over_72h"]
        | legs["speed_over_120kmh"]
        | legs["vehicle_type_conflict"]
    )
    vehicle_codes, _ = pd.factorize(legs["vehicle"], sort=True)
    legs["vehicle_id"] = pd.Series(vehicle_codes, index=legs.index).map(
        lambda value: f"vehicle_{value + 1:05d}"
    )
    legs = legs.drop(columns="vehicle")
    return legs, inconsistent


def build_vehicle_type_summary(legs: pd.DataFrame) -> pd.DataFrame:
    eligible = legs[legs["analysis_eligible"]]
    return (
        eligible.groupby(
            ["vehicle_type_code", "vehicle_type_name", "energy_observation"],
            dropna=False,
            as_index=False,
        )
        .agg(
            physical_legs=("vehicle_id", "size"),
            vehicles=("vehicle_id", "nunique"),
            eligible_legs=("analysis_eligible", "sum"),
            total_distance_km=("distance_km", "sum"),
            median_distance_km=("distance_km", "median"),
            median_duration_hours=("duration_hours", "median"),
            total_weight_kg=("total_weight_kg", "sum"),
            total_volume_cm3=("total_volume_cm3", "sum"),
        )
        .sort_values(["physical_legs", "vehicles"], ascending=False)
    )


def build_audit(
    source_path: Path,
    orders: pd.DataFrame,
    waybills: pd.DataFrame,
    legs: pd.DataFrame,
    inconsistent: pd.DataFrame,
) -> dict[str, object]:
    matched = orders["waybill"].isin(waybills["waybill"])
    return {
        "source_file": source_path.name,
        "orders": {
            "rows": len(orders),
            "unique_waybills": int(orders["waybill"].nunique()),
            "matched_to_waybill_data": int(matched.sum()),
            "completed_at_min": str(orders["completed_at"].min()),
            "completed_at_max": str(orders["completed_at"].max()),
        },
        "waybill_data": {
            "rows": len(waybills),
            "unique_waybills": int(waybills["waybill"].nunique()),
            "unique_vehicles": int(waybills["vehicle"].nunique()),
            "departed_at_min": str(waybills["departed_at"].min()),
            "arrived_at_max": str(waybills["arrived_at"].max()),
        },
        "physical_legs": {
            "rows": len(legs),
            "analysis_eligible": int(legs["analysis_eligible"].sum()),
            "invalid_time": int(legs["invalid_time"].sum()),
            "invalid_distance": int(legs["invalid_distance"].sum()),
            "duration_over_72h": int(legs["duration_over_72h"].sum()),
            "speed_over_120kmh": int(legs["speed_over_120kmh"].sum()),
            "vehicle_type_conflicts": len(inconsistent),
            "new_energy_eligible_legs": int(
                (
                    (legs["energy_observation"] == "新能源")
                    & legs["analysis_eligible"]
                ).sum()
            ),
            "new_energy_vehicles": int(
                legs.loc[
                    (legs["energy_observation"] == "新能源")
                    & legs["analysis_eligible"],
                    "vehicle_id",
                ].nunique()
            ),
        },
        "interpretation": {
            "unmarked_energy_type": "车型名称未标明能源类型，不能直接认定为燃油车。",
            "analysis_eligible_rule": "时长和里程为正，时长不超过72小时，平均速度不超过120km/h，且运输事件内部车型一致。",
        },
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件: {args.input}")

    orders_raw, waybills_raw = load_source(args.input)
    orders = prepare_orders(orders_raw)
    waybills = prepare_waybills(waybills_raw)
    legs, inconsistent = build_physical_legs(orders, waybills)
    type_summary = build_vehicle_type_summary(legs)
    audit = build_audit(args.input, orders, waybills, legs, inconsistent)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    legs.to_csv(args.output_dir / "physical_legs.csv", index=False, encoding="utf-8-sig")
    type_summary.to_csv(
        args.summary_dir / "vehicle_type_summary.csv", index=False, encoding="utf-8-sig"
    )
    (args.summary_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
