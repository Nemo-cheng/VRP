#!/usr/bin/env python3
"""Build anonymized lane-day demand tables from 订单数据.xlsx only."""

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
    "shipper_province": "shipper_province_name（发货地省名称）",
    "shipper_city": "shipper_city_name（发货地市名称）",
    "receiver_province": "receiver_province_name（收货地省名称）",
    "receiver_city": "receiver_city_name（收货地省名称）",
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
    parser = argparse.ArgumentParser(description="准备 Excel 干线运输优化数据。")
    parser.add_argument("--input", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/company"))
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_excel(path, sheet_name="订单数据")
    waybills = pd.read_excel(path, sheet_name="运单数据")
    missing_orders = set(ORDER_COLUMNS.values()) - set(orders.columns)
    missing_waybills = set(WAYBILL_COLUMNS.values()) - set(waybills.columns)
    if missing_orders or missing_waybills:
        raise ValueError(
            f"缺少字段，订单={sorted(missing_orders)}，运单={sorted(missing_waybills)}"
        )
    orders = orders.rename(columns={value: key for key, value in ORDER_COLUMNS.items()})
    waybills = waybills.rename(
        columns={value: key for key, value in WAYBILL_COLUMNS.items()}
    )
    for column in ["volume_cm3", "weight_kg", "pieces"]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    orders["completed_at"] = pd.to_datetime(orders["completed_at"], errors="coerce")
    for column in ["departed_at", "arrived_at"]:
        waybills[column] = pd.to_datetime(waybills[column], errors="coerce")
    waybills["distance_km"] = pd.to_numeric(waybills["distance_km"], errors="coerce")
    return orders, waybills


def build_transport_events(
    orders: pd.DataFrame, waybills: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = orders.set_index("waybill")[
        [
            "volume_cm3",
            "weight_kg",
            "pieces",
            "shipper_province",
            "shipper_city",
            "receiver_province",
            "receiver_city",
            "completed_at",
        ]
    ]
    linked = waybills.join(payload, on="waybill", validate="many_to_one")
    consistency = linked.groupby(LEG_KEY, dropna=False).agg(
        vehicle_type_count=("vehicle_type_name", "nunique"),
        distance_count=("distance_km", "nunique"),
        arrival_count=("arrived_at", "nunique"),
    )
    conflicts = consistency[
        (consistency["vehicle_type_count"] > 1)
        | (consistency["distance_count"] > 1)
        | (consistency["arrival_count"] > 1)
    ].reset_index()
    events = linked.groupby(LEG_KEY, dropna=False, as_index=False).agg(
        vehicle_type_code=("vehicle_type_code", "first"),
        vehicle_type_name=("vehicle_type_name", "first"),
        arrived_at=("arrived_at", "first"),
        distance_km=("distance_km", "first"),
        order_count=("waybill", "nunique"),
        total_weight_kg=("weight_kg", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        total_pieces=("pieces", "sum"),
    )
    city_mix = linked.groupby(LEG_KEY, dropna=False).agg(
        origin_city=("shipper_city", lambda values: values.mode().iat[0] if not values.mode().empty else pd.NA),
        destination_city=("receiver_city", lambda values: values.mode().iat[0] if not values.mode().empty else pd.NA),
        origin_city_count=("shipper_city", "nunique"),
        destination_city_count=("receiver_city", "nunique"),
    ).reset_index()
    events = events.merge(city_mix, on=LEG_KEY, how="left", validate="one_to_one")
    events["duration_hours"] = (
        events["arrived_at"] - events["departed_at"]
    ).dt.total_seconds() / 3600
    events["service_date"] = events["departed_at"].dt.date.astype("string")
    events["departure_hour"] = (
        events["departed_at"].dt.hour + events["departed_at"].dt.minute / 60
    )
    events["weekday"] = events["departed_at"].dt.dayofweek
    events["valid_time"] = events["duration_hours"].gt(0)
    events["valid_distance"] = events["distance_km"].gt(0)
    events["duration_under_72h"] = events["duration_hours"].le(72)
    events["speed_kmh"] = events["distance_km"] / events["duration_hours"]
    events["speed_under_120kmh"] = events["speed_kmh"].le(120)
    events["valid_payload"] = events[["total_weight_kg", "total_volume_cm3"]].notna().all(axis=1)
    events["vehicle_type_conflict"] = pd.MultiIndex.from_frame(
        events[LEG_KEY]
    ).isin(pd.MultiIndex.from_frame(conflicts[LEG_KEY]))
    events["analysis_eligible"] = (
        events["valid_time"]
        & events["valid_distance"]
        & events["duration_under_72h"]
        & events["speed_under_120kmh"]
        & events["valid_payload"]
        & ~events["vehicle_type_conflict"]
    )
    events = events.reset_index(drop=True)
    events["event_id"] = [f"event_{index + 1:06d}" for index in events.index]
    site_names = pd.Index(
        pd.concat(
            [events["departure_site"].astype("string"), events["arrival_site"].astype("string")]
        ).dropna().unique()
    )
    site_map = {name: f"site_{index + 1:04d}" for index, name in enumerate(sorted(site_names))}
    events["origin_site_id"] = events["departure_site"].map(site_map)
    events["destination_site_id"] = events["arrival_site"].map(site_map)
    events["lane_id"] = (
        events["origin_site_id"] + "__" + events["destination_site_id"]
    )
    return events, conflicts


def build_lane_daily(events: pd.DataFrame) -> pd.DataFrame:
    eligible = events[events["analysis_eligible"]].copy()
    daily = eligible.groupby(
        ["lane_id", "origin_site_id", "destination_site_id", "service_date"],
        dropna=False,
        as_index=False,
    ).agg(
        observed_trips=("event_id", "nunique"),
        order_count=("order_count", "sum"),
        total_weight_kg=("total_weight_kg", "sum"),
        total_volume_cm3=("total_volume_cm3", "sum"),
        total_pieces=("total_pieces", "sum"),
        distance_km_p50=("distance_km", "median"),
        duration_hours_p50=("duration_hours", "median"),
        duration_hours_p90=("duration_hours", lambda values: values.quantile(0.9)),
        departure_hour_p50=("departure_hour", "median"),
        destination_city_count_p50=("destination_city_count", "median"),
    )
    daily["month"] = pd.to_datetime(daily["service_date"]).dt.month
    daily["weekday"] = pd.to_datetime(daily["service_date"]).dt.dayofweek
    daily["volume_m3"] = daily["total_volume_cm3"] / 1_000_000
    return daily


def build_lane_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.groupby(
        ["lane_id", "origin_site_id", "destination_site_id"],
        dropna=False,
        as_index=False,
    ).agg(
        active_days=("service_date", "nunique"),
        total_observed_trips=("observed_trips", "sum"),
        total_orders=("order_count", "sum"),
        total_weight_kg=("total_weight_kg", "sum"),
        total_volume_m3=("volume_m3", "sum"),
        distance_km_p50=("distance_km_p50", "median"),
        duration_hours_p50=("duration_hours_p50", "median"),
        duration_hours_p90=("duration_hours_p90", "median"),
        daily_weight_p50_kg=("total_weight_kg", "median"),
        daily_weight_p90_kg=("total_weight_kg", lambda values: values.quantile(0.9)),
        daily_volume_p50_m3=("volume_m3", "median"),
        daily_volume_p90_m3=("volume_m3", lambda values: values.quantile(0.9)),
        destination_city_count_p50=("destination_city_count_p50", "median"),
    ).sort_values("total_orders", ascending=False)


def main() -> None:
    args = parse_args()
    orders, waybills = load_data(args.input)
    events, conflicts = build_transport_events(orders, waybills)
    daily = build_lane_daily(events)
    summary = build_lane_summary(daily)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "transport_events.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(args.output_dir / "lane_daily_demand.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.result_dir / "lane_summary.csv", index=False, encoding="utf-8-sig")
    audit = {
        "source": args.input.name,
        "orders": int(len(orders)),
        "waybill_rows": int(len(waybills)),
        "transport_events": int(len(events)),
        "eligible_events": int(events["analysis_eligible"].sum()),
        "lane_count": int(summary["lane_id"].nunique()),
        "lane_day_rows": int(len(daily)),
        "vehicle_type_conflicts": int(len(conflicts)),
        "receiver_city_field_used": ORDER_COLUMNS["receiver_city"],
        "privacy_note": "Raw vehicle, waybill and site names remain in ignored processed outputs; public summaries use anonymized site and lane IDs.",
    }
    (args.result_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
