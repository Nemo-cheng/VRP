#!/usr/bin/env python3
"""Allocate physical-leg carbon emissions to anonymized waybills."""

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
from typing import Any

import pandas as pd

from experiment.build_carbon_baseline import build_baseline
from experiment.prepare_company_data import (
    LEG_KEY,
    load_source,
    prepare_orders,
    prepare_waybills,
)
from experiment.validate_parameter_registry import load_registry

ANON_LEG_KEY = ["vehicle_id", "departed_at", "departure_site", "arrival_site"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建匿名逐运单碳排放账本。")
    parser.add_argument("--orders", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--vehicles", type=Path, default=Path("车辆数据.xlsx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/carbon_ledger"),
    )
    return parser.parse_args()


def add_allocation_shares(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    positive_weight = result["weight_kg"].fillna(0).clip(lower=0)
    result["_positive_weight"] = positive_weight
    result["_leg_weight"] = result.groupby(ANON_LEG_KEY, dropna=False)[
        "_positive_weight"
    ].transform("sum")
    result["_leg_shipments"] = result.groupby(ANON_LEG_KEY, dropna=False)[
        "waybill"
    ].transform("size")
    result["allocation_method"] = "weight_share"
    result["allocation_share"] = result["_positive_weight"] / result["_leg_weight"]
    equal_mask = result["_leg_weight"] <= 0
    result.loc[equal_mask, "allocation_method"] = "equal_share"
    result.loc[equal_mask, "allocation_share"] = (
        1 / result.loc[equal_mask, "_leg_shipments"]
    )
    return result.drop(
        columns=["_positive_weight", "_leg_weight", "_leg_shipments"]
    )


def anonymize(values: pd.Series, prefix: str) -> pd.Series:
    codes, _ = pd.factorize(values, sort=True)
    return pd.Series(codes, index=values.index).map(
        lambda value: f"{prefix}_{value + 1:05d}"
    )


def build_ledger(
    orders_path: Path,
    vehicles_path: Path,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible, _, _ = build_baseline(orders_path, vehicles_path, registry)
    orders_raw, waybills_raw = load_source(orders_path)
    orders = prepare_orders(orders_raw)
    waybills = prepare_waybills(waybills_raw)

    vehicle_names = sorted(waybills["vehicle"].dropna().unique())
    vehicle_ids = {
        vehicle: f"vehicle_{index + 1:05d}"
        for index, vehicle in enumerate(vehicle_names)
    }
    waybills["vehicle_id"] = waybills["vehicle"].map(vehicle_ids)
    relations = (
        waybills[["waybill", *LEG_KEY]]
        .assign(vehicle_id=waybills["vehicle_id"])
        .drop(columns="vehicle")
        .drop_duplicates()
    )
    relations = relations.merge(
        orders[["waybill", "weight_kg"]],
        on="waybill",
        how="left",
        validate="many_to_one",
    )
    leg_values = eligible[
        [
            *ANON_LEG_KEY,
            "vehicle_type_name",
            "fuel",
            "payload_t",
            "distance_km",
            "emissions_kgco2_min",
            "emissions_kgco2_max",
        ]
    ]
    allocated = relations.merge(
        leg_values,
        on=ANON_LEG_KEY,
        how="inner",
        validate="many_to_one",
    )
    allocated = add_allocation_shares(allocated)
    allocated["allocated_emissions_kgco2_min"] = (
        allocated["emissions_kgco2_min"] * allocated["allocation_share"]
    )
    allocated["allocated_emissions_kgco2_max"] = (
        allocated["emissions_kgco2_max"] * allocated["allocation_share"]
    )

    share_check = allocated.groupby(ANON_LEG_KEY, dropna=False)[
        "allocation_share"
    ].sum()
    if not ((share_check - 1).abs() < 1e-9).all():
        raise ValueError("物理运输段的排放分摊比例未守恒")

    allocated["waybill_id"] = anonymize(allocated["waybill"], "waybill")
    allocated["leg_key"] = list(
        zip(
            *[allocated[column] for column in ANON_LEG_KEY],
            strict=True,
        )
    )
    allocated["leg_id"] = anonymize(allocated["leg_key"], "leg")
    all_sites = pd.concat(
        [allocated["departure_site"], allocated["arrival_site"]],
        ignore_index=True,
    )
    site_names = sorted(all_sites.dropna().unique())
    site_ids = {
        site: f"site_{index + 1:04d}" for index, site in enumerate(site_names)
    }
    allocated["departure_site_id"] = allocated["departure_site"].map(site_ids)
    allocated["arrival_site_id"] = allocated["arrival_site"].map(site_ids)
    allocated["lane_key"] = list(
        zip(
            allocated["departure_site_id"],
            allocated["arrival_site_id"],
            strict=True,
        )
    )
    allocated["lane_id"] = anonymize(allocated["lane_key"], "lane")

    waybill_summary = (
        allocated.groupby("waybill_id", as_index=False)
        .agg(
            physical_legs=("vehicle_id", "size"),
            transported_weight_kg=("weight_kg", "first"),
            carried_distance_km=("distance_km", "sum"),
            emissions_kgco2_min=("allocated_emissions_kgco2_min", "sum"),
            emissions_kgco2_max=("allocated_emissions_kgco2_max", "sum"),
        )
        .sort_values("emissions_kgco2_max", ascending=False)
    )
    lane_waybills = (
        allocated.groupby(
            ["lane_id", "departure_site_id", "arrival_site_id"], as_index=False
        )
        .agg(
            waybills=("waybill_id", "nunique"),
        )
    )
    lane_legs = (
        allocated.drop_duplicates("leg_id")
        .groupby(
            ["lane_id", "departure_site_id", "arrival_site_id"], as_index=False
        )
        .agg(
            physical_legs=("leg_id", "size"),
            vehicle_distance_km=("distance_km", "sum"),
            emissions_kgco2_min=("emissions_kgco2_min", "sum"),
            emissions_kgco2_max=("emissions_kgco2_max", "sum"),
        )
    )
    lane_summary = (
        lane_legs.merge(
            lane_waybills,
            on=["lane_id", "departure_site_id", "arrival_site_id"],
            validate="one_to_one",
        )
        .sort_values("emissions_kgco2_max", ascending=False)
    )
    vehicle_type_summary = (
        eligible.groupby(
            ["vehicle_type_name", "fuel", "payload_t"],
            as_index=False,
            dropna=False,
        )
        .agg(
            physical_legs=("vehicle_id", "size"),
            vehicles=("vehicle_id", "nunique"),
            distance_km=("distance_km", "sum"),
            emissions_kgco2_min=("emissions_kgco2_min", "sum"),
            emissions_kgco2_max=("emissions_kgco2_max", "sum"),
        )
        .sort_values("emissions_kgco2_max", ascending=False)
    )

    leg_min = float(eligible["emissions_kgco2_min"].sum())
    leg_max = float(eligible["emissions_kgco2_max"].sum())
    allocated_min = float(allocated["allocated_emissions_kgco2_min"].sum())
    allocated_max = float(allocated["allocated_emissions_kgco2_max"].sum())
    summary = {
        "allocation_standard": "ISO 14083:2023",
        "allocation_method": "weight_share",
        "fallback_method": "equal_share_when_total_positive_weight_is_zero",
        "eligible_physical_legs": len(eligible),
        "allocated_leg_waybill_relations": len(allocated),
        "allocated_waybills": int(allocated["waybill_id"].nunique()),
        "source_waybills": int(orders["waybill"].nunique()),
        "excluded_waybills": int(
            orders["waybill"].nunique() - allocated["waybill_id"].nunique()
        ),
        "lanes": int(allocated["lane_id"].nunique()),
        "equal_share_relations": int(
            (allocated["allocation_method"] == "equal_share").sum()
        ),
        "leg_emissions_kgco2": {"minimum": leg_min, "maximum": leg_max},
        "allocated_emissions_kgco2": {
            "minimum": allocated_min,
            "maximum": allocated_max,
        },
        "reconciliation_error_kgco2": {
            "minimum": allocated_min - leg_min,
            "maximum": allocated_max - leg_max,
        },
    }
    return waybill_summary, lane_summary, vehicle_type_summary, summary


def main() -> None:
    args = parse_args()
    for path in (args.orders, args.vehicles):
        if not path.exists():
            raise FileNotFoundError(f"找不到输入文件: {path}")
    waybills, lanes, vehicle_types, summary = build_ledger(
        args.orders, args.vehicles, load_registry()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waybills.to_csv(
        args.output_dir / "waybill_carbon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lanes.to_csv(
        args.output_dir / "lane_carbon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vehicle_types.to_csv(
        args.output_dir / "vehicle_type_carbon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "carbon_ledger_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
