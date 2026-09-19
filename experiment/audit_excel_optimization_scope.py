#!/usr/bin/env python3
"""Audit which optimization questions are identifiable from the Excel workbook."""

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

from prepare_excel_transport_data import build_transport_events, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计订单数据可支持的优化问题。")
    parser.add_argument("--input", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/company_transport/optimization_scope_audit.json"),
    )
    return parser.parse_args()


def audit_scope(orders: pd.DataFrame, waybills: pd.DataFrame) -> dict[str, object]:
    events, _ = build_transport_events(orders, waybills)
    eligible = events[events["analysis_eligible"]].copy()

    legs = waybills.sort_values(["waybill", "departed_at"]).copy()
    legs["next_origin"] = legs.groupby("waybill")["departure_site"].shift(-1)
    legs["next_departure"] = legs.groupby("waybill")["departed_at"].shift(-1)
    adjacent = legs["next_origin"].notna()
    legs["site_link_valid"] = ~adjacent | legs["arrival_site"].eq(legs["next_origin"])
    legs["time_link_valid"] = ~adjacent | legs["arrived_at"].le(
        legs["next_departure"]
    )

    grouped = legs.groupby("waybill", sort=False)
    routes = grouped.agg(
        origin=("departure_site", "first"),
        destination=("arrival_site", "last"),
        leg_count=("waybill", "size"),
        first_departure=("departed_at", "min"),
        last_arrival=("arrived_at", "max"),
        total_distance_km=("distance_km", "sum"),
        site_sequence_valid=("site_link_valid", "all"),
        time_sequence_valid=("time_link_valid", "all"),
    )
    origin_path = grouped["departure_site"].agg(
        lambda values: ">".join(values.astype("string"))
    )
    routes["path"] = origin_path + ">" + grouped["arrival_site"].last().astype("string")

    routes["valid_sequence"] = (
        routes["site_sequence_valid"] & routes["time_sequence_valid"]
    )
    valid_routes = routes[routes["valid_sequence"]]
    path_counts = (
        valid_routes.groupby(["origin", "destination", "path"], dropna=False)
        .size()
        .rename("observations")
        .reset_index()
    )
    od_summary = path_counts.groupby(["origin", "destination"], dropna=False).agg(
        orders=("observations", "sum"),
        paths=("path", "nunique"),
        paths_observed_at_least_twice=(
            "observations",
            lambda values: int((values >= 2).sum()),
        ),
    )

    edge_counts = legs.groupby(["departure_site", "arrival_site"], dropna=False).size()
    departure_hour = legs["departed_at"].dt.hour
    legs["period"] = pd.cut(
        departure_hour,
        bins=[-1, 5, 9, 15, 19, 23],
        labels=["night", "morning_peak", "daytime", "evening_peak", "evening"],
    )
    edge_period_counts = legs.groupby(
        ["departure_site", "arrival_site", "period"],
        observed=True,
        dropna=False,
    ).size()

    multi_path_od = od_summary["paths"] >= 2
    repeated_alternatives = od_summary["paths_observed_at_least_twice"] >= 2
    event_order_quantiles = eligible["order_count"].quantile([0.5, 0.9, 0.99])
    weight_quantiles = eligible["total_weight_kg"].quantile([0.5, 0.9, 0.99])

    return {
        "source_only": "订单数据.xlsx",
        "recommended_problem": "网点级历史候选路径与发车时段优化",
        "identifiability_conclusion": {
            "supported": [
                "运单经过的网点路径重构",
                "历史候选路径的距离、运输时间、换乘次数和稳定性比较",
                "高频运输边的分时段旅行时间估计",
                "以订单重量乘距离表示的运输周转量优化",
            ],
            "not_supported": [
                "车辆真实装载率和车次合并",
                "车型容量优化和车队规模配置",
                "燃油车与电动车替换决策",
                "真实运输成本和碳排放量核算",
                "客户级道路访问顺序和充电路径",
            ],
        },
        "manifest_coverage_evidence": {
            "eligible_physical_events": int(len(eligible)),
            "events_with_one_observed_order_share": float(
                eligible["order_count"].eq(1).mean()
            ),
            "orders_per_event_p50": float(event_order_quantiles.loc[0.5]),
            "orders_per_event_p90": float(event_order_quantiles.loc[0.9]),
            "orders_per_event_p99": float(event_order_quantiles.loc[0.99]),
            "observed_weight_kg_p50": float(weight_quantiles.loc[0.5]),
            "observed_weight_kg_p90": float(weight_quantiles.loc[0.9]),
            "observed_weight_kg_p99": float(weight_quantiles.loc[0.99]),
            "interpretation": "订单记录不是车辆完整载货清单，不能用汇总重量反推车辆利用率。",
        },
        "route_evidence": {
            "waybills": int(len(routes)),
            "multi_leg_waybills": int(routes["leg_count"].gt(1).sum()),
            "valid_sequence_waybills": int(routes["valid_sequence"].sum()),
            "valid_sequence_share": float(routes["valid_sequence"].mean()),
            "origin_destination_pairs": int(len(od_summary)),
            "multi_path_pairs": int(multi_path_od.sum()),
            "orders_in_multi_path_pairs": int(od_summary.loc[multi_path_od, "orders"].sum()),
            "pairs_with_two_repeated_alternatives": int(repeated_alternatives.sum()),
            "adjacent_legs": int(adjacent.sum()),
            "site_continuity_rate": float(
                legs.loc[adjacent, "arrival_site"]
                .eq(legs.loc[adjacent, "next_origin"])
                .mean()
            ),
            "non_overlapping_time_rate": float(
                legs.loc[adjacent, "arrived_at"]
                .le(legs.loc[adjacent, "next_departure"])
                .mean()
            ),
        },
        "time_dependent_edge_evidence": {
            "directed_edges": int(len(edge_counts)),
            "edges_with_at_least_5_observations": int(edge_counts.ge(5).sum()),
            "edges_with_at_least_10_observations": int(edge_counts.ge(10).sum()),
            "edges_with_at_least_20_observations": int(edge_counts.ge(20).sum()),
            "edge_period_cells": int(len(edge_period_counts)),
            "edge_period_cells_with_at_least_5_observations": int(
                edge_period_counts.ge(5).sum()
            ),
            "edge_period_cells_with_at_least_10_observations": int(
                edge_period_counts.ge(10).sum()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    orders, waybills = load_data(args.input)
    report = audit_scope(orders, waybills)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
