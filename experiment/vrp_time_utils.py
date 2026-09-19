"""Shared time-bucket and path travel-time helpers for the VRP pipeline."""

from __future__ import annotations

import pandas as pd


def period_for_timestamp(timestamp: pd.Timestamp) -> str:
    hour = timestamp.hour
    if hour <= 5:
        return "night"
    if hour <= 9:
        return "morning_peak"
    if hour <= 15:
        return "daytime"
    if hour <= 19:
        return "evening_peak"
    return "evening"


def evaluate_time_dependent_path(
    path: list[str],
    departed_at: pd.Timestamp,
    edge_lookup: dict[tuple[str, str], float],
    period_lookup: dict[tuple[str, str, str], float],
) -> tuple[float, int, int]:
    current_time = departed_at
    reliable_period_edges = 0
    total_edges = 0
    for origin, destination in zip(path, path[1:]):
        total_edges += 1
        period = period_for_timestamp(current_time)
        period_key = (origin, destination, period)
        if period_key in period_lookup:
            duration = period_lookup[period_key]
            reliable_period_edges += 1
        else:
            duration = edge_lookup[(origin, destination)]
        current_time += pd.to_timedelta(duration, unit="h")
    return (
        (current_time - departed_at).total_seconds() / 3600,
        reliable_period_edges,
        total_edges,
    )
