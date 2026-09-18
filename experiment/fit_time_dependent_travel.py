#!/usr/bin/env python3
"""Fit time-dependent composite transition models from recovered LaDe routes."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=15",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_SEGMENT_ROWS = 500
NEAR_ZERO_DISTANCE_KM = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拟合 LaDe 分时段综合转移时间。")
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("processed/lade/shanghai_historical_routes.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed/lade")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/time_model")
    )
    return parser.parse_args()


def time_period(hour: pd.Series) -> pd.Series:
    conditions = [
        hour.between(7, 9, inclusive="both"),
        hour.between(10, 13, inclusive="both"),
        hour.between(14, 16, inclusive="both"),
        hour.between(17, 20, inclusive="both"),
    ]
    labels = ["morning_peak", "midday", "afternoon", "evening_peak"]
    return pd.Series(
        np.select(conditions, labels, default="night"), index=hour.index
    )


def build_transition_data(nodes: pd.DataFrame) -> pd.DataFrame:
    data = nodes.loc[
        nodes["gap_minutes"].notna()
        & nodes["adjacent_distance_km"].notna()
        & nodes["gap_minutes"].between(0.5, 120)
        & nodes["adjacent_distance_km"].between(0, 10)
    ].copy()
    data["previous_completion_dt"] = data["delivery_dt"] - pd.to_timedelta(
        data["gap_minutes"], unit="m"
    )
    data["day_type"] = np.where(
        data["previous_completion_dt"].dt.dayofweek < 5, "weekday", "weekend"
    )
    data["time_period"] = time_period(data["previous_completion_dt"].dt.hour)
    data["service_date"] = data["delivery_dt"].dt.normalize()
    return data


def robust_slope(
    distance: pd.Series, duration: pd.Series, service_minutes: float
) -> tuple[float, int]:
    x = distance.to_numpy(dtype=float)
    y = duration.to_numpy(dtype=float) - service_minutes
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0.01) & (y >= 0)
    for _ in range(4):
        slope = max(0.0, float(np.dot(x[keep], y[keep]) / np.dot(x[keep], x[keep])))
        residual = y - slope * x
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        if mad == 0:
            break
        keep &= np.abs(residual - median) <= 3 * 1.4826 * mad
    return slope, int(keep.sum())


def fit_parameters(train: pd.DataFrame) -> pd.DataFrame:
    near_zero = train.loc[
        train["adjacent_distance_km"] <= NEAR_ZERO_DISTANCE_KM, "gap_minutes"
    ]
    global_service = float(near_zero.median())
    global_slope, global_fit_rows = robust_slope(
        train["adjacent_distance_km"], train["gap_minutes"], global_service
    )
    rows = [
        {
            "day_type": "all",
            "time_period": "all",
            "observations": len(train),
            "near_zero_observations": len(near_zero),
            "service_minutes": global_service,
            "minutes_per_straight_km": global_slope,
            "fit_rows": global_fit_rows,
            "fallback_to_global": False,
        }
    ]
    for (day_type, period), group in train.groupby(
        ["day_type", "time_period"], sort=True
    ):
        local_zero = group.loc[
            group["adjacent_distance_km"] <= NEAR_ZERO_DISTANCE_KM,
            "gap_minutes",
        ]
        enough = len(group) >= MIN_SEGMENT_ROWS and len(local_zero) >= 100
        service = float(local_zero.median()) if enough else global_service
        slope, fit_rows = (
            robust_slope(
                group["adjacent_distance_km"], group["gap_minutes"], service
            )
            if enough
            else (global_slope, global_fit_rows)
        )
        rows.append(
            {
                "day_type": day_type,
                "time_period": period,
                "observations": len(group),
                "near_zero_observations": len(local_zero),
                "service_minutes": service,
                "minutes_per_straight_km": slope,
                "fit_rows": fit_rows,
                "fallback_to_global": not enough,
            }
        )
    return pd.DataFrame(rows)


def predict(frame: pd.DataFrame, parameters: pd.DataFrame) -> pd.Series:
    segmented = parameters[
        ~parameters["day_type"].eq("all")
    ].set_index(["day_type", "time_period"])
    keys = pd.MultiIndex.from_frame(frame[["day_type", "time_period"]])
    service = segmented["service_minutes"].reindex(keys).to_numpy()
    slope = segmented["minutes_per_straight_km"].reindex(keys).to_numpy()
    return pd.Series(
        service + slope * frame["adjacent_distance_km"].to_numpy(), index=frame.index
    )


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = (actual - predicted).abs()
    return {
        "mae_minutes": float(error.mean()),
        "median_absolute_error_minutes": float(error.median()),
        "mape_percent_for_gaps_at_least_5min": float(
            (error[actual >= 5] / actual[actual >= 5]).mean() * 100
        ),
    }


def fit_and_validate(
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dates = np.sort(transitions["service_date"].unique())
    split_index = max(1, int(len(dates) * 0.8))
    train_dates = dates[:split_index]
    validation_dates = dates[split_index:]
    train = transitions[transitions["service_date"].isin(train_dates)].copy()
    validation = transitions[
        transitions["service_date"].isin(validation_dates)
    ].copy()
    parameters = fit_parameters(train)
    segmented_prediction = predict(validation, parameters)
    global_row = parameters.iloc[0]
    global_prediction = (
        global_row["service_minutes"]
        + global_row["minutes_per_straight_km"]
        * validation["adjacent_distance_km"]
    )
    validation_output = validation[
        [
            "route_id",
            "stop_index",
            "service_date",
            "day_type",
            "time_period",
            "adjacent_distance_km",
            "gap_minutes",
        ]
    ].copy()
    validation_output["segmented_prediction_minutes"] = segmented_prediction
    validation_output["global_prediction_minutes"] = global_prediction
    segmented_metrics = error_metrics(
        validation["gap_minutes"], segmented_prediction
    )
    global_metrics = error_metrics(validation["gap_minutes"], global_prediction)
    report = {
        "eligible_transition_pairs": int(len(transitions)),
        "train_pairs": int(len(train)),
        "validation_pairs": int(len(validation)),
        "train_date_min": str(pd.Timestamp(train_dates.min()).date()),
        "train_date_max": str(pd.Timestamp(train_dates.max()).date()),
        "validation_date_min": str(pd.Timestamp(validation_dates.min()).date()),
        "validation_date_max": str(pd.Timestamp(validation_dates.max()).date()),
        "near_zero_distance_threshold_km": NEAR_ZERO_DISTANCE_KM,
        "segmented_model": segmented_metrics,
        "global_model": global_metrics,
        "mae_improvement_percent": float(
            (
                1
                - segmented_metrics["mae_minutes"]
                / global_metrics["mae_minutes"]
            )
            * 100
        ),
        "interpretation": "Predictions are composite transition times based on straight-line distance. They are not pure vehicle travel times.",
    }
    return parameters, validation_output, report


def main() -> None:
    args = parse_args()
    nodes = pd.read_parquet(args.nodes)
    transitions = build_transition_data(nodes)
    parameters, validation, report = fit_and_validate(transitions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(
        args.output_dir / "time_dependent_transition_parameters.csv", index=False
    )
    validation.to_parquet(
        args.output_dir / "time_model_validation_predictions.parquet", index=False
    )
    parameters.to_csv(args.result_dir / "time_model_parameters.csv", index=False)
    (args.result_dir / "validation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
