import pandas as pd

from fit_time_dependent_travel import (
    build_transition_data,
    fit_parameters,
    predict,
    time_period,
)


def test_time_period_boundaries() -> None:
    periods = time_period(pd.Series([6, 7, 9, 10, 14, 17, 21]))
    assert periods.tolist() == [
        "night",
        "morning_peak",
        "morning_peak",
        "midday",
        "afternoon",
        "evening_peak",
        "night",
    ]


def test_build_fit_and_predict_transition_model() -> None:
    rows = 1200
    nodes = pd.DataFrame(
        {
            "route_id": ["R1"] * rows,
            "stop_index": range(1, rows + 1),
            "delivery_dt": pd.date_range(
                "2022-05-02 08:00", periods=rows, freq="10min"
            ),
            "gap_minutes": [10.0] * rows,
            "adjacent_distance_km": [0.01] * 200 + [1.0] * 1000,
        }
    )
    transitions = build_transition_data(nodes)
    parameters = fit_parameters(transitions)
    predictions = predict(transitions.iloc[:10], parameters)

    assert len(transitions) == rows
    assert parameters.iloc[0]["service_minutes"] == 10.0
    assert predictions.notna().all()
