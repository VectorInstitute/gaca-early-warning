"""Evaluation metrics computation for model performance assessment.

This module provides functions to compute RMSE, MAE, and other evaluation
metrics by comparing model predictions against NOAA ground truth observations.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gaca_ews.core.data_extraction import fetch_last_hours


def compute_rmse(predictions: NDArray[Any], targets: NDArray[Any]) -> float:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    predictions : NDArray[Any]
        Model predictions
    targets : NDArray[Any]
        Ground truth values

    Returns
    -------
    float
        RMSE value
    """
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mae(predictions: NDArray[Any], targets: NDArray[Any]) -> float:
    """Compute Mean Absolute Error.

    Parameters
    ----------
    predictions : NDArray[Any]
        Model predictions
    targets : NDArray[Any]
    Ground truth values

    Returns
    -------
    float
        MAE value
    """
    return float(np.mean(np.abs(predictions - targets)))


def fetch_ground_truth_for_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch NOAA ground truth data matching prediction timestamps.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with forecast_time, lat, lon columns

    Returns
    -------
    pd.DataFrame
        Ground truth data with timestamp, lat, lon, actual_temp columns
    """
    # Get unique forecast times
    forecast_times = pd.to_datetime(predictions_df["forecast_time"]).unique()

    if len(forecast_times) == 0:
        return pd.DataFrame(columns=["timestamp", "lat", "lon", "actual_temp"])

    # Determine time range
    min_time = forecast_times.min()
    max_time = forecast_times.max()

    # Fetch NOAA data for the time range
    # Calculate hours back from max_time to min_time
    hours_back = int((max_time - min_time).total_seconds() / 3600) + 1

    try:
        # Fetch data using existing extraction function
        # Create config object matching expected signature
        cfg = SimpleNamespace(
            num_hours_to_fetch=hours_back,
            region=SimpleNamespace(
                lat_min=predictions_df["lat"].min(),
                lat_max=predictions_df["lat"].max(),
                lon_min=predictions_df["lon"].min(),
                lon_max=predictions_df["lon"].max(),
            ),
        )
        noaa_data, _ = fetch_last_hours(cfg)

        # Convert to ground truth format
        ground_truth = []
        for timestamp in noaa_data["time"].unique():
            time_data = noaa_data[noaa_data["time"] == timestamp]

            for _, row in time_data.iterrows():
                ground_truth.append(
                    {
                        "timestamp": pd.to_datetime(timestamp),
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "actual_temp": row["t2m"],  # 2m temperature in Celsius
                    }
                )

        return pd.DataFrame(ground_truth)

    except Exception as e:
        # If ground truth fetch fails, return empty DataFrame
        print(f"Failed to fetch ground truth: {e}")
        return pd.DataFrame(columns=["timestamp", "lat", "lon", "actual_temp"])


def match_predictions_with_ground_truth(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    tolerance_km: float = 5.0,
) -> pd.DataFrame:
    """Match predictions with ground truth observations.

    Matches based on forecast time and spatial proximity (lat/lon).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with forecast_time, horizon_hours, lat, lon, predicted_temp
    ground_truth_df : pd.DataFrame
        Ground truth with timestamp, lat, lon, actual_temp
    tolerance_km : float
        Maximum distance in kilometers for spatial matching

    Returns
    -------
    pd.DataFrame
        Matched data with predicted_temp, actual_temp, horizon_hours
    """
    # Convert forecast_time to datetime
    predictions_df = predictions_df.copy()
    predictions_df["forecast_time"] = pd.to_datetime(predictions_df["forecast_time"])

    ground_truth_df = ground_truth_df.copy()
    ground_truth_df["timestamp"] = pd.to_datetime(ground_truth_df["timestamp"])

    matched_data = []

    # Approximate degrees for tolerance
    # At 45° latitude (Ontario): 1° lat ≈ 111km, 1° lon ≈ 78km
    lat_tolerance = tolerance_km / 111.0
    lon_tolerance = tolerance_km / 78.0

    for _, pred_row in predictions_df.iterrows():
        # Find ground truth observations matching the forecast time
        time_matches = ground_truth_df[
            ground_truth_df["timestamp"] == pred_row["forecast_time"]
        ]

        if len(time_matches) == 0:
            continue

        # Find spatial matches within tolerance
        spatial_matches = time_matches[
            (abs(time_matches["lat"] - pred_row["lat"]) <= lat_tolerance)
            & (abs(time_matches["lon"] - pred_row["lon"]) <= lon_tolerance)
        ]

        if len(spatial_matches) > 0:
            # Use the closest match
            spatial_matches = spatial_matches.copy()
            spatial_matches["distance"] = np.sqrt(
                (spatial_matches["lat"] - pred_row["lat"]) ** 2
                + (spatial_matches["lon"] - pred_row["lon"]) ** 2
            )
            closest = spatial_matches.loc[spatial_matches["distance"].idxmin()]

            matched_data.append(
                {
                    "forecast_time": pred_row["forecast_time"],
                    "horizon_hours": pred_row["horizon_hours"],
                    "lat": pred_row["lat"],
                    "lon": pred_row["lon"],
                    "predicted_temp": pred_row["predicted_temp"],
                    "actual_temp": closest["actual_temp"],
                }
            )

    return pd.DataFrame(matched_data)


def compute_evaluation_metrics(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> dict[str, Any]:
    """Compute evaluation metrics by horizon.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with forecast_time, horizon_hours, lat, lon, predicted_temp
    ground_truth_df : pd.DataFrame
        Ground truth with timestamp, lat, lon, actual_temp

    Returns
    -------
    dict[str, Any]
        Evaluation metrics organized by horizon
    """
    # Match predictions with ground truth
    matched_df = match_predictions_with_ground_truth(predictions_df, ground_truth_df)

    if len(matched_df) == 0:
        return {
            "overall": {"rmse": None, "mae": None, "sample_count": 0},
            "by_horizon": {},
        }

    # Compute overall metrics
    overall_rmse = compute_rmse(
        matched_df["predicted_temp"].to_numpy(), matched_df["actual_temp"].to_numpy()
    )
    overall_mae = compute_mae(
        matched_df["predicted_temp"].to_numpy(), matched_df["actual_temp"].to_numpy()
    )

    # Compute metrics by horizon
    by_horizon = {}
    for horizon in sorted(matched_df["horizon_hours"].unique()):
        horizon_data = matched_df[matched_df["horizon_hours"] == horizon]

        if len(horizon_data) > 0:
            rmse = compute_rmse(
                horizon_data["predicted_temp"].to_numpy(),
                horizon_data["actual_temp"].to_numpy(),
            )
            mae = compute_mae(
                horizon_data["predicted_temp"].to_numpy(),
                horizon_data["actual_temp"].to_numpy(),
            )

            by_horizon[str(int(horizon))] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "sample_count": len(horizon_data),
            }

    return {
        "overall": {
            "rmse": float(overall_rmse),
            "mae": float(overall_mae),
            "sample_count": len(matched_df),
        },
        "by_horizon": by_horizon,
    }


def compute_static_evaluation(
    predictions_list: list[dict[str, Any]], ground_truth_list: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute static evaluation metrics for validation period.

    Parameters
    ----------
    predictions_list : list[dict[str, Any]]
        List of prediction run dictionaries from storage
    ground_truth_list : list[dict[str, Any]]
        List of ground truth observations from storage

    Returns
    -------
    dict[str, Any]
        Static evaluation metrics
    """
    # Convert to DataFrames
    all_predictions = []
    for run in predictions_list:
        for pred in run.get("predictions", []):
            all_predictions.append(pred)

    if len(all_predictions) == 0:
        return {
            "overall": {"rmse": None, "mae": None, "sample_count": 0},
            "by_horizon": {},
        }

    predictions_df = pd.DataFrame(all_predictions)
    ground_truth_df = pd.DataFrame(ground_truth_list)

    return compute_evaluation_metrics(predictions_df, ground_truth_df)


def compute_dynamic_evaluation(
    predictions_list: list[dict[str, Any]],
    ground_truth_list: list[dict[str, Any]],
    window_days: int = 30,
) -> dict[str, Any]:
    """Compute dynamic evaluation metrics for rolling window.

    Parameters
    ----------
    predictions_list : list[dict[str, Any]]
        List of recent prediction runs from storage
    ground_truth_list : list[dict[str, Any]]
        List of recent ground truth observations
    window_days : int
        Rolling window size in days

    Returns
    -------
    dict[str, Any]
        Dynamic evaluation metrics with time series
    """
    if len(predictions_list) == 0:
        return {
            "current": {
                "overall": {"rmse": None, "mae": None, "sample_count": 0},
                "by_horizon": {},
            },
            "time_series": [],
        }

    # Compute current window metrics (same as static for the window period)
    all_predictions = []
    for run in predictions_list:
        for pred in run.get("predictions", []):
            all_predictions.append(pred)

    predictions_df = pd.DataFrame(all_predictions)
    ground_truth_df = pd.DataFrame(ground_truth_list)

    current_metrics = compute_evaluation_metrics(predictions_df, ground_truth_df)

    # Compute daily time series for the window
    time_series = []
    predictions_df["forecast_time"] = pd.to_datetime(predictions_df["forecast_time"])
    predictions_df["date"] = predictions_df["forecast_time"].dt.date

    for date in sorted(predictions_df["date"].unique()):
        daily_preds = predictions_df[predictions_df["date"] == date]

        if len(daily_preds) > 0:
            daily_metrics = compute_evaluation_metrics(daily_preds, ground_truth_df)

            time_series.append(
                {
                    "date": str(date),
                    "metrics": daily_metrics,
                }
            )

    return {
        "current": current_metrics,
        "time_series": time_series,
    }
