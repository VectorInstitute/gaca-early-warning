"""BigQuery storage for evaluation data.

This module handles storing and retrieving predictions, ground truth data,
and computed evaluation metrics in Google Cloud BigQuery.
"""

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from google.cloud import bigquery
from rich.console import Console


console = Console()


class EvaluationStorage:
    """Manages BigQuery storage for evaluation data."""

    def __init__(self) -> None:
        """Initialize BigQuery client.

        Uses Application Default Credentials (ADC) for authentication.
        In Cloud Run, this automatically uses the service account.
        For local development, use `gcloud auth application-default login`.

        Raises
        ------
        Exception
            If BigQuery connection cannot be established
        """
        # Get project ID from environment or let BigQuery auto-detect
        project_id = os.getenv("GCP_PROJECT_ID")
        dataset_id = os.getenv("BIGQUERY_DATASET", "gaca_evaluation")

        self.client = (
            bigquery.Client(project=project_id) if project_id else bigquery.Client()
        )
        self.dataset_id = dataset_id
        self.project_id = self.client.project

        # Test connection by checking dataset exists
        try:
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            self.client.get_dataset(dataset_ref)
        except Exception as e:
            raise Exception(
                f"BigQuery connection failed. Dataset '{dataset_ref}' not found. "
                f"Run: ./bigquery/setup.sh {self.project_id}"
            ) from e

    def store_predictions(
        self, predictions_df: pd.DataFrame, run_timestamp: datetime
    ) -> int:
        """Store predictions from a model run using fast bulk loading.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with columns: forecast_time, horizon_hours, lat, lon,
            predicted_temp
        run_timestamp : datetime
            Timestamp when the inference was run

        Returns
        -------
        int
            Number of rows loaded
        """
        table_id = f"{self.project_id}.{self.dataset_id}.predictions"

        # Add run_timestamp column
        df = predictions_df.copy()
        df["run_timestamp"] = run_timestamp

        # Ensure correct data types
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
        df["forecast_time"] = pd.to_datetime(df["forecast_time"])
        df["horizon_hours"] = df["horizon_hours"].astype(int)
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
        df["predicted_temp"] = df["predicted_temp"].astype(float)

        # Reorder columns to match schema
        df = df[
            [
                "run_timestamp",
                "forecast_time",
                "horizon_hours",
                "lat",
                "lon",
                "predicted_temp",
            ]
        ]

        console.print(f"[cyan]Loading {len(df):,} predictions to BigQuery...[/cyan]")

        # Configure load job for speed
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema=[
                bigquery.SchemaField("run_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("forecast_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("horizon_hours", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("lat", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("lon", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("predicted_temp", "FLOAT64", mode="REQUIRED"),
            ],
        )

        # Bulk load using pandas DataFrame (very fast)
        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for completion

        console.print(f"[green]✓[/green] Loaded {len(df):,} predictions to BigQuery")

        return len(df)

    def store_ground_truth(self, ground_truth_df: pd.DataFrame) -> int:
        """Store ground truth observations using fast bulk loading.

        Parameters
        ----------
        ground_truth_df : pd.DataFrame
            DataFrame with columns: timestamp, lat, lon, actual_temp

        Returns
        -------
        int
            Number of rows loaded
        """
        table_id = f"{self.project_id}.{self.dataset_id}.ground_truth"

        # Ensure correct data types
        df = ground_truth_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
        df["actual_temp"] = df["actual_temp"].astype(float)

        # Reorder columns to match schema
        df = df[["timestamp", "lat", "lon", "actual_temp"]]

        console.print(
            f"[cyan]Loading {len(df):,} ground truth records to BigQuery...[/cyan]"
        )

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema=[
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("lat", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("lon", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("actual_temp", "FLOAT64", mode="REQUIRED"),
            ],
        )

        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()

        console.print(f"[green]✓[/green] Loaded {len(df):,} ground truth records")

        return len(df)

    def store_evaluation_metrics(
        self,
        evaluation_date: datetime,
        metrics: dict[str, Any],
        eval_type: str,
    ) -> None:
        """Store computed evaluation metrics.

        Parameters
        ----------
        evaluation_date : datetime
            Date when the evaluation was computed
        metrics : dict[str, Any]
            Dictionary containing RMSE, MAE, and other metrics by horizon
        eval_type : str
            Type of evaluation: "static" or "dynamic"
        """
        table_id = f"{self.project_id}.{self.dataset_id}.evaluation_metrics"

        # Create DataFrame with single row
        df = pd.DataFrame(
            [
                {
                    "evaluation_date": evaluation_date,
                    "eval_type": eval_type,
                    "metrics": json.dumps(metrics),
                }
            ]
        )

        df["evaluation_date"] = pd.to_datetime(df["evaluation_date"])

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema=[
                bigquery.SchemaField("evaluation_date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("eval_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("metrics", "STRING", mode="REQUIRED"),
            ],
        )

        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()

        console.print(f"[green]✓[/green] Stored {eval_type} evaluation metrics")

    def compute_metrics_for_period(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Compute RMSE and MAE metrics for a date range using SQL aggregation.

        This is much faster than fetching data and computing in Python.

        Parameters
        ----------
        start_date : datetime
            Start of the range (inclusive)
        end_date : datetime
            End of the range (inclusive)

        Returns
        -------
        dict[str, Any]
            Computed metrics with overall and per-horizon breakdowns
        """
        query = f"""
        WITH matched AS (
            SELECT
                p.horizon_hours,
                p.predicted_temp,
                g.actual_temp
            FROM `{self.project_id}.{self.dataset_id}.predictions` p
            JOIN `{self.project_id}.{self.dataset_id}.ground_truth` g
                ON p.forecast_time = g.timestamp
                AND ROUND(p.lat, 2) = ROUND(g.lat, 2)
                AND ROUND(p.lon, 2) = ROUND(g.lon, 2)
            WHERE p.run_timestamp BETWEEN @start_date AND @end_date
        )
        SELECT
            horizon_hours,
            SQRT(AVG(POW(predicted_temp - actual_temp, 2))) AS rmse,
            AVG(ABS(predicted_temp - actual_temp)) AS mae,
            COUNT(*) AS sample_count
        FROM matched
        GROUP BY horizon_hours
        ORDER BY horizon_hours
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
                bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
            ]
        )

        console.print(
            f"[cyan]Computing metrics for {start_date} to {end_date}...[/cyan]"
        )
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        # Parse results
        by_horizon = {}
        total_samples = 0
        sum_squared_errors = 0.0
        sum_abs_errors = 0.0

        for row in results:
            horizon = int(row["horizon_hours"])
            rmse = float(row["rmse"])
            mae = float(row["mae"])
            count = int(row["sample_count"])

            by_horizon[horizon] = {
                "rmse": rmse,
                "mae": mae,
                "sample_count": count,
            }

            # Accumulate for overall metrics
            total_samples += count
            sum_squared_errors += (rmse**2) * count
            sum_abs_errors += mae * count

        if total_samples == 0:
            return {
                "overall_rmse": 0.0,
                "overall_mae": 0.0,
                "total_samples": 0,
                "by_horizon": {},
            }

        overall_rmse = (sum_squared_errors / total_samples) ** 0.5
        overall_mae = sum_abs_errors / total_samples

        console.print(
            f"[green]✓[/green] Computed metrics over {total_samples:,} samples "
            f"(RMSE: {overall_rmse:.3f}°C, MAE: {overall_mae:.3f}°C)"
        )

        return {
            "overall_rmse": overall_rmse,
            "overall_mae": overall_mae,
            "total_samples": total_samples,
            "by_horizon": by_horizon,
        }

    def get_latest_evaluation_metrics(self, eval_type: str) -> dict[str, Any] | None:
        """Get the most recent evaluation metrics.

        Parameters
        ----------
        eval_type : str
            Type of evaluation: "static" or "dynamic"

        Returns
        -------
        dict[str, Any] | None
            Latest evaluation metrics or None if not found
        """
        query = f"""
        SELECT evaluation_date, metrics
        FROM `{self.project_id}.{self.dataset_id}.evaluation_metrics`
        WHERE eval_type = @eval_type
        ORDER BY evaluation_date DESC
        LIMIT 1
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("eval_type", "STRING", eval_type),
            ]
        )

        query_job = self.client.query(query, job_config=job_config)
        results = list(query_job.result())

        if not results:
            return None

        row = results[0]
        return {
            "evaluation_date": row["evaluation_date"],
            "metrics": json.loads(row["metrics"]),
        }

    def get_evaluation_metrics_history(
        self, eval_type: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get historical evaluation metrics.

        Parameters
        ----------
        eval_type : str
            Type of evaluation: "static" or "dynamic"
        limit : int
            Maximum number of records to return

        Returns
        -------
        list[dict[str, Any]]
            List of evaluation metric dictionaries
        """
        query = f"""
        SELECT evaluation_date, metrics
        FROM `{self.project_id}.{self.dataset_id}.evaluation_metrics`
        WHERE eval_type = @eval_type
        ORDER BY evaluation_date DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("eval_type", "STRING", eval_type),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        return [
            {
                "evaluation_date": row["evaluation_date"],
                "metrics": json.loads(row["metrics"]),
            }
            for row in results
        ]
