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
        console.print("[yellow]DEBUG: Query parameters:[/yellow]")
        console.print(f"  start_date: {start_date}")
        console.print(f"  end_date: {end_date}")
        console.print(f"  project: {self.project_id}")
        console.print(f"  dataset: {self.dataset_id}")

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        console.print(
            "[yellow]DEBUG: BigQuery job completed, parsing results...[/yellow]"
        )

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

            console.print(
                f"[yellow]DEBUG: Horizon {horizon}h - "
                f"RMSE={rmse:.4f}°C, MAE={mae:.4f}°C, samples={count:,}[/yellow]"
            )

            by_horizon[horizon] = {
                "rmse": rmse,
                "mae": mae,
                "sample_count": count,
            }

            # Accumulate for overall metrics
            total_samples += count
            sum_squared_errors += (rmse**2) * count
            sum_abs_errors += mae * count

            console.print(
                f"[yellow]DEBUG: Accumulating - "
                f"total_samples={total_samples:,}, "
                f"sum_squared_errors={sum_squared_errors:.4f}, "
                f"sum_abs_errors={sum_abs_errors:.4f}[/yellow]"
            )

        if total_samples == 0:
            console.print("[red]DEBUG: No samples found, returning zeros[/red]")
            return {
                "overall_rmse": 0.0,
                "overall_mae": 0.0,
                "total_samples": 0,
                "by_horizon": {},
            }

        overall_rmse = (sum_squared_errors / total_samples) ** 0.5
        overall_mae = sum_abs_errors / total_samples

        console.print("[yellow]DEBUG: Overall computation:[/yellow]")
        console.print(
            f"  Overall RMSE = sqrt({sum_squared_errors:.4f} / {total_samples:,}) = {overall_rmse:.4f}°C"
        )
        console.print(
            f"  Overall MAE = {sum_abs_errors:.4f} / {total_samples:,} = {overall_mae:.4f}°C"
        )

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

    def get_latest_predictions(self, limit: int = 1000) -> pd.DataFrame:
        """Get the most recent prediction run from BigQuery.

        Parameters
        ----------
        limit : int
            Maximum number of predictions to return

        Returns
        -------
        pd.DataFrame
            Latest predictions with columns: forecast_time, horizon_hours,
            lat, lon, predicted_temp, run_timestamp
        """
        query = f"""
        SELECT
            run_timestamp,
            forecast_time,
            horizon_hours,
            lat,
            lon,
            predicted_temp
        FROM `{self.project_id}.{self.dataset_id}.predictions`
        WHERE run_timestamp = (
            SELECT MAX(run_timestamp)
            FROM `{self.project_id}.{self.dataset_id}.predictions`
        )
        ORDER BY forecast_time, horizon_hours, lat, lon
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        query_job = self.client.query(query, job_config=job_config)
        return query_job.result().to_dataframe()

    def log_forecast_run(
        self,
        run_timestamp: datetime,
        status: str,
        duration_seconds: float,
        records_generated: int,
        noaa_data_timestamp: datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        """Log a forecast run to the forecast_runs table.

        Parameters
        ----------
        run_timestamp : datetime
            Timestamp when the forecast run started
        status : str
            Status: "success" or "error"
        duration_seconds : float
            Duration of the forecast run in seconds
        records_generated : int
            Number of prediction records generated
        noaa_data_timestamp : datetime | None
            Most recent NOAA data timestamp used for this forecast
        error_message : str | None
            Error message if status is "error"
        """
        table_id = f"{self.project_id}.{self.dataset_id}.forecast_runs"

        df = pd.DataFrame(
            [
                {
                    "run_timestamp": run_timestamp,
                    "status": status,
                    "duration_seconds": duration_seconds,
                    "records_generated": records_generated,
                    "noaa_data_timestamp": noaa_data_timestamp,
                    "error_message": error_message,
                }
            ]
        )

        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
        if noaa_data_timestamp is not None:
            df["noaa_data_timestamp"] = pd.to_datetime(df["noaa_data_timestamp"])

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema=[
                bigquery.SchemaField("run_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("duration_seconds", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("records_generated", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField(
                    "noaa_data_timestamp", "TIMESTAMP", mode="NULLABLE"
                ),
                bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
            ],
        )

        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()

    def get_forecast_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent forecast run logs.

        Parameters
        ----------
        limit : int
            Maximum number of logs to return

        Returns
        -------
        list[dict[str, Any]]
            List of forecast run logs ordered by run_timestamp descending

        Raises
        ------
        Exception
            If the forecast_runs table doesn't exist yet
        """
        query = f"""
        SELECT
            run_timestamp,
            status,
            duration_seconds,
            records_generated,
            noaa_data_timestamp,
            error_message
        FROM `{self.project_id}.{self.dataset_id}.forecast_runs`
        ORDER BY run_timestamp DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()

            return [
                {
                    "run_timestamp": row["run_timestamp"].isoformat(),
                    "status": row["status"],
                    "duration_seconds": float(row["duration_seconds"]),
                    "records_generated": int(row["records_generated"]),
                    "noaa_data_timestamp": (
                        row["noaa_data_timestamp"].isoformat()
                        if row["noaa_data_timestamp"]
                        else None
                    ),
                    "error_message": row["error_message"],
                }
                for row in results
            ]
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "table" in error_msg:
                raise Exception(
                    f"forecast_runs table not found. Run: ./bigquery/setup.sh {self.project_id}"
                ) from e
            raise

    def get_last_forecast_timestamp(self) -> str | None:
        """Get just the timestamp of the most recent forecast run (lightweight).

        This method is optimized for minimal BigQuery scanning by only reading
        the partitioned column. Use this for polling/checking if new data exists.

        Returns
        -------
        str | None
            ISO format timestamp of last run, or None if no predictions exist
        """
        query = f"""
        SELECT MAX(run_timestamp) as last_run
        FROM `{self.project_id}.{self.dataset_id}.predictions`
        """

        query_job = self.client.query(query)
        results = list(query_job.result())

        if not results or results[0]["last_run"] is None:
            return None

        return results[0]["last_run"].isoformat()

    def compute_monthly_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Compute error metrics grouped by month for error analysis.

        Parameters
        ----------
        start_date : datetime
            Start of the range (inclusive)
        end_date : datetime
            End of the range (inclusive)

        Returns
        -------
        dict[str, Any]
            Monthly metrics with overall and per-horizon breakdowns
        """
        query = f"""
        WITH matched AS (
            SELECT
                p.forecast_time,
                p.horizon_hours,
                p.predicted_temp,
                g.actual_temp
            FROM `{self.project_id}.{self.dataset_id}.predictions` p
            JOIN `{self.project_id}.{self.dataset_id}.ground_truth` g
                ON p.forecast_time = g.timestamp
                AND ROUND(p.lat, 2) = ROUND(g.lat, 2)
                AND ROUND(p.lon, 2) = ROUND(g.lon, 2)
            WHERE p.run_timestamp BETWEEN @start_date AND @end_date
        ),
        monthly AS (
            SELECT
                FORMAT_TIMESTAMP('%Y-%m', forecast_time) AS month,
                horizon_hours,
                predicted_temp,
                actual_temp
            FROM matched
            WHERE forecast_time >= @start_date
              AND forecast_time <= @end_date
        )
        SELECT
            month,
            horizon_hours,
            SQRT(AVG(POW(predicted_temp - actual_temp, 2))) AS rmse,
            AVG(ABS(predicted_temp - actual_temp)) AS mae,
            COUNT(*) AS sample_count
        FROM monthly
        GROUP BY month, horizon_hours
        ORDER BY month, horizon_hours
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
                bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
            ]
        )

        console.print(
            f"[cyan]Computing monthly metrics for {start_date} to {end_date}...[/cyan]"
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        # Parse results into nested structure
        by_month: dict[str, dict[str, Any]] = {}

        for row in results:
            month = str(row["month"])
            horizon = int(row["horizon_hours"])
            rmse = float(row["rmse"])
            mae = float(row["mae"])
            count = int(row["sample_count"])

            if month not in by_month:
                by_month[month] = {"by_horizon": {}, "samples": 0}

            by_month[month]["by_horizon"][horizon] = {
                "rmse": rmse,
                "mae": mae,
                "sample_count": count,
            }
            by_month[month]["samples"] += count

        # Compute overall metrics per month (across all horizons)
        for _month, month_data in by_month.items():
            total_samples = 0
            sum_squared_errors = 0.0
            sum_abs_errors = 0.0

            for horizon_data in month_data["by_horizon"].values():
                count = horizon_data["sample_count"]
                rmse = horizon_data["rmse"]
                mae = horizon_data["mae"]

                total_samples += count
                sum_squared_errors += (rmse**2) * count
                sum_abs_errors += mae * count

            if total_samples > 0:
                month_data["overall_rmse"] = (sum_squared_errors / total_samples) ** 0.5
                month_data["overall_mae"] = sum_abs_errors / total_samples
            else:
                month_data["overall_rmse"] = 0.0
                month_data["overall_mae"] = 0.0

        console.print(
            f"[green]✓[/green] Computed monthly metrics for {len(by_month)} months"
        )

        return {"by_month": by_month}

    def get_last_forecast_run_info(self) -> dict[str, Any] | None:
        """Get information about the last forecast run.

        Returns
        -------
        dict[str, Any] | None
            Dictionary with run_timestamp, forecast_time_range, and count,
            or None if no predictions exist
        """
        query = f"""
        SELECT
            MAX(run_timestamp) as last_run,
            MIN(forecast_time) as earliest_forecast,
            MAX(forecast_time) as latest_forecast,
            COUNT(*) as prediction_count
        FROM `{self.project_id}.{self.dataset_id}.predictions`
        WHERE run_timestamp = (
            SELECT MAX(run_timestamp)
            FROM `{self.project_id}.{self.dataset_id}.predictions`
        )
        """

        query_job = self.client.query(query)
        results = list(query_job.result())

        if not results or results[0]["last_run"] is None:
            return None

        row = results[0]
        return {
            "run_timestamp": row["last_run"].isoformat(),
            "earliest_forecast": row["earliest_forecast"].isoformat(),
            "latest_forecast": row["latest_forecast"].isoformat(),
            "prediction_count": int(row["prediction_count"]),
        }

    def get_forecast_times_needing_ground_truth(
        self, lookback_hours: int = 72
    ) -> list[datetime]:
        """Get distinct forecast_times from predictions that need ground truth.

        Finds forecast_times that:
        1. Are in the past (already occurred)
        2. Don't have matching ground truth data
        3. Are within the lookback window

        Parameters
        ----------
        lookback_hours : int
            How far back to look for missing ground truth (default: 72 hours)

        Returns
        -------
        list[datetime]
            List of forecast_times that need ground truth data
        """
        query = f"""
        WITH past_forecasts AS (
            SELECT DISTINCT forecast_time
            FROM `{self.project_id}.{self.dataset_id}.predictions`
            WHERE forecast_time < CURRENT_TIMESTAMP()
              AND forecast_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_hours} HOUR)
        ),
        existing_ground_truth AS (
            SELECT DISTINCT timestamp
            FROM `{self.project_id}.{self.dataset_id}.ground_truth`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_hours} HOUR)
        )
        SELECT pf.forecast_time
        FROM past_forecasts pf
        LEFT JOIN existing_ground_truth egt
            ON pf.forecast_time = egt.timestamp
        WHERE egt.timestamp IS NULL
        ORDER BY pf.forecast_time
        """

        console.print(
            f"[cyan]Checking for forecast times needing ground truth "
            f"(last {lookback_hours}h)...[/cyan]"
        )

        query_job = self.client.query(query)
        results = list(query_job.result())

        timestamps = [row["forecast_time"] for row in results]

        if timestamps:
            console.print(
                f"[yellow]Found {len(timestamps)} forecast times needing ground truth[/yellow]"
            )
        else:
            console.print("[green]✓[/green] All forecast times have ground truth")

        return timestamps

    def get_ground_truth_coverage(self) -> dict[str, Any]:
        """Get statistics about ground truth coverage for predictions.

        Returns
        -------
        dict[str, Any]
            Coverage statistics including total predictions, matched, and missing
        """
        query = f"""
        WITH prediction_times AS (
            SELECT
                COUNT(DISTINCT forecast_time) as total_forecast_times,
                MIN(forecast_time) as earliest,
                MAX(forecast_time) as latest
            FROM `{self.project_id}.{self.dataset_id}.predictions`
            WHERE forecast_time < CURRENT_TIMESTAMP()
        ),
        ground_truth_times AS (
            SELECT COUNT(DISTINCT timestamp) as total_ground_truth
            FROM `{self.project_id}.{self.dataset_id}.ground_truth`
        ),
        matched AS (
            SELECT COUNT(DISTINCT p.forecast_time) as matched_times
            FROM `{self.project_id}.{self.dataset_id}.predictions` p
            INNER JOIN `{self.project_id}.{self.dataset_id}.ground_truth` g
                ON p.forecast_time = g.timestamp
            WHERE p.forecast_time < CURRENT_TIMESTAMP()
        )
        SELECT
            pt.total_forecast_times,
            pt.earliest,
            pt.latest,
            gt.total_ground_truth,
            m.matched_times
        FROM prediction_times pt
        CROSS JOIN ground_truth_times gt
        CROSS JOIN matched m
        """

        query_job = self.client.query(query)
        results = list(query_job.result())

        if not results:
            return {
                "total_forecast_times": 0,
                "total_ground_truth": 0,
                "matched_times": 0,
                "coverage_pct": 0.0,
            }

        row = results[0]
        total = row["total_forecast_times"] or 0
        matched = row["matched_times"] or 0

        return {
            "total_forecast_times": total,
            "earliest_forecast": (
                row["earliest"].isoformat() if row["earliest"] else None
            ),
            "latest_forecast": row["latest"].isoformat() if row["latest"] else None,
            "total_ground_truth": row["total_ground_truth"] or 0,
            "matched_times": matched,
            "coverage_pct": (matched / total * 100) if total > 0 else 0.0,
        }
