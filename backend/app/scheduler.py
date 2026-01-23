"""Scheduler for automated forecasting and evaluation tasks.

This module manages periodic execution of forecasting and evaluation tasks
using APScheduler. It ensures forecasts run when new NOAA data is available
and evaluations run daily.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from gaca_ews.core.data_extraction import fetch_historical_hours
from gaca_ews.core.inference import InferenceEngine
from gaca_ews.core.logger import get_logger
from gaca_ews.evaluation.storage import EvaluationStorage


logger = get_logger()


class ForecastScheduler:
    """Manages automated forecast execution on hourly schedule."""

    def __init__(
        self,
        engine: InferenceEngine,
        storage: EvaluationStorage | None = None,
        output_dir: Path = Path("forecasts"),
    ) -> None:
        """Initialize the forecast scheduler.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine for running forecasts
        storage : EvaluationStorage | None
            BigQuery storage client for storing predictions
        output_dir : Path
            Directory to save forecast outputs
        """
        self.engine = engine
        self.storage = storage
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.last_run_timestamp: datetime | None = None
        self.last_data_timestamp: datetime | None = None
        self.is_running = False

        logger.info("ForecastScheduler initialized")

    def start(self) -> None:
        """Start the scheduler with hourly forecast execution.

        Runs every hour at 15 minutes past the hour to allow NOAA data
        to be available (typically published ~5-10 minutes after the hour).
        """
        # Run every hour at :15 past the hour
        self.scheduler.add_job(
            self._run_forecast_job,
            trigger=CronTrigger(minute=15, timezone="UTC"),
            id="hourly_forecast",
            name="Hourly Temperature Forecast",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
        )

        self.scheduler.start()
        logger.info("Forecast scheduler started - will run hourly at :15")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Forecast scheduler stopped")

    async def _run_forecast_job(self) -> None:
        """Execute a forecast run and store to BigQuery."""
        if self.is_running:
            logger.warning("Forecast job already running, skipping...")
            return

        self.is_running = True
        job_start = datetime.utcnow()
        status = "error"
        records_generated = 0
        error_message = None
        noaa_data_timestamp = None

        try:
            logger.info("=" * 80)
            logger.info(f"Starting scheduled forecast run at {job_start.isoformat()}")
            logger.info("=" * 80)

            # Step 1: Fetch latest data (run in thread to avoid blocking)
            logger.info("Fetching latest NOAA data...")
            data, latest_ts = await asyncio.to_thread(self.engine.fetch_data)
            noaa_data_timestamp = latest_ts

            # Check if this is new data
            if self._is_duplicate_run(latest_ts):
                logger.info(
                    f"Data timestamp {latest_ts.isoformat()} already processed. "
                    "Skipping duplicate run."
                )
                return

            logger.info(f"Latest data timestamp: {latest_ts.isoformat()}")

            # Step 2: Preprocess (run in thread to avoid blocking)
            logger.info("Preprocessing features...")
            X = await asyncio.to_thread(self.engine.preprocess, data)

            # Step 3: Run inference (run in thread to avoid blocking)
            logger.info("Running model inference...")
            predictions = await asyncio.to_thread(self.engine.predict, X)

            # Step 4: Save predictions (run in thread to avoid blocking)
            output_file = (
                self.output_dir / f"predictions_{job_start.strftime('%Y%m%d_%H%M')}.csv"
            )
            logger.info(f"Saving predictions to {output_file}...")

            await asyncio.to_thread(
                self.engine.save_predictions, predictions, latest_ts, str(output_file)
            )

            # Step 5: Store to BigQuery if available
            if self.storage:
                logger.info("Storing predictions to BigQuery...")
                df = await asyncio.to_thread(pd.read_csv, output_file)
                rows_loaded = await asyncio.to_thread(
                    self.storage.store_predictions, df, job_start
                )
                records_generated = rows_loaded
                logger.info(f"Stored {rows_loaded:,} predictions to BigQuery")
            else:
                logger.warning("BigQuery storage not available, skipping upload")
                records_generated = predictions.size

            # Update tracking
            self.last_run_timestamp = job_start
            self.last_data_timestamp = latest_ts
            status = "success"

            job_duration = (datetime.utcnow() - job_start).total_seconds()
            logger.info("=" * 80)
            logger.info(f"Forecast job completed successfully in {job_duration:.1f}s")
            logger.info("=" * 80)

        except Exception as e:
            error_message = str(e)
            logger.error(f"Forecast job failed: {e}", exc_info=True)
            raise
        finally:
            self.is_running = False

            # Log the run to BigQuery (run in thread to avoid blocking)
            if self.storage:
                try:
                    job_duration = (datetime.utcnow() - job_start).total_seconds()
                    await asyncio.to_thread(
                        self.storage.log_forecast_run,
                        run_timestamp=job_start,
                        status=status,
                        duration_seconds=job_duration,
                        records_generated=records_generated,
                        noaa_data_timestamp=noaa_data_timestamp,
                        error_message=error_message,
                    )
                except Exception as log_error:
                    logger.error(f"Failed to log forecast run: {log_error}")

    def _is_duplicate_run(self, data_timestamp: datetime) -> bool:
        """Check if we've already processed this data timestamp.

        Parameters
        ----------
        data_timestamp : datetime
            The timestamp of the fetched data

        Returns
        -------
        bool
            True if this data has already been processed
        """
        if self.last_data_timestamp is None:
            return False

        # Consider it a duplicate if we've seen this exact timestamp
        return data_timestamp <= self.last_data_timestamp

    def get_status(self) -> dict[str, Any]:
        """Get current scheduler status.

        Returns
        -------
        dict[str, Any]
            Status information including last run time and next scheduled run
        """
        next_run = None
        job = self.scheduler.get_job("hourly_forecast")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

        return {
            "is_running": self.is_running,
            "scheduler_active": self.scheduler.running,
            "last_run_timestamp": (
                self.last_run_timestamp.isoformat() if self.last_run_timestamp else None
            ),
            "last_data_timestamp": (
                self.last_data_timestamp.isoformat()
                if self.last_data_timestamp
                else None
            ),
            "next_scheduled_run": next_run,
        }


class EvaluationScheduler:
    """Manages automated daily evaluation computation."""

    def __init__(self, storage: EvaluationStorage) -> None:
        """Initialize the evaluation scheduler.

        Parameters
        ----------
        storage : EvaluationStorage
            BigQuery storage client for computing metrics
        """
        self.storage = storage
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.last_eval_date: datetime | None = None
        self.is_running = False

        logger.info("EvaluationScheduler initialized")

    def start(self) -> None:
        """Start the scheduler with daily evaluation at 00:30 UTC."""
        # Run daily at 00:30 UTC to compute metrics for rolling 30-day window
        self.scheduler.add_job(
            self._run_evaluation_job,
            trigger=CronTrigger(hour=0, minute=30, timezone="UTC"),
            id="daily_evaluation",
            name="Daily Rolling Evaluation",
            replace_existing=True,
            max_instances=1,
        )

        self.scheduler.start()
        logger.info("Evaluation scheduler started - will run daily at 00:30 UTC")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Evaluation scheduler stopped")

    async def _run_evaluation_job(self) -> None:
        """Execute daily evaluation computation."""
        if self.is_running:
            logger.warning("Evaluation job already running, skipping...")
            return

        self.is_running = True
        job_start = datetime.utcnow()

        try:
            logger.info("=" * 80)
            logger.info(f"Starting scheduled evaluation at {job_start.isoformat()}")
            logger.info("=" * 80)

            # Compute rolling 30-day metrics
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)

            logger.info(
                f"Computing metrics for {start_date.date()} to {end_date.date()}"
            )

            # Run metrics computation in thread to avoid blocking
            metrics = await asyncio.to_thread(
                self.storage.compute_metrics_for_period, start_date, end_date
            )

            # Store computed metrics (run in thread to avoid blocking)
            await asyncio.to_thread(
                self.storage.store_evaluation_metrics,
                evaluation_date=job_start,
                metrics=metrics,
                eval_type="dynamic",
            )

            self.last_eval_date = job_start

            job_duration = (datetime.utcnow() - job_start).total_seconds()
            logger.info("=" * 80)
            logger.info(
                f"Evaluation job completed in {job_duration:.1f}s "
                f"({metrics['total_samples']:,} samples)"
            )
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Evaluation job failed: {e}", exc_info=True)
            raise
        finally:
            self.is_running = False

    def get_status(self) -> dict[str, Any]:
        """Get current scheduler status.

        Returns
        -------
        dict[str, Any]
            Status information including last evaluation time
        """
        next_run = None
        job = self.scheduler.get_job("daily_evaluation")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

        return {
            "is_running": self.is_running,
            "scheduler_active": self.scheduler.running,
            "last_eval_date": (
                self.last_eval_date.isoformat() if self.last_eval_date else None
            ),
            "next_scheduled_run": next_run,
        }


class GroundTruthScheduler:
    """Manages automated ground truth data collection for evaluation.

    This scheduler fetches actual observed temperatures from NOAA for
    forecast times that have already passed, enabling evaluation of
    prediction accuracy.
    """

    # Region boundaries (must match trained model)
    LAT_MIN = 42.0
    LAT_MAX = 45.0
    LON_MIN = -81.0
    LON_MAX = -78.0

    def __init__(
        self,
        storage: EvaluationStorage,
        lookback_hours: int = 72,
    ) -> None:
        """Initialize the ground truth scheduler.

        Parameters
        ----------
        storage : EvaluationStorage
            BigQuery storage client for storing ground truth
        lookback_hours : int
            How far back to look for missing ground truth (default: 72 hours)
        """
        self.storage = storage
        self.lookback_hours = lookback_hours
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.last_run_timestamp: datetime | None = None
        self.is_running = False
        self.timestamps_processed = 0

        logger.info("GroundTruthScheduler initialized")

    def start(self) -> None:
        """Start the scheduler with hourly ground truth collection.

        Runs every hour at :45 past the hour, after forecasts have run
        and NOAA data for past hours is available.
        """
        # Run every hour at :45 past the hour
        self.scheduler.add_job(
            self._run_ground_truth_job,
            trigger=CronTrigger(minute=45, timezone="UTC"),
            id="hourly_ground_truth",
            name="Hourly Ground Truth Collection",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
        )

        self.scheduler.start()
        logger.info("Ground truth scheduler started - will run hourly at :45")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Ground truth scheduler stopped")

    async def _run_ground_truth_job(self) -> None:
        """Execute ground truth collection for missing timestamps."""
        if self.is_running:
            logger.warning("Ground truth job already running, skipping...")
            return

        self.is_running = True
        job_start = datetime.utcnow()
        timestamps_filled = 0

        try:
            logger.info("=" * 80)
            logger.info(f"Starting ground truth collection at {job_start.isoformat()}")
            logger.info("=" * 80)

            # Get forecast times that need ground truth
            logger.info("Checking for forecast times needing ground truth...")
            missing_timestamps = await asyncio.to_thread(
                self.storage.get_forecast_times_needing_ground_truth,
                self.lookback_hours,
            )

            if not missing_timestamps:
                logger.info("No missing ground truth - all forecast times covered")
                return

            logger.info(
                f"Found {len(missing_timestamps)} timestamps needing ground truth"
            )

            # Fetch and store ground truth for each missing timestamp
            for ts in missing_timestamps:
                try:
                    logger.info(f"Fetching ground truth for {ts.isoformat()}...")

                    # Fetch NOAA data for this specific hour
                    # We only need 1 hour of data for the specific timestamp
                    data = await asyncio.to_thread(
                        fetch_historical_hours,
                        target_datetime=ts,
                        hours_back=1,
                        lat_min=self.LAT_MIN,
                        lat_max=self.LAT_MAX,
                        lon_min=self.LON_MIN,
                        lon_max=self.LON_MAX,
                    )

                    if data is None or data.empty:
                        logger.warning(f"No NOAA data available for {ts.isoformat()}")
                        continue

                    # Extract ground truth (t2m in Celsius)
                    # The data has 'datetime' column from fetch_historical_hours
                    ground_truth_df = data[["datetime", "lat", "lon", "t2m"]].copy()
                    ground_truth_df = ground_truth_df.rename(
                        columns={"datetime": "timestamp", "t2m": "actual_temp"}
                    )

                    # Convert from Kelvin to Celsius
                    ground_truth_df["actual_temp"] = (
                        ground_truth_df["actual_temp"] - 273.15
                    )

                    # Store to BigQuery
                    rows_loaded = await asyncio.to_thread(
                        self.storage.store_ground_truth, ground_truth_df
                    )

                    logger.info(
                        f"Stored {rows_loaded} ground truth records for {ts.isoformat()}"
                    )
                    timestamps_filled += 1

                except Exception as e:
                    logger.error(
                        f"Failed to fetch ground truth for {ts.isoformat()}: {e}"
                    )
                    continue

            self.last_run_timestamp = job_start
            self.timestamps_processed += timestamps_filled

            job_duration = (datetime.utcnow() - job_start).total_seconds()
            logger.info("=" * 80)
            logger.info(
                f"Ground truth job completed in {job_duration:.1f}s "
                f"({timestamps_filled} timestamps filled)"
            )
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Ground truth job failed: {e}", exc_info=True)
            raise
        finally:
            self.is_running = False

    async def run_immediate(self) -> dict[str, Any]:
        """Run ground truth collection immediately (for manual trigger).

        Returns
        -------
        dict[str, Any]
            Results of the collection run
        """
        if self.is_running:
            return {"status": "error", "message": "Job already running"}

        await self._run_ground_truth_job()

        return {
            "status": "success",
            "timestamps_processed": self.timestamps_processed,
            "last_run": (
                self.last_run_timestamp.isoformat() if self.last_run_timestamp else None
            ),
        }

    def get_status(self) -> dict[str, Any]:
        """Get current scheduler status.

        Returns
        -------
        dict[str, Any]
            Status information including last run and coverage stats
        """
        next_run = None
        job = self.scheduler.get_job("hourly_ground_truth")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

        return {
            "is_running": self.is_running,
            "scheduler_active": self.scheduler.running,
            "last_run_timestamp": (
                self.last_run_timestamp.isoformat() if self.last_run_timestamp else None
            ),
            "timestamps_processed_total": self.timestamps_processed,
            "next_scheduled_run": next_run,
            "lookback_hours": self.lookback_hours,
        }
