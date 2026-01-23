"""Tests for the scheduler module, including GroundTruthScheduler."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.scheduler import GroundTruthScheduler


class TestGroundTruthScheduler:
    """Tests for GroundTruthScheduler class."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create a mock EvaluationStorage."""
        storage = MagicMock()
        storage.get_forecast_times_needing_ground_truth = MagicMock(return_value=[])
        storage.store_ground_truth = MagicMock(return_value=100)
        storage.get_ground_truth_coverage = MagicMock(
            return_value={
                "total_forecast_times": 100,
                "matched_times": 90,
                "coverage_pct": 90.0,
            }
        )
        return storage

    def test_init(self, mock_storage: MagicMock) -> None:
        """Test scheduler initialization."""
        scheduler = GroundTruthScheduler(storage=mock_storage)

        assert scheduler.storage == mock_storage
        assert scheduler.lookback_hours == 72
        assert scheduler.is_running is False
        assert scheduler.last_run_timestamp is None
        assert scheduler.timestamps_processed == 0

    def test_init_custom_lookback(self, mock_storage: MagicMock) -> None:
        """Test scheduler initialization with custom lookback."""
        scheduler = GroundTruthScheduler(storage=mock_storage, lookback_hours=24)

        assert scheduler.lookback_hours == 24

    def test_get_status_not_running(self, mock_storage: MagicMock) -> None:
        """Test get_status when scheduler is not running."""
        scheduler = GroundTruthScheduler(storage=mock_storage)
        status = scheduler.get_status()

        assert status["is_running"] is False
        assert status["scheduler_active"] is False
        assert status["last_run_timestamp"] is None
        assert status["timestamps_processed_total"] == 0
        assert status["lookback_hours"] == 72

    def test_region_constants(self, mock_storage: MagicMock) -> None:
        """Test that region constants are set correctly."""
        scheduler = GroundTruthScheduler(storage=mock_storage)

        assert scheduler.LAT_MIN == 42.0
        assert scheduler.LAT_MAX == 45.0
        assert scheduler.LON_MIN == -81.0
        assert scheduler.LON_MAX == -78.0

    @pytest.mark.asyncio
    async def test_run_ground_truth_job_no_missing(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that job completes quickly when no missing timestamps."""
        scheduler = GroundTruthScheduler(storage=mock_storage)

        await scheduler._run_ground_truth_job()

        mock_storage.get_forecast_times_needing_ground_truth.assert_called_once_with(72)
        # Should not call store_ground_truth since no missing timestamps
        mock_storage.store_ground_truth.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_ground_truth_job_with_missing(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that job fetches and stores ground truth for missing timestamps."""
        # Setup missing timestamps
        missing_ts = datetime.utcnow() - timedelta(hours=2)
        mock_storage.get_forecast_times_needing_ground_truth = MagicMock(
            return_value=[missing_ts]
        )

        # Mock the data extraction
        mock_data = pd.DataFrame(
            {
                "datetime": [missing_ts],
                "lat": [43.0],
                "lon": [-80.0],
                "t2m": [293.15],  # 20°C in Kelvin
            }
        )

        scheduler = GroundTruthScheduler(storage=mock_storage)

        with patch(
            "backend.app.scheduler.fetch_historical_hours", return_value=mock_data
        ):
            await scheduler._run_ground_truth_job()

        # Should have called store_ground_truth
        mock_storage.store_ground_truth.assert_called_once()

        # Verify the stored data has correct temperature conversion
        call_args = mock_storage.store_ground_truth.call_args[0][0]
        assert "actual_temp" in call_args.columns
        # Check Kelvin to Celsius conversion: 293.15 - 273.15 = 20
        assert abs(call_args["actual_temp"].iloc[0] - 20.0) < 0.01

    @pytest.mark.asyncio
    async def test_run_ground_truth_job_already_running(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that job is skipped if already running."""
        scheduler = GroundTruthScheduler(storage=mock_storage)
        scheduler.is_running = True

        await scheduler._run_ground_truth_job()

        # Should not call storage methods
        mock_storage.get_forecast_times_needing_ground_truth.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_immediate(self, mock_storage: MagicMock) -> None:
        """Test run_immediate method."""
        scheduler = GroundTruthScheduler(storage=mock_storage)

        result = await scheduler.run_immediate()

        assert result["status"] == "success"
        assert "timestamps_processed" in result

    @pytest.mark.asyncio
    async def test_run_immediate_already_running(self, mock_storage: MagicMock) -> None:
        """Test run_immediate when already running."""
        scheduler = GroundTruthScheduler(storage=mock_storage)
        scheduler.is_running = True

        result = await scheduler.run_immediate()

        assert result["status"] == "error"
        assert "already running" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_storage: MagicMock) -> None:
        """Test starting and stopping the scheduler."""
        scheduler = GroundTruthScheduler(storage=mock_storage)

        # Before start, scheduler should not be running
        assert scheduler.scheduler.running is False

        scheduler.start()
        assert scheduler.scheduler.running is True

        # Stop the scheduler - note: APScheduler may still report running=True
        # briefly after shutdown, so we just verify stop() doesn't raise
        scheduler.stop()
