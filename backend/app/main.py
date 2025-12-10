"""FastAPI application for GACA Early Warning System.

This module provides a REST API for interacting with the GACA temperature
forecasting model. The system runs automated hourly forecasts and daily
evaluations using APScheduler.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.scheduler import EvaluationScheduler, ForecastScheduler
from gaca_ews.core.inference import InferenceEngine
from gaca_ews.core.logger import get_logger
from gaca_ews.evaluation.storage import EvaluationStorage


app = FastAPI(
    title="GACA Early Warning System API",
    description="Temperature forecasting API for Southwestern Ontario",
    version="0.1.0",
)

# Configure CORS for both local development and Cloud Run
# Get allowed origins from environment variable or use defaults
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [
    "http://localhost:3000",  # Local development
    "http://localhost:8080",  # Local backend
]

# Add origins from environment variable (for Cloud Run deployments)
if ALLOWED_ORIGINS_ENV:
    additional_origins = [
        origin.strip() for origin in ALLOWED_ORIGINS_ENV.split(",") if origin.strip()
    ]
    allowed_origins.extend(additional_origins)

# For Cloud Run, also allow origins that match the pattern
# https://*-HASH.REGION.run.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.run\.app",  # Allow all Cloud Run URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    forecast_time: str
    horizon_hours: int
    lat: float
    lon: float
    predicted_temp: float
    run_timestamp: str | None = None


class FeatureMetadata(BaseModel):
    """Metadata for a single input feature."""

    name: str
    label: str
    icon: str


class ModelInfo(BaseModel):
    """Model information response."""

    model_architecture: str
    num_nodes: int
    input_features: list[str]
    feature_metadata: list[FeatureMetadata]
    prediction_horizons: list[int]
    region: dict[str, float]
    status: str


@app.on_event("startup")
async def startup_event() -> None:
    """Load model and start schedulers on startup."""
    logger = get_logger()
    logger.setLevel(logging.INFO)

    logger.info("Initializing GACA Early Warning System...")

    # Initialize inference engine
    app.state.engine = InferenceEngine("config.yaml")
    app.state.engine.load_artifacts()
    logger.info("Model artifacts loaded successfully")

    # Initialize evaluation storage (only if BigQuery is available)
    try:
        app.state.eval_storage = EvaluationStorage()
        logger.info("BigQuery storage initialized successfully")
    except Exception as e:
        logger.warning(
            f"Failed to initialize BigQuery: {e}. Evaluation endpoints will be disabled."
        )
        app.state.eval_storage = None

    # Initialize schedulers (but don't start background jobs on Cloud Run)
    # Background APScheduler jobs get killed by Cloud Run when idle
    # Use Cloud Scheduler to call /scheduler/trigger-forecast instead
    forecast_output_dir = Path(os.getenv("FORECAST_OUTPUT_DIR", "forecasts"))
    app.state.forecast_scheduler = ForecastScheduler(
        engine=app.state.engine,
        storage=app.state.eval_storage,
        output_dir=forecast_output_dir,
    )

    # Only start background scheduler if explicitly enabled (for local dev)
    enable_background_scheduler = (
        os.getenv("ENABLE_BACKGROUND_SCHEDULER", "false").lower() == "true"
    )
    if enable_background_scheduler:
        app.state.forecast_scheduler.start()
        logger.info("Background forecast scheduler started (runs hourly at :15)")

        if app.state.eval_storage:
            app.state.eval_scheduler = EvaluationScheduler(
                storage=app.state.eval_storage
            )
            app.state.eval_scheduler.start()
            logger.info(
                "Background evaluation scheduler started (runs daily at 00:30 UTC)"
            )
    else:
        logger.info("Background schedulers DISABLED (Cloud Scheduler mode)")
        logger.info("Forecasts triggered via: POST /scheduler/trigger-forecast")
        logger.info(
            "Evaluations triggered via: POST /scheduler/trigger-evaluation (TODO)"
        )

    logger.info("=" * 80)
    logger.info("GACA Early Warning System is online")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Stop schedulers on shutdown."""
    logger = get_logger()
    logger.info("Shutting down GACA Early Warning System...")

    if (
        hasattr(app.state, "forecast_scheduler")
        and app.state.forecast_scheduler.scheduler.running
    ):
        app.state.forecast_scheduler.stop()
        logger.info("Forecast scheduler stopped")

    if (
        hasattr(app.state, "eval_scheduler")
        and app.state.eval_scheduler.scheduler.running
    ):
        app.state.eval_scheduler.stop()
        logger.info("Evaluation scheduler stopped")

    logger.info("Shutdown complete")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "GACA Early Warning System API",
        "version": "0.1.0",
        "status": "online",
    }


@app.get("/health")
async def health() -> dict[str, str | bool]:
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": hasattr(app.state, "engine")}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """Get information about the loaded model."""
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = app.state.engine.get_model_info()

    # Feature metadata mapping
    feature_info = {
        "t2m": {"label": "2m Temperature", "icon": "thermometer"},
        "d2m": {"label": "2m Dewpoint", "icon": "droplets"},
        "u10": {"label": "10m U-Wind", "icon": "wind"},
        "v10": {"label": "10m V-Wind", "icon": "compass"},
        "sp": {"label": "Surface Pressure", "icon": "gauge"},
        "orog": {"label": "Orography", "icon": "mountain"},
    }

    feature_metadata = [
        FeatureMetadata(
            name=feat,
            label=feature_info.get(feat, {}).get("label", feat),
            icon=feature_info.get(feat, {}).get("icon", "circle"),
        )
        for feat in info["input_features"]
    ]

    return ModelInfo(
        model_architecture=info["model_architecture"],
        num_nodes=info["num_nodes"],
        input_features=info["input_features"],
        feature_metadata=feature_metadata,
        prediction_horizons=info["prediction_horizons"],
        region=info["region"],
        status="loaded",
    )


@app.get("/forecasts/latest-timestamp")
async def get_latest_forecast_timestamp() -> dict[str, Any]:
    """Get just the timestamp of the most recent forecast (lightweight check).

    This endpoint is optimized for polling to check if new data is available
    without scanning full prediction data. Use this before calling
    /forecasts/latest to minimize BigQuery costs.

    Returns
    -------
    dict[str, Any]
        Dictionary with last_run_timestamp or null if no forecasts exist
    """
    if not hasattr(app.state, "eval_storage") or app.state.eval_storage is None:
        raise HTTPException(
            status_code=503,
            detail="BigQuery storage not available.",
        )

    try:
        # Run BigQuery call in thread to avoid blocking event loop
        timestamp = await asyncio.to_thread(
            app.state.eval_storage.get_last_forecast_timestamp
        )

        return {
            "last_run_timestamp": timestamp,
            "has_data": timestamp is not None,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check forecast timestamp: {str(e)}",
        ) from e


@app.get("/forecasts/latest", response_model=list[PredictionResponse])
async def get_latest_forecasts() -> list[PredictionResponse]:
    """Get the most recent forecast predictions from BigQuery.

    Returns
    -------
    list[PredictionResponse]
        Latest predictions with forecast times, locations, and temperatures
    """
    if not hasattr(app.state, "eval_storage") or app.state.eval_storage is None:
        raise HTTPException(
            status_code=503,
            detail="BigQuery storage not available. Cannot retrieve forecasts.",
        )

    try:
        # Run BigQuery call in thread to avoid blocking event loop
        df = await asyncio.to_thread(
            app.state.eval_storage.get_latest_predictions, limit=100000
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No forecasts found. Waiting for first automated run.",
            )

        # Convert to response models (run in thread as iterrows can be slow)
        def convert_to_responses() -> list[PredictionResponse]:
            return [
                PredictionResponse(
                    forecast_time=row["forecast_time"].isoformat(),
                    horizon_hours=int(row["horizon_hours"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    predicted_temp=float(row["predicted_temp"]),
                    run_timestamp=row["run_timestamp"].isoformat(),
                )
                for _, row in df.iterrows()
            ]

        return await asyncio.to_thread(convert_to_responses)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve forecasts: {str(e)}"
        ) from e


@app.get("/forecasts/status")
async def get_forecast_status() -> dict[str, Any]:
    """Get information about the last forecast run and next scheduled run.

    Returns
    -------
    dict[str, Any]
        Status information including timestamps and scheduler state
    """
    if not hasattr(app.state, "forecast_scheduler"):
        raise HTTPException(
            status_code=503, detail="Forecast scheduler not initialized"
        )

    scheduler_status = app.state.forecast_scheduler.get_status()

    # Get additional info from BigQuery if available (run in thread)
    forecast_info = None
    if hasattr(app.state, "eval_storage") and app.state.eval_storage:
        try:
            forecast_info = await asyncio.to_thread(
                app.state.eval_storage.get_last_forecast_run_info
            )
        except Exception as e:
            get_logger().warning(f"Failed to get forecast info from BigQuery: {e}")

    return {
        "scheduler": scheduler_status,
        "last_forecast": forecast_info,
    }


@app.get("/scheduler/status")
async def get_scheduler_status() -> dict[str, Any]:
    """Get status of all schedulers (forecast and evaluation).

    Returns
    -------
    dict[str, Any]
        Status of forecast and evaluation schedulers
    """
    status = {}

    if hasattr(app.state, "forecast_scheduler"):
        status["forecast"] = app.state.forecast_scheduler.get_status()
    else:
        status["forecast"] = {"error": "Forecast scheduler not initialized"}

    if hasattr(app.state, "eval_scheduler"):
        status["evaluation"] = app.state.eval_scheduler.get_status()
    else:
        status["evaluation"] = {"error": "Evaluation scheduler not initialized"}

    return status


@app.get("/evaluation/static")
async def get_static_evaluation() -> dict[str, Any]:
    """Get static evaluation metrics (Feb 6, 2024 - July 19, 2024).

    Uses BigQuery SQL aggregation for fast metrics computation.

    Returns
    -------
    dict[str, Any]
        Static evaluation metrics with RMSE/MAE by horizon
    """
    logger = get_logger()
    logger.info("=" * 80)
    logger.info("GET /evaluation/static endpoint called")

    if not hasattr(app.state, "eval_storage"):
        raise HTTPException(
            status_code=503,
            detail="Evaluation storage not available. BigQuery may not be configured.",
        )

    try:
        # Define static evaluation period: Feb 6, 2024 - July 19, 2024
        start_date = datetime(2024, 2, 6, 12, 0, 0)
        end_date = datetime(2024, 7, 19, 17, 0, 0)

        logger.info(f"Static evaluation period: {start_date} to {end_date}")

        # Compute metrics using fast SQL aggregation in BigQuery (run in thread)
        logger.info("Calling compute_metrics_for_period...")
        raw_metrics = await asyncio.to_thread(
            app.state.eval_storage.compute_metrics_for_period, start_date, end_date
        )

        logger.info("Raw metrics from compute_metrics_for_period:")
        logger.info(f"  overall_rmse: {raw_metrics['overall_rmse']:.4f}°C")
        logger.info(f"  overall_mae: {raw_metrics['overall_mae']:.4f}°C")
        logger.info(f"  total_samples: {raw_metrics['total_samples']:,}")
        logger.info(f"  by_horizon keys: {list(raw_metrics['by_horizon'].keys())}")

        # Restructure metrics to match frontend expectations
        metrics = {
            "overall": {
                "rmse": raw_metrics["overall_rmse"],
                "mae": raw_metrics["overall_mae"],
                "sample_count": raw_metrics["total_samples"],
            },
            "by_horizon": raw_metrics["by_horizon"],
        }

        logger.info("Restructured metrics for frontend:")
        logger.info(f"  metrics['overall']: {metrics['overall']}")
        logger.info(f"  metrics['by_horizon']: {metrics['by_horizon']}")

        if raw_metrics["total_samples"] == 0:
            logger.warning("No samples found for static evaluation period")
            return {
                "message": "No predictions found for static evaluation period",
                "evaluation_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "metrics": metrics,
            }

        # Store computed metrics for caching (run in thread)
        logger.info("Storing evaluation metrics to BigQuery...")
        await asyncio.to_thread(
            app.state.eval_storage.store_evaluation_metrics,
            evaluation_date=datetime.now(),
            metrics=raw_metrics,  # Store raw format internally
            eval_type="static",
        )

        response = {
            "evaluation_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "metrics": metrics,
            "computed_at": datetime.now().isoformat(),
        }

        logger.info("Static evaluation endpoint completed successfully")
        logger.info("=" * 80)
        return response

    except Exception as e:
        logger.error(f"Failed to compute static evaluation: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute static evaluation: {str(e)}",
        ) from e


@app.post("/scheduler/trigger-forecast")
async def trigger_forecast() -> dict[str, Any]:
    """Manually trigger a forecast run via HTTP.

    This endpoint is designed to be called by Cloud Scheduler to ensure
    the forecast job completes within an HTTP request context, preventing
    Cloud Run from killing the instance mid-execution.

    Returns
    -------
    dict[str, Any]
        Status of the forecast run including duration and records generated
    """
    if not hasattr(app.state, "forecast_scheduler"):
        raise HTTPException(
            status_code=503, detail="Forecast scheduler not initialized"
        )

    logger = get_logger()

    # Check if already running
    if app.state.forecast_scheduler.is_running:
        logger.warning("Forecast job already running, rejecting duplicate request")
        return {
            "status": "already_running",
            "message": "A forecast job is already in progress",
        }

    try:
        # Run the forecast job directly (not via scheduler)
        await app.state.forecast_scheduler._run_forecast_job()

        return {
            "status": "success",
            "message": "Forecast completed successfully",
            "last_run_timestamp": (
                app.state.forecast_scheduler.last_run_timestamp.isoformat()
                if app.state.forecast_scheduler.last_run_timestamp
                else None
            ),
            "last_data_timestamp": (
                app.state.forecast_scheduler.last_data_timestamp.isoformat()
                if app.state.forecast_scheduler.last_data_timestamp
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Triggered forecast failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Forecast execution failed: {str(e)}",
        ) from e


@app.get("/logs/forecast-runs")
async def get_forecast_logs(limit: int = 100) -> dict[str, Any]:
    """Get recent forecast run logs.

    Parameters
    ----------
    limit : int
        Maximum number of logs to return (default: 100)

    Returns
    -------
    dict[str, Any]
        List of forecast run logs with metadata. If table doesn't exist,
        returns setup_required: true
    """
    if not hasattr(app.state, "eval_storage") or app.state.eval_storage is None:
        raise HTTPException(
            status_code=503,
            detail="BigQuery storage not available.",
        )

    try:
        # Run BigQuery call in thread to avoid blocking event loop
        logs = await asyncio.to_thread(
            app.state.eval_storage.get_forecast_logs, limit=limit
        )

        return {
            "count": len(logs),
            "logs": logs,
            "setup_required": False,
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg and "forecast_runs" in error_msg:
            # Table doesn't exist yet - return helpful message instead of error
            return {
                "count": 0,
                "logs": [],
                "setup_required": True,
                "setup_command": f"./bigquery/setup.sh {app.state.eval_storage.project_id}",
            }

        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch forecast logs: {str(e)}",
        ) from e


@app.get("/evaluation/dynamic")
async def get_dynamic_evaluation() -> dict[str, Any]:
    """Get dynamic evaluation metrics (rolling 1-month window).

    Uses BigQuery SQL aggregation for fast metrics computation.

    Returns
    -------
    dict[str, Any]
        Dynamic evaluation metrics for last 30 days
    """
    logger = get_logger()
    logger.info("=" * 80)
    logger.info("GET /evaluation/dynamic endpoint called")

    if not hasattr(app.state, "eval_storage"):
        raise HTTPException(
            status_code=503,
            detail="Evaluation storage not available. BigQuery may not be configured.",
        )

    try:
        # Define dynamic evaluation window: last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        logger.info(f"Evaluation period: {start_date} to {end_date} (30 days)")

        # Compute metrics using fast SQL aggregation in BigQuery (run in thread)
        logger.info("Calling compute_metrics_for_period...")
        raw_metrics = await asyncio.to_thread(
            app.state.eval_storage.compute_metrics_for_period, start_date, end_date
        )

        logger.info("Raw metrics from compute_metrics_for_period:")
        logger.info(f"  overall_rmse: {raw_metrics['overall_rmse']:.4f}°C")
        logger.info(f"  overall_mae: {raw_metrics['overall_mae']:.4f}°C")
        logger.info(f"  total_samples: {raw_metrics['total_samples']:,}")
        logger.info(f"  by_horizon keys: {list(raw_metrics['by_horizon'].keys())}")

        # Restructure metrics to match frontend expectations
        metrics = {
            "overall": {
                "rmse": raw_metrics["overall_rmse"],
                "mae": raw_metrics["overall_mae"],
                "sample_count": raw_metrics["total_samples"],
            },
            "by_horizon": raw_metrics["by_horizon"],
        }

        logger.info("Restructured metrics for frontend:")
        logger.info(f"  metrics['overall']: {metrics['overall']}")
        logger.info(f"  metrics['by_horizon']: {metrics['by_horizon']}")

        if raw_metrics["total_samples"] == 0:
            logger.warning("No samples found for dynamic evaluation window")
            return {
                "message": "No predictions found for dynamic evaluation window",
                "evaluation_window": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": 30,
                },
                "metrics": metrics,
            }

        # Store computed metrics (run in thread)
        logger.info("Storing evaluation metrics to BigQuery...")
        await asyncio.to_thread(
            app.state.eval_storage.store_evaluation_metrics,
            evaluation_date=datetime.now(),
            metrics=raw_metrics,  # Store raw format internally
            eval_type="dynamic",
        )

        response = {
            "evaluation_window": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": 30,
            },
            "metrics": metrics,
            "computed_at": datetime.now().isoformat(),
        }

        logger.info("Evaluation endpoint completed successfully")
        logger.info("=" * 80)
        return response

    except Exception as e:
        logger.error(f"Failed to compute dynamic evaluation: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute dynamic evaluation: {str(e)}",
        ) from e
