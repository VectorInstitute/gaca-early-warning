"""FastAPI application for GACA Early Warning System.

This module provides a REST API for interacting with the GACA temperature
forecasting model, including endpoints for running inference and retrieving
predictions.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gaca_ews.core.inference import InferenceEngine
from gaca_ews.core.logger import get_logger
from gaca_ews.evaluation.storage import EvaluationStorage


# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=1)


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


class PredictionRequest(BaseModel):
    """Request model for inference."""

    num_hours: int = 24


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    forecast_time: str
    horizon_hours: int
    lat: float
    lon: float
    predicted_temp: float


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
    """Load model on startup."""
    # Set logger to WARNING level for API to reduce noise
    logger = get_logger()
    logger.setLevel(logging.WARNING)

    app.state.engine = InferenceEngine("config.yaml")
    app.state.engine.load_artifacts()

    # Initialize evaluation storage (only if BigQuery is available)
    try:
        app.state.eval_storage = EvaluationStorage()
    except Exception as e:
        logger.warning(
            f"Failed to initialize BigQuery: {e}. Evaluation endpoints will be disabled."
        )


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


async def run_pipeline_with_progress(
    websocket: WebSocket | None = None,
) -> list[PredictionResponse]:
    """Run inference pipeline with progress updates via WebSocket."""
    engine = app.state.engine

    async def send_progress(step: str, status: str = "in_progress") -> None:
        """Send progress update via WebSocket."""
        if websocket:
            await websocket.send_json(
                {
                    "type": "progress",
                    "step": step,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    try:
        # Step 1: Load artifacts (if not already loaded)
        await send_progress("Loading model artifacts", "in_progress")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, engine.load_artifacts)
        await send_progress("Loading model artifacts", "completed")

        # Step 2: Fetch data
        await send_progress("Fetching NOAA meteorological data", "in_progress")
        data, latest_ts = await loop.run_in_executor(executor, engine.fetch_data)
        await send_progress("Fetching NOAA meteorological data", "completed")

        # Step 3: Preprocess
        await send_progress("Preprocessing features", "in_progress")
        X = await loop.run_in_executor(executor, engine.preprocess, data)
        await send_progress("Preprocessing features", "completed")

        # Step 4: Run inference
        await send_progress("Running model inference", "in_progress")
        predictions = await loop.run_in_executor(executor, engine.predict, X)
        await send_progress("Running model inference", "completed")

        # Format response
        num_pred_nodes = predictions.shape[2]
        pred_offsets = engine.config["pred_offsets"]

        response = []
        for h_idx, horizon in enumerate(pred_offsets):
            forecast_time = latest_ts + timedelta(hours=horizon)

            for node_idx in range(num_pred_nodes):
                lat = engine.nodes_df.iloc[node_idx]["lat"]
                lon = engine.nodes_df.iloc[node_idx]["lon"]
                pred_val = float(predictions[0, h_idx, node_idx, 0])

                response.append(
                    PredictionResponse(
                        forecast_time=forecast_time.isoformat(),
                        horizon_hours=horizon,
                        lat=lat,
                        lon=lon,
                        predicted_temp=pred_val,
                    )
                )

        return response

    except Exception as e:
        if websocket:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        raise


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket) -> None:
    """WebSocket endpoint for inference with real-time progress updates."""
    await websocket.accept()

    if not hasattr(app.state, "engine"):
        await websocket.send_json({"type": "error", "message": "Model not loaded"})
        await websocket.close()
        return

    try:
        # Run pipeline with progress updates
        response = await run_pipeline_with_progress(websocket)

        # Send completion with data
        await websocket.send_json(
            {
                "type": "complete",
                "data": [pred.model_dump() for pred in response],
                "timestamp": datetime.now().isoformat(),
            }
        )

    except WebSocketDisconnect:
        # Client disconnected normally; no error.
        get_logger().debug("WebSocket client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


@app.post("/predict", response_model=list[PredictionResponse])
async def run_inference(request: PredictionRequest) -> list[PredictionResponse]:
    """Run inference and return predictions (legacy REST endpoint)."""
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return await run_pipeline_with_progress(None)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        ) from e


@app.get("/predictions/latest")
async def get_latest_predictions() -> dict[str, Any]:
    """Get the most recent predictions from file if available."""
    predictions_file = Path("TEST/predictions.csv")

    if not predictions_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No predictions file found. Run inference first.",
        )

    try:
        df = pd.read_csv(predictions_file)

        # Convert to list of dicts
        predictions = df.to_dict(orient="records")

        return {
            "count": len(predictions),
            "predictions": predictions,
            "file_timestamp": datetime.fromtimestamp(
                predictions_file.stat().st_mtime
            ).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read predictions: {str(e)}"
        ) from e


@app.get("/evaluation/static")
async def get_static_evaluation() -> dict[str, Any]:
    """Get static evaluation metrics (Feb 6, 2024 - July 19, 2024).

    Uses BigQuery SQL aggregation for fast metrics computation.

    Returns
    -------
    dict[str, Any]
        Static evaluation metrics with RMSE/MAE by horizon
    """
    if not hasattr(app.state, "eval_storage"):
        raise HTTPException(
            status_code=503,
            detail="Evaluation storage not available. BigQuery may not be configured.",
        )

    try:
        # Define static evaluation period: Feb 6, 2024 - July 19, 2024
        start_date = datetime(2024, 2, 6, 12, 0, 0)
        end_date = datetime(2024, 7, 19, 17, 0, 0)

        # Compute metrics using fast SQL aggregation in BigQuery
        raw_metrics = app.state.eval_storage.compute_metrics_for_period(
            start_date, end_date
        )

        # Restructure metrics to match frontend expectations
        metrics = {
            "overall": {
                "rmse": raw_metrics["overall_rmse"],
                "mae": raw_metrics["overall_mae"],
                "sample_count": raw_metrics["total_samples"],
            },
            "by_horizon": raw_metrics["by_horizon"],
        }

        if raw_metrics["total_samples"] == 0:
            return {
                "message": "No predictions found for static evaluation period",
                "evaluation_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "metrics": metrics,
            }

        # Store computed metrics for caching
        app.state.eval_storage.store_evaluation_metrics(
            evaluation_date=datetime.now(),
            metrics=raw_metrics,  # Store raw format internally
            eval_type="static",
        )

        return {
            "evaluation_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "metrics": metrics,
            "computed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute static evaluation: {str(e)}",
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
    if not hasattr(app.state, "eval_storage"):
        raise HTTPException(
            status_code=503,
            detail="Evaluation storage not available. BigQuery may not be configured.",
        )

    try:
        # Define dynamic evaluation window: last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Compute metrics using fast SQL aggregation in BigQuery
        raw_metrics = app.state.eval_storage.compute_metrics_for_period(
            start_date, end_date
        )

        # Restructure metrics to match frontend expectations
        metrics = {
            "overall": {
                "rmse": raw_metrics["overall_rmse"],
                "mae": raw_metrics["overall_mae"],
                "sample_count": raw_metrics["total_samples"],
            },
            "by_horizon": raw_metrics["by_horizon"],
        }

        if raw_metrics["total_samples"] == 0:
            return {
                "message": "No predictions found for dynamic evaluation window",
                "evaluation_window": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": 30,
                },
                "metrics": metrics,
            }

        # Store computed metrics
        app.state.eval_storage.store_evaluation_metrics(
            evaluation_date=datetime.now(),
            metrics=raw_metrics,  # Store raw format internally
            eval_type="dynamic",
        )

        return {
            "evaluation_window": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": 30,
            },
            "metrics": metrics,
            "computed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute dynamic evaluation: {str(e)}",
        ) from e


@app.post("/evaluation/store-run")
async def store_prediction_run() -> dict[str, Any]:
    """Store the latest prediction run to BigQuery for evaluation.

    Reads predictions from TEST/predictions.csv and stores them.

    Returns
    -------
    dict[str, Any]
        Status message with row count
    """
    if not hasattr(app.state, "eval_storage"):
        raise HTTPException(
            status_code=503,
            detail="Evaluation storage not available. BigQuery may not be configured.",
        )

    predictions_file = Path("TEST/predictions.csv")

    if not predictions_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No predictions file found. Run inference first.",
        )

    try:
        df = pd.read_csv(predictions_file)
        run_timestamp = datetime.fromtimestamp(predictions_file.stat().st_mtime)

        rows_loaded = app.state.eval_storage.store_predictions(df, run_timestamp)

        return {
            "status": "success",
            "message": "Predictions stored successfully",
            "rows_loaded": rows_loaded,
            "run_timestamp": run_timestamp.isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store predictions: {str(e)}",
        ) from e
