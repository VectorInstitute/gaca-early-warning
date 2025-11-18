"""FastAPI application for GACA Early Warning System.

This module provides a REST API for interacting with the GACA temperature
forecasting model, including endpoints for running inference and retrieving
predictions.
"""

import asyncio
import logging
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


# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=1)


app = FastAPI(
    title="GACA Early Warning System API",
    description="Temperature forecasting API for Southwestern Ontario",
    version="0.1.0",
)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        pass
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
