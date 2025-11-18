"""FastAPI application for GACA Early Warning System.

This module provides a REST API for interacting with the GACA temperature
forecasting model, including endpoints for running inference and retrieving
predictions.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gaca_ews.core.inference import InferenceEngine


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


class ModelInfo(BaseModel):
    """Model information response."""

    model_architecture: str
    num_nodes: int
    input_features: list[str]
    prediction_horizons: list[int]
    region: dict[str, float]
    status: str


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
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

    return ModelInfo(
        model_architecture=info["model_architecture"],
        num_nodes=info["num_nodes"],
        input_features=info["input_features"],
        prediction_horizons=info["prediction_horizons"],
        region=info["region"],
        status="loaded",
    )


@app.post("/predict", response_model=list[PredictionResponse])
async def run_inference(request: PredictionRequest) -> list[PredictionResponse]:
    """Run inference and return predictions."""
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Run full pipeline
        predictions, latest_ts = app.state.engine.run_full_pipeline()

        # Format response
        num_pred_nodes = predictions.shape[2]
        pred_offsets = app.state.engine.config["pred_offsets"]

        response = []
        for h_idx, horizon in enumerate(pred_offsets):
            forecast_time = latest_ts + timedelta(hours=horizon)

            for node_idx in range(num_pred_nodes):
                lat = app.state.engine.nodes_df.iloc[node_idx]["lat"]
                lon = app.state.engine.nodes_df.iloc[node_idx]["lon"]
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
