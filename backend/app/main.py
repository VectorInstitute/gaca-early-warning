"""FastAPI application for GACA Early Warning System.

This module provides a REST API for interacting with the GACA temperature
forecasting model, including endpoints for running inference and retrieving
predictions.
"""

from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from gaca_ews.core.data_extraction import fetch_last_hours
from gaca_ews.core.preprocessing import preprocess_for_inference
from gaca_ews.model.gcngru import GCNGRU


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


class Config:
    """Configuration holder for model and paths."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize configuration from YAML file."""
        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        # Convert to namespace-like object
        def to_namespace(obj: Any) -> Any:
            if isinstance(obj, dict):
                ns = Namespace()
                for key, value in obj.items():
                    setattr(ns, key, to_namespace(value))
                return ns
            return obj

        self.config = to_namespace(cfg_dict)


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


class ModelState(TypedDict, total=False):
    """Type definition for global model state."""

    model: GCNGRU
    config: Namespace
    artifacts_loaded: bool
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    nodes_df: pd.DataFrame
    device: torch.device


# Global state
model_state: ModelState = {"artifacts_loaded": False}


def load_model_artifacts() -> None:
    """Load model and artifacts into memory."""
    if model_state["artifacts_loaded"]:
        return

    config = Config()
    cfg = config.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scalers
    feature_scaler = joblib.load(cfg.model.feature_scaler_path)
    target_scaler = joblib.load(cfg.model.target_scaler_path)

    # Load graph components
    edge_index = torch.load(cfg.graph.edge_index_path, map_location=device)
    edge_weight = torch.load(cfg.graph.edge_weight_path, map_location=device)
    nodes_df = pd.read_csv(cfg.graph.nodes_csv_path)

    # Load model
    checkpoint = torch.load(cfg.model.model_path, map_location=device)
    model_class = checkpoint["model_class"]
    model_args = checkpoint["model_args"]
    raw_state = checkpoint["model_state_dict"]

    # Clean state dict
    clean_state = {}
    for k, v in raw_state.items():
        if k.startswith("module."):
            clean_state[k.replace("module.", "", 1)] = v
        else:
            clean_state[k] = v

    # Adjust model args
    if "pred_offsets" in model_args:
        model_args["pred_horizons"] = len(model_args["pred_offsets"])
        del model_args["pred_offsets"]
        if "model_name" in model_args:
            del model_args["model_name"]

    model_args["num_nodes"] = edge_index.max().item() + 1

    if model_class == "DistributedDataParallel":
        model_class = cfg.model_arch

    if model_class == "GCNGRU":
        model = GCNGRU(**model_args)
    else:
        raise ValueError(f"Unknown model: {model_class}")

    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()

    # Store in global state
    model_state["model"] = model
    model_state["feature_scaler"] = feature_scaler
    model_state["target_scaler"] = target_scaler
    model_state["edge_index"] = edge_index
    model_state["edge_weight"] = edge_weight
    model_state["nodes_df"] = nodes_df
    model_state["config"] = cfg
    model_state["device"] = device
    model_state["artifacts_loaded"] = True


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    load_model_artifacts()


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
    return {"status": "healthy", "model_loaded": model_state["artifacts_loaded"]}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """Get information about the loaded model."""
    if not model_state["artifacts_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cfg = model_state["config"]
    nodes_df = model_state["nodes_df"]

    assert cfg is not None, "Config not loaded"
    assert nodes_df is not None, "Nodes dataframe not loaded"

    return ModelInfo(
        model_architecture=cfg.model_arch,
        num_nodes=len(nodes_df),
        input_features=cfg.features,
        prediction_horizons=cfg.pred_offsets,
        region={
            "lat_min": cfg.region.lat_min,
            "lat_max": cfg.region.lat_max,
            "lon_min": cfg.region.lon_min,
            "lon_max": cfg.region.lon_max,
        },
        status="loaded",
    )


@app.post("/predict", response_model=list[PredictionResponse])
async def run_inference(request: PredictionRequest) -> list[PredictionResponse]:
    """Run inference and return predictions."""
    if not model_state["artifacts_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Fetch data
        cfg = model_state["config"]
        assert cfg is not None, "Config not loaded"
        df, latest_ts = fetch_last_hours(cfg)

        # Preprocess
        feature_scaler = model_state["feature_scaler"]
        assert feature_scaler is not None, "Feature scaler not loaded"
        X_seq, timestamps, in_channels, num_nodes = preprocess_for_inference(
            data=df, feature_scaler=feature_scaler, args=cfg
        )

        # Run inference
        device = model_state["device"]
        assert device is not None, "Device not set"
        X_test = X_seq.to(device)

        model = model_state["model"]
        edge_index = model_state["edge_index"]
        edge_weight = model_state["edge_weight"]
        assert model is not None, "Model not loaded"
        assert edge_index is not None, "Edge index not loaded"
        assert edge_weight is not None, "Edge weight not loaded"

        with torch.no_grad():
            preds_scaled = model(X_test, edge_index, edge_weight)

        # Inverse transform
        target_scaler = model_state["target_scaler"]
        assert target_scaler is not None, "Target scaler not loaded"
        preds_np = preds_scaled.cpu().numpy()
        preds_unscaled = target_scaler.inverse_transform(
            preds_np.reshape(-1, preds_np.shape[-1])
        ).reshape(preds_np.shape)

        # Format response
        nodes_df = model_state["nodes_df"]
        assert nodes_df is not None, "Nodes dataframe not loaded"
        num_pred_nodes = preds_unscaled.shape[2]

        predictions = []
        for h_idx, horizon in enumerate(cfg.pred_offsets):
            forecast_time = latest_ts + timedelta(hours=horizon)

            for node_idx in range(num_pred_nodes):
                lat = nodes_df.iloc[node_idx]["lat"]
                lon = nodes_df.iloc[node_idx]["lon"]
                pred_val = float(preds_unscaled[0, h_idx, node_idx, 0])

                predictions.append(
                    PredictionResponse(
                        forecast_time=forecast_time.isoformat(),
                        horizon_hours=horizon,
                        lat=lat,
                        lon=lon,
                        predicted_temp=pred_val,
                    )
                )

        return predictions

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
