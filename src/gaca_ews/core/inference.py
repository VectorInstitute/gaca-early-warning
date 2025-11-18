"""Core inference module for GACA Early Warning System.

This module provides the main inference pipeline functions that are shared
by the CLI and API backend. All model loading and prediction logic is centralized here.
"""

from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
import yaml
from numpy.typing import NDArray

from gaca_ews.core.data_extraction import fetch_last_hours
from gaca_ews.core.plotting import plot_inference_maps, plot_prediction_timeseries
from gaca_ews.core.preprocessing import preprocess_for_inference
from gaca_ews.model.gcngru import GCNGRU


class InferenceEngine:
    """Manages model artifacts and runs inference."""

    def __init__(self, config_path: str | Path) -> None:
        """Initialize inference engine with configuration.

        Parameters
        ----------
        config_path : str | Path
            Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.args = self._config_to_namespace(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Artifacts (loaded on demand)
        self.model: GCNGRU | None = None
        self.feature_scaler: Any = None
        self.target_scaler: Any = None
        self.edge_index: torch.Tensor | None = None
        self.edge_weight: torch.Tensor | None = None
        self.nodes_df: pd.DataFrame | None = None
        self._artifacts_loaded = False

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _config_to_namespace(self, config: dict[str, Any]) -> Namespace:
        """Convert config dict to Namespace for compatibility."""
        ns = Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(ns, key, self._config_to_namespace(value))
            else:
                setattr(ns, key, value)
        return ns

    def load_artifacts(self) -> None:
        """Load all model artifacts (model, scalers, graph components)."""
        if self._artifacts_loaded:
            return

        # Load scalers
        self.feature_scaler = joblib.load(self.config["model"]["feature_scaler_path"])
        self.target_scaler = joblib.load(self.config["model"]["target_scaler_path"])

        # Load graph components
        self.edge_index = torch.load(
            self.config["graph"]["edge_index_path"], map_location=self.device
        )
        self.edge_weight = torch.load(
            self.config["graph"]["edge_weight_path"], map_location=self.device
        )
        self.nodes_df = pd.read_csv(self.config["graph"]["nodes_csv_path"])

        # Load model
        checkpoint = torch.load(
            self.config["model"]["model_path"], map_location=self.device
        )
        model_class = checkpoint["model_class"]
        model_args = checkpoint["model_args"]
        raw_state = checkpoint["model_state_dict"]

        # Clean state dict (remove DDP prefix if present)
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

        model_args["num_nodes"] = self.edge_index.max().item() + 1

        # Handle DDP wrapper
        if model_class == "DistributedDataParallel":
            model_class = self.config.get("model_arch", "GCNGRU")

        # Instantiate model
        if model_class == "GCNGRU":
            self.model = GCNGRU(**model_args)
        else:
            raise ValueError(f"Unknown model class: {model_class}")

        self.model.load_state_dict(clean_state)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._artifacts_loaded = True

    def fetch_data(self) -> tuple[pd.DataFrame, datetime]:
        """Fetch meteorological data from NOAA.

        Returns
        -------
        tuple[pd.DataFrame, datetime]
            Raw data DataFrame and latest timestamp
        """
        return fetch_last_hours(self.args)

    def preprocess(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess raw data into model input format.

        Parameters
        ----------
        data : pd.DataFrame
            Raw meteorological data

        Returns
        -------
        torch.Tensor
            Preprocessed input tensor ready for model
        """
        X_seq, timestamps, in_channels, num_nodes = preprocess_for_inference(
            data=data, feature_scaler=self.feature_scaler, args=self.args
        )
        return X_seq.to(self.device)

    def predict(self, X: torch.Tensor) -> NDArray:
        """Run model inference and return predictions.

        Parameters
        ----------
        X : torch.Tensor
            Preprocessed input tensor

        Returns
        -------
        NDArray
            Unscaled predictions array [batch, horizons, nodes, features]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")

        with torch.no_grad():
            preds_scaled = self.model(X, self.edge_index, self.edge_weight)

        # Inverse transform to get actual temperature values
        preds_np = preds_scaled.cpu().numpy()
        return self.target_scaler.inverse_transform(
            preds_np.reshape(-1, preds_np.shape[-1])
        ).reshape(preds_np.shape)

    def run_full_pipeline(self) -> tuple[NDArray, datetime]:
        """Execute complete inference pipeline: fetch -> preprocess -> predict.

        Returns
        -------
        tuple[NDArray, datetime]
            Predictions array and latest timestamp
        """
        # Ensure artifacts are loaded
        self.load_artifacts()

        # Fetch data
        data, latest_ts = self.fetch_data()

        # Preprocess
        X = self.preprocess(data)

        # Predict
        predictions = self.predict(X)

        return predictions, latest_ts

    def save_predictions(
        self, predictions: NDArray, latest_ts: datetime, output_path: str | Path
    ) -> Path:
        """Save predictions to CSV file.

        Parameters
        ----------
        predictions : NDArray
            Prediction array [batch, horizons, nodes, features]
        latest_ts : datetime
            Latest timestamp for calculating forecast times
        output_path : str | Path
            Path where CSV should be saved

        Returns
        -------
        Path
            Path to saved CSV file
        """
        if self.nodes_df is None:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_nodes = predictions.shape[2]
        pred_offsets = self.config["pred_offsets"]
        rows = []

        for h_idx, horizon in enumerate(pred_offsets):
            forecast_time = latest_ts + timedelta(hours=horizon)

            for node_idx in range(num_nodes):
                lat = self.nodes_df.iloc[node_idx]["lat"]
                lon = self.nodes_df.iloc[node_idx]["lon"]
                pred_val = predictions[0, h_idx, node_idx, 0]

                rows.append([forecast_time, horizon, lat, lon, pred_val])

        df = pd.DataFrame(
            rows,
            columns=["forecast_time", "horizon_hours", "lat", "lon", "predicted_temp"],
        )
        df.to_csv(output_path, index=False)

        return output_path

    def generate_plots(
        self, predictions: NDArray, latest_ts: datetime, output_dir: str | Path
    ) -> None:
        """Generate visualization plots.

        Parameters
        ----------
        predictions : NDArray
            Prediction array [batch, horizons, nodes, features]
        latest_ts : datetime
            Latest timestamp
        output_dir : str | Path
            Directory where plots should be saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate spatial maps
        plot_inference_maps(
            preds_unscaled=predictions,
            nodes_df=self.nodes_df,
            pred_offsets=self.config["pred_offsets"],
            out_dir=str(output_dir / "maps"),
            latest_timestamp=latest_ts,
        )

        # Generate timeseries
        plot_prediction_timeseries(
            preds_unscaled=predictions,
            pred_offsets=self.config["pred_offsets"],
            out_dir=str(output_dir / "timeseries"),
            latest_timestamp=latest_ts,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get model configuration information.

        Returns
        -------
        dict[str, Any]
            Model configuration details
        """
        self.load_artifacts()

        if self.nodes_df is None:
            raise RuntimeError("Failed to load nodes DataFrame.")

        return {
            "model_architecture": self.config.get("model_arch", "GCNGRU"),
            "num_nodes": len(self.nodes_df),
            "input_features": self.config["features"],
            "prediction_horizons": self.config["pred_offsets"],
            "region": self.config["region"],
            "device": str(self.device),
            "artifacts_loaded": self._artifacts_loaded,
        }
