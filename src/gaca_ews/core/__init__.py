"""Core utilities for data extraction, preprocessing, and configuration."""

from gaca_ews.core.config import get_args
from gaca_ews.core.data_extraction import fetch_last_hours
from gaca_ews.core.inference import InferenceEngine
from gaca_ews.core.logger import (
    get_file_handler,
    get_logger,
    get_rich_handler,
    logger,
    setup_console_logging,
    setup_file_logging,
)
from gaca_ews.core.plotting import plot_inference_maps, plot_prediction_timeseries
from gaca_ews.core.preprocessing import preprocess_for_inference


# Backward compatibility alias
setup_logging = setup_file_logging


__all__ = [
    "get_args",
    "fetch_last_hours",
    "InferenceEngine",
    "logger",
    "setup_logging",  # Backward compatibility
    "setup_file_logging",
    "setup_console_logging",
    "get_logger",
    "get_rich_handler",
    "get_file_handler",
    "plot_inference_maps",
    "plot_prediction_timeseries",
    "preprocess_for_inference",
]
