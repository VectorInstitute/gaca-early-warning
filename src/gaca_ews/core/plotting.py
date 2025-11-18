"""Visualization and plotting utilities for model predictions.

This module provides functions for creating spatial maps and timeseries plots of
weather forecast predictions. Visualizations include spatial temperature distributions
across the prediction domain and temporal evolution of predictions at specific nodes
or averaged across all nodes.

Functions
---------
plot_inference_maps
    Generate spatial map visualizations for each prediction horizon.
plot_prediction_timeseries
    Create timeseries plots showing temperature predictions over time.

Notes
-----
Spatial maps use scatter plots colored by predicted temperature values with geographic
coordinates. Timeseries plots show both node-averaged predictions and individual node
predictions. All plots are saved as PNG files with descriptive filenames including
timestamps and forecast horizons.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to define plotting functions for visualization
###########################################################################################

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from numpy.typing import NDArray

from gaca_ews.core.logger import logger


def plot_inference_maps(
    preds_unscaled: NDArray[Any],
    nodes_df: pd.DataFrame,
    pred_offsets: list[int],
    out_dir: str,
    latest_timestamp: pd.Timestamp,
) -> None:
    """Generate and save spatial temperature maps for each prediction horizon.

    Parameters
    ----------
    preds_unscaled : np.ndarray
        Unscaled predictions of shape (batch_size, horizons, num_nodes, features).
    nodes_df : pd.DataFrame
        DataFrame containing node information with 'lat' and 'lon' columns.
    pred_offsets : list
        List of prediction horizon offsets in hours.
    out_dir : str
        Output directory path for saving plots.
    latest_timestamp : pd.Timestamp
        The latest timestamp in the input sequence.
    """
    os.makedirs(out_dir, exist_ok=True)

    predictions = preds_unscaled[0]  # [H, N, 1]
    num_horizons = predictions.shape[0]

    cmap = "inferno"

    # loop through horizons
    for h in range(num_horizons):
        pred_vals = predictions[h, :, 0]
        vmin, vmax = pred_vals.min(), pred_vals.max()

        # calculate future timestamp based on latest ts + offset
        future_ts = latest_timestamp + pd.Timedelta(hours=pred_offsets[h])
        future_ts_str = future_ts.strftime("%Y-%m-%d_%H-%M")

        fig, ax = plt.subplots(figsize=(7, 6))

        # plot info via scatter graph
        ax.scatter(
            nodes_df["lon"],
            nodes_df["lat"],
            c=pred_vals,
            cmap=cmap,
            s=10,
            vmin=vmin,
            vmax=vmax,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Predicted Temp (°C)")

        ax.set_title(
            f"Forecast for {future_ts.strftime('%Y-%m-%d %H:%M')} (+{pred_offsets[h]}h)"
        )
        ax.set_axis_off()

        # save using actual timestamp for context
        save_name = f"spatial_{future_ts_str}_plus{pred_offsets[h]}h.png"
        save_path = os.path.join(out_dir, save_name)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"📌 saved spatial map → {save_path}")


def plot_prediction_timeseries(
    preds_unscaled: NDArray[Any],
    pred_offsets: list[int],
    out_dir: str,
    latest_timestamp: pd.Timestamp,
    node_ids: list[int] | None = None,
    num_random_nodes: int = 5,
) -> None:
    """Generate timeseries plots of temperature predictions.

    Creates two types of plots showing mean temperature and individual node
    predictions: node-averaged temperature over time and individual node
    temperature forecasts for randomly selected or specified nodes.

    Parameters
    ----------
    preds_unscaled : np.ndarray
        Unscaled predictions of shape (batch_size, horizons, num_nodes, features).
    pred_offsets : list
        List of prediction horizon offsets in hours.
    out_dir : str
        Output directory path for saving plots.
    latest_timestamp : pd.Timestamp
        The latest timestamp in the input sequence.
    node_ids : list, optional
        Specific node IDs to plot. If None, randomly selects nodes.
    num_random_nodes : int, optional
        Number of random nodes to plot if node_ids is None. Default is 5.
    """
    os.makedirs(out_dir, exist_ok=True)

    preds = preds_unscaled[0]  # [H, N, 1]
    num_nodes = preds.shape[1]

    future_times = [latest_timestamp + pd.Timedelta(hours=h) for h in pred_offsets]
    future_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in future_times]

    # mean across nodes
    mean_vals = preds[:, :, 0].mean(axis=1)

    plt.figure(figsize=(11, 4))
    plt.plot(future_labels, mean_vals, marker="o", color="#0072B2")
    plt.xlabel("Timestamp")
    plt.ylabel("Mean predicted temp (°C)")
    plt.title(
        f"Node-Averaged Forecast\nStarting from {latest_timestamp.strftime('%Y-%m-%d %H:%M')}"
    )
    plt.xticks(rotation=45)
    plt.grid(True)

    save_path = os.path.join(
        out_dir, f"mean_timeseries_{latest_timestamp.strftime('%Y%m%d_%H%M')}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📌 saved mean-timeseries → {save_path}")

    # single node timeseries
    if node_ids is None:  # if didnt pre-select node ids to plot,
        # randomly choose num_random_nodes nodes
        node_ids = np.random.choice(
            num_nodes, size=num_random_nodes, replace=False
        ).tolist()

    # create a plot per node
    for node_id in node_ids:
        vals = preds[:, node_id, 0]

        plt.figure(figsize=(11, 4))
        plt.plot(future_labels, vals, marker="o", color="#D55E00")
        plt.xlabel("Timestamp")
        plt.ylabel("Predicted temp (°C)")
        plt.title(f"Node {node_id} Forecast\nStarting {latest_timestamp}")
        plt.xticks(rotation=45)
        plt.grid(True)

        save_path = os.path.join(
            out_dir,
            f"node_{node_id}_timeseries_{latest_timestamp.strftime('%Y%m%d_%H%M')}.png",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📌 saved node-timeseries → {save_path}")
