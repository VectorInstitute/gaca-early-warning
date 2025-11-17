###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to define plotting functions for visualization
###########################################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from util.logger import logger

def plot_inference_maps(preds_unscaled, nodes_df, pred_offsets, out_dir,
                        latest_timestamp):
    """
    func that plots spatial maps per horizon
    """
    os.makedirs(out_dir, exist_ok=True)

    predictions = preds_unscaled[0]  # [H, N, 1]
    H, N = predictions.shape[0], predictions.shape[1]

    # build positions (lon, lat)
    pos = {
            i: (nodes_df.iloc[i]["lon"], nodes_df.iloc[i]["lat"])
            for i in range(N)
          }
    
    cmap = "inferno"
    
    # loop through horizons
    for h in range(H):
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
            vmax=vmax
        )

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Predicted Temp (°C)")

        ax.set_title(f"Forecast for {future_ts.strftime('%Y-%m-%d %H:%M')} (+{pred_offsets[h]}h)")
        ax.set_axis_off()

        # save using actual timestamp for context
        save_name = f"spatial_{future_ts_str}_plus{pred_offsets[h]}h.png"
        save_path = os.path.join(out_dir, save_name)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"📌 saved spatial map → {save_path}")

def plot_prediction_timeseries(preds_unscaled, pred_offsets, out_dir,
                               latest_timestamp, node_ids=None, num_random_nodes=5):
    """
    func that plots timeseries plots for nodes
    - one plot is mean temp across all nodes at horizon h
    - other plots timeseries for n random nodes (unless you specify via the node_ids param)
    """
    os.makedirs(out_dir, exist_ok=True)

    preds = preds_unscaled[0]     # [H, N, 1]
    H, N = preds.shape[0], preds.shape[1]

    future_times = [
        latest_timestamp + pd.Timedelta(hours=h)
        for h in pred_offsets
    ]
    future_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in future_times]

    # mean across nodes
    mean_vals = preds[:, :, 0].mean(axis=1)

    plt.figure(figsize=(11, 4))
    plt.plot(future_labels, mean_vals, marker='o', color="#0072B2")
    plt.xlabel("Timestamp")
    plt.ylabel("Mean predicted temp (°C)")
    plt.title(f"Node-Averaged Forecast\nStarting from {latest_timestamp.strftime('%Y-%m-%d %H:%M')}")
    plt.xticks(rotation=45)
    plt.grid(True)

    save_path = os.path.join(out_dir, f"mean_timeseries_{latest_timestamp.strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📌 saved mean-timeseries → {save_path}")

    # single node timeseries
    if node_ids is None: # if didnt pre-select node ids to plot,
        # randomly choose num_random_nodes nodes
        node_ids = np.random.choice(N, size=num_random_nodes, replace=False).tolist()

    # create a plot per node
    for node_id in node_ids:
        vals = preds[:, node_id, 0]

        plt.figure(figsize=(11, 4))
        plt.plot(future_labels, vals, marker='o', color="#D55E00")
        plt.xlabel("Timestamp")
        plt.ylabel("Predicted temp (°C)")
        plt.title(f"Node {node_id} Forecast\nStarting {latest_timestamp}")
        plt.xticks(rotation=45)
        plt.grid(True)

        save_path = os.path.join(
            out_dir,
            f"node_{node_id}_timeseries_{latest_timestamp.strftime('%Y%m%d_%H%M')}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📌 saved node-timeseries → {save_path}")
