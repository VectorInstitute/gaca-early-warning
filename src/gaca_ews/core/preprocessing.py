"""Data preprocessing utilities for model inference.

This module provides functions for preprocessing meteorological data before feeding
it to the GCNGRU model. Operations include feature engineering, missing value
imputation using spatial neighbors, temperature unit conversion, feature scaling,
and graph construction.

Functions
---------
summarize_feature_nans
    Analyze and report NaN values in feature sequences.
fill_graph_feature_nans_with_log
    Fill missing values using spatial neighbor averaging on the graph.
build_graph_from_nodes
    Construct a spatial graph from node coordinates using distance thresholds.
preprocess_for_inference
    Main preprocessing pipeline that prepares raw data for model input.

Notes
-----
The preprocessing pipeline handles:
- Temporal ordering and consistency checks
- Temperature unit conversions (Kelvin to Celsius)
- Positional encoding based on geographic coordinates
- Missing value imputation using graph-based spatial averaging
- Feature scaling while preserving positional encodings
- Windowing for sequence-to-sequence modeling

The module uses NetworkX for graph operations and expects data to be organized
with consistent spatial ordering across time steps.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to define preprocessing functions to set up data for inference
###########################################################################################

import pickle
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic
from numpy.typing import NDArray

from gaca_ews.core.logger import logger


def summarize_feature_nans(
    features_sequence: list[NDArray[Any]], feature_names: list[str] | None = None
) -> int:
    """Analyze and report count and distribution of NaN values.

    Parameters
    ----------
    features_sequence : list of np.ndarray
        List of feature matrices, one per time step.
    feature_names : list of str, optional
        Names of features for detailed reporting. Default is None.

    Returns
    -------
    int
        Total number of NaN values across all time steps.
    """
    total_nans = 0
    for t, mat in enumerate(features_sequence):
        nan_mask = np.isnan(mat)
        count = np.sum(nan_mask)
        if count > 0:
            logger.info(f"🕒 Snapshot {t}: {count} NaNs")
            if feature_names:
                per_feature = np.sum(nan_mask, axis=0)
                for i, fname in enumerate(feature_names):
                    if per_feature[i] > 0:
                        logger.info(f"    ➤ {fname}: {per_feature[i]} NaNs")
        total_nans += count
    logger.info(f"🧮 Total NaNs in sequence: {total_nans}")
    return total_nans


def fill_graph_feature_nans_with_log(
    features_sequence: list[NDArray[Any]],
    graph_base: nx.Graph,
    timestamps: list[Any] | None = None,
    feature_names: list[str] | None = None,
) -> tuple[list[NDArray[Any]], list[tuple[Any, ...]]]:
    """Fill missing feature values using spatial neighbor averaging on the graph.

    Parameters
    ----------
    features_sequence : list of np.ndarray
        List of feature matrices to fill, one per time step. Modified in-place.
    graph_base : nx.Graph
        NetworkX graph defining spatial neighbor relationships.
    timestamps : list, optional
        Timestamps for each time step for logging. Default is None.
    feature_names : list of str, optional
        Names of features for logging. Default is None.

    Returns
    -------
    tuple
        - features_sequence : list of np.ndarray
            The filled feature sequence (modified in-place).
        - fill_log : list of tuple
            Log of all filled values with metadata.
    """
    filled = 0
    fill_log = []  # store (time, node, feature, value, neighbour vals)

    for t, feature_matrix in enumerate(features_sequence):
        for node in range(feature_matrix.shape[0]):
            for f in range(feature_matrix.shape[1]):
                if np.isnan(feature_matrix[node, f]):
                    neighbours = list(graph_base.neighbors(node))
                    neighbour_vals = [
                        feature_matrix[n, f]
                        for n in neighbours
                        if not np.isnan(feature_matrix[n, f])
                    ]
                    if neighbour_vals:
                        filled_val = np.mean(neighbour_vals)
                        features_sequence[t][node, f] = filled_val
                        time_label = (
                            timestamps[t] if timestamps is not None else f"T={t}"
                        )
                        feature_label = feature_names[f] if feature_names else f"F{f}"
                        fill_log.append(
                            (
                                time_label,
                                node,
                                feature_label,
                                filled_val,
                                neighbour_vals,
                            )
                        )
                        filled += 1

    logger.info(f"🩹 filled {filled} nans using spatial neighbors")

    # summary of filled entries
    n = 5
    for entry in fill_log[:n]:  # print only first n just to double check
        logger.info(
            f"🧾 filled nan at {entry[0]}, node {entry[1]}, feature '{entry[2]}' → value: {entry[3]:.3f} (neighbour vals: {neighbour_vals})"
        )

    if filled > n:
        logger.info(f"… and {filled - n} more fills.")

    return features_sequence, fill_log


def build_graph_from_nodes(
    nodes_df: pd.DataFrame,
    distance_threshold: float = 4,
    inverse_distance: bool = False,
) -> nx.Graph:
    """Construct spatial graph from coordinates using distance threshold.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame containing node coordinates with 'lat' and 'lon' columns.
    distance_threshold : float, optional
        Maximum distance in kilometers for edge creation. Default is 4.
    inverse_distance : bool, optional
        If True, use inverse distance as edge weight. Default is False.

    Returns
    -------
    nx.Graph
        NetworkX graph with nodes positioned by coordinates and edges weighted by
        distance.
    """
    coords = nodes_df[["lat", "lon"]].to_numpy()
    num_nodes = len(coords)

    graph: nx.Graph = nx.Graph()

    # add nodes with pos attribute
    for i in range(num_nodes):
        lat, lon = coords[i]
        graph.add_node(i, pos=(lat, lon))

    # add edges based on geodesic distance
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = geodesic(coords[i], coords[j]).km
            if dist <= distance_threshold:
                weight = 1.0 / (dist + 1e-5) if inverse_distance else dist
                graph.add_edge(i, j, weight=weight)

    return graph


def build_feature_sequences(
    data: pd.DataFrame, feats: list[str], celsius: bool
) -> tuple[list[NDArray[Any]], list[datetime]]:
    """Build per-timestamp feature sequences with positional encodings.

    Parameters
    ----------
    data : pd.DataFrame
        Raw meteorological data with datetime, lat, lon, and feature columns.
    feats : list
        List of feature column names to extract.
    celsius : bool
        Whether to convert temperature features from Kelvin to Celsius.

    Returns
    -------
    tuple
        (features_seq, timestamps_sorted, pos_encodings)
    """
    first_datetime = data["datetime"].min()

    constant_data = (
        data[data["datetime"] == first_datetime][["lat", "lon"]]
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )

    pos_encodings = constant_data[["lat", "lon"]].to_numpy(np.float32) * 0.01

    # build per-timestamp feature sequence
    timestamp_dfs: dict[Any, pd.DataFrame] = {
        ts: grp.sort_values(["lat", "lon"]).reset_index(drop=True)
        for ts, grp in data.groupby("datetime")
    }

    timestamps_sorted: list[datetime] = [
        ts if isinstance(ts, datetime) else ts.to_pydatetime()
        for ts in sorted(timestamp_dfs.keys())
    ]
    features_seq = []

    for ts in timestamps_sorted:
        hourly_df = timestamp_dfs[ts]

        # enforce node ordering consistency
        if not np.allclose(
            hourly_df[["lat", "lon"]].values, constant_data[["lat", "lon"]].values
        ):
            raise ValueError(f"[{ts!r}] lat/lon mismatch!")

        # extract features
        raw_feats = hourly_df[feats].to_numpy(np.float32)

        # celsius conversion if needed
        if celsius:
            if "t2m" in feats:
                raw_feats[:, feats.index("t2m")] -= 273.15
            if "d2m" in feats:
                raw_feats[:, feats.index("d2m")] -= 273.15

        # add positional encodings
        raw_feats = np.concatenate([raw_feats, pos_encodings], axis=1)

        features_seq.append(raw_feats)

    logger.info(f"🧠 generated {len(features_seq)} temporal snapshots")

    return features_seq, timestamps_sorted


def handle_nan_filling(
    features_seq: list[NDArray[Any]],
    graph_base: nx.Graph | None,
    data: pd.DataFrame,
    feats: list[str],
) -> list[NDArray[Any]]:
    """Handle NaN detection and filling using graph-based spatial averaging.

    Parameters
    ----------
    features_seq : list
        List of feature arrays for each timestamp.
    graph_base : nx.Graph or None
        Spatial graph for neighbor-based filling.
    data : pd.DataFrame
        Raw data with datetime information.
    feats : list
        List of feature names.

    Returns
    -------
    list
        Updated feature sequence with NaNs filled.
    """
    if graph_base is None:
        logger.info("skipping nan analysis; G_base not provided")
        return features_seq

    total_nans = summarize_feature_nans(features_seq, feature_names=feats)

    if total_nans > 0:
        logger.info("‼️ nans detected in data, need to fill!")
        timestamps = sorted(data["datetime"].unique())[: len(features_seq)]

        features_seq, fill_log = fill_graph_feature_nans_with_log(
            features_seq, graph_base, timestamps=timestamps, feature_names=feats
        )

        logger.info("👀 checking nans after filling!")
        total_nans = summarize_feature_nans(features_seq, feature_names=feats)
        if total_nans > 0:
            logger.info("‼️ nans STILL detected in data, need to check what's going on!")
        else:
            logger.info("✅ NO nans left in data!")
    else:
        logger.info("✅ NO nans detected in data!")

    return features_seq


def preprocess_for_inference(
    data: pd.DataFrame, feature_scaler: Any, args: Any
) -> tuple[torch.Tensor, list[datetime], int, int]:
    """Transform raw data into model-ready tensors with scaling.

    Parameters
    ----------
    data : pd.DataFrame
        Raw meteorological data with datetime, lat, lon, and feature columns.
    feature_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for feature normalization.
    args : Namespace
        Configuration object containing model and data parameters.

    Returns
    -------
    tuple
        - X_test : torch.Tensor
            Model input tensor of shape (1, input_window, num_nodes, in_channels).
        - timestamps_used : list
            List of timestamps used in the input window.
        - in_channels : int
            Number of input channels/features.
        - num_nodes : int
            Number of spatial nodes.
    """
    logger.info("starting preprocessing for inference!")

    # get needed vars from args
    input_window = args.num_hours_to_fetch
    feats = args.features
    celsius = args.use_celsius.lower() in ["y", "yes", "1"]

    logger.info(f"feats to be used: {feats}\ncelsius: {celsius}")

    # load graph for filling nans via neighbours
    graph_base = None
    logger.info("loading graph!")
    try:
        with open(args.graph.G_base_path, "rb") as f:
            graph_base = pickle.load(f)
    except Exception as e:
        logger.error("❌ failed to load G_base graph:")
        logger.error(f"exception: {type(e).__name__}: {e}")

    logger.info("now building per-timestamp feature sequecne")

    # build per-timestamp feature sequences
    features_seq, timestamps_sorted = build_feature_sequences(data, feats, celsius)

    # handle NaN filling
    features_seq = handle_nan_filling(features_seq, graph_base, data, feats)

    # function to scale features while preserving lat/lon
    def scale_with_geo(
        features_raw: NDArray[Any], scaler: Any, num_pos: int = 2
    ) -> NDArray[np.float32]:
        features_scaled = features_raw.copy()
        features_scaled[:, :-num_pos] = scaler.transform(features_raw[:, :-num_pos])
        return features_scaled.astype(np.float32)

    # apply scaling
    features_scaled = [scale_with_geo(f, feature_scaler) for f in features_seq]

    logger.info(f"📐 final per-hour feature shape: {features_scaled[0].shape}")

    # pack into final model input
    # last input_window hours → model input
    assert len(features_scaled) >= input_window, "❌ not enough timestamps"

    x_window = features_scaled[-input_window:]
    x_np = np.stack(x_window)
    X_test = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)

    in_channels = X_test.shape[-1]
    num_nodes = X_test.shape[2]

    return (
        X_test,  # model input
        timestamps_sorted[-input_window:],  # timestamps used
        in_channels,
        num_nodes,
    )
