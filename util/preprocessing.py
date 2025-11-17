###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to define preprocessing functions to set up data for inference
###########################################################################################

import os
import numpy as np
import pandas as pd
import torch
import networkx as nx
from itertools import combinations
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

from util.logger import logger
import re 
import gc

import numpy as np
import networkx as nx
from geopy.distance import geodesic

import pickle

def summarize_feature_nans(features_sequence, feature_names=None):
    """
    func to summarize number of nans in data - also used to check whether we should be filling in nans
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
    
def fill_graph_feature_nans_with_log(features_sequence, G_base, timestamps=None, feature_names=None):
    """
    func that fills in missing values using mean of neigbour vals
    """
    filled = 0
    fill_log = []  # store (time, node, feature, value, neighbour vals)

    for t, feature_matrix in enumerate(features_sequence):
        for node in range(feature_matrix.shape[0]):
            for f in range(feature_matrix.shape[1]):
                if np.isnan(feature_matrix[node, f]):
                    neighbours = list(G_base.neighbors(node))
                    neighbour_vals = [
                        feature_matrix[n, f] for n in neighbours
                        if not np.isnan(feature_matrix[n, f])
                    ]
                    if neighbour_vals:
                        filled_val = np.mean(neighbour_vals)
                        features_sequence[t][node, f] = filled_val
                        time_label = timestamps[t] if timestamps is not None else f"T={t}"
                        feature_label = feature_names[f] if feature_names else f"F{f}"
                        fill_log.append((time_label, node, feature_label, filled_val, neighbour_vals))
                        filled += 1

    logger.info(f"🩹 filled {filled} nans using spatial neighbors")
    
    # summary of filled entries
    n = 5
    for entry in fill_log[:n]:  # print only first n just to double check
        logger.info(f"🧾 filled nan at {entry[0]}, node {entry[1]}, feature '{entry[2]}' → value: {entry[3]:.3f} (neighbour vals: {neighbour_vals})")
        
    if filled > n:
        logger.info(f"… and {filled - n} more fills.")
        
    return features_sequence, fill_log

def build_graph_from_nodes(nodes_df, distance_threshold=4, inverse_distance=False):
    """
    func that reconstructs the training graph using nodes_latlon.csv (have from training)
    """
    coords = nodes_df[['lat', 'lon']].to_numpy()
    N = len(coords)

    G = nx.Graph()

    # add nodes with pos attribute
    for i in range(N):
        lat, lon = coords[i]
        G.add_node(i, pos=(lat, lon))

    # add edges based on geodesic distance
    for i in range(N):
        for j in range(i + 1, N):
            dist = geodesic(coords[i], coords[j]).km
            if dist <= distance_threshold:
                if inverse_distance:
                    weight = 1.0 / (dist + 1e-5)
                else:
                    weight = dist
                G.add_edge(i, j, weight=weight)

    return G


def preprocess_for_inference(data, feature_scaler, args):
    """
    func that fully preprocesses noaa input data so its ready to be used for inference
    """
    logger.info("starting preprocessing for inference!")

    # get needed vars from args
    pred_offsets = args.pred_offsets       
    input_window = args.num_hours_to_fetch
    feats = args.features
    celsius = args.use_celsius.lower() in ["y", "yes", "1"]
        
    logger.info(f"feats to be used: {feats}\ncelsius: {celsius}")
    
    # load graph for filling nans via neighbours 
    G_base = None
    logger.info(f"loading graph!")
    try:
        with open(args.graph.G_base_path, "rb") as f:
            G_base = pickle.load(f)
    except Exception as e:
        logger.error("❌ failed to load G_base graph:")
        logger.error(f"exception: {type(e).__name__}: {e}")

    logger.info(f"now building per-timestamp feature sequecne")
    
    # pos encoding based on layout
    first_datetime = data['datetime'].min()

    constant_data = (
        data[data['datetime'] == first_datetime]
        [['lat', 'lon']]
        .sort_values(['lat', 'lon'])
        .reset_index(drop=True)
    )

    pos_encodings = constant_data[['lat','lon']].to_numpy(np.float32) * 0.01

    # build per-timestamp feature sequence
    timestamp_dfs = {
        ts: grp.sort_values(['lat','lon']).reset_index(drop=True)
        for ts, grp in data.groupby('datetime')
    }

    timestamps_sorted = sorted(timestamp_dfs.keys())
    features_seq = []

    for ts in timestamps_sorted:
        hourly_df = timestamp_dfs[ts]

        # enforce node ordering consistency
        if not np.allclose(
            hourly_df[['lat','lon']].values,
            constant_data[['lat','lon']].values
        ):
            raise ValueError(f"[{ts}] lat/lon mismatch!")

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

    if G_base is not None:
        total_nans = summarize_feature_nans(features_seq, feature_names=feats)
    
        if total_nans > 0:
            logger.info("‼️ nans detected in data, need to fill!")
            timestamps = sorted(data['datetime'].unique())[:len(features_seq)]
    
            features_seq, fill_log = fill_graph_feature_nans_with_log(
                features_seq, G_base,
                timestamps=timestamps,
                feature_names=feats
            )
            
            logger.info("👀 checking nans after filling!")
            total_nans = summarize_feature_nans(features_seq, feature_names=feats)
            if total_nans > 0: logger.info("‼️ nans STILL detected in data, need to check what's going on!")
            else: logger.info("✅ NO nans left in data!")
        else:
            logger.info("✅ NO nans detected in data!")
    else:
        logger.info("skipping nan analysis; G_base not provided")

    # function to scale features while preserving lat/lon
    def scale_with_geo(features_raw, scaler, num_pos=2):
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
    num_nodes   = X_test.shape[2]

    return (
        X_test,                              # model input
        timestamps_sorted[-input_window:],   # timestamps used
        in_channels,
        num_nodes
    )
