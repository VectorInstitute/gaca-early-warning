###########################################################################################
# INFERENCE PIPELINE - GACA NOAA GNN
###########################################################################################
# authored nov 14, 2025 by jelshawa
# purpose: to auto-run entire pipeline from start (data fetch) to finish (preds + plots)
###########################################################################################

import os
import pickle
import time
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from contextlib import contextmanager

from util.logger import setup_logging, logger
from util.data_extraction import fetch_last_hours
from util.config import get_args
from util.preprocessing import preprocess_for_inference
from util.plotting import plot_inference_maps, plot_prediction_timeseries

from model.gcngru import GCNGRU 

@contextmanager
def timer(name="Block"):
    """
    timer context for logging time just so we know how long each block takes
    """
    start = time.time()
    yield
    end = time.time()
    logger.info(f"[⏱️ {name}] took {end - start:.2f} sec")


def main():
    """
    main func that runs everything from start to finish
    """
    # setup args + logger + device 
    args = get_args()
    setup_logging(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("🔥 starting inference pipeline!")
    logger.info(f"📂 run dir: {args.run_dir}")

    # load scalers + graph using config paths
    logger.info("📦 loading pretrained artifacts...")

    feature_scaler = joblib.load(f"{args.model.feature_scaler_path}")
    target_scaler  = joblib.load(f"{args.model.target_scaler_path}")

    edge_index  = torch.load(args.graph.edge_index_path, map_location='cpu')
    edge_weight = torch.load(args.graph.edge_weight_path, map_location='cpu')
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    nodes_df = pd.read_csv(f"{args.graph.nodes_csv_path}")

    # load model
    logger.info("🧠 loading trained model weights...")
    
    checkpoint = torch.load(args.model.model_path, map_location=device)
    
    model_class = checkpoint["model_class"]
    model_args  = checkpoint["model_args"]
    raw_state = checkpoint["model_state_dict"]
    
    clean_state = {} 
    for k, v in raw_state.items(): # need to clean up state dict before using bc wrapped with ddp
        if k.startswith("module."):
            clean_state[k.replace("module.", "", 1)] = v
        else:
            clean_state[k] = v
            
    # adjust args for expected model params
    if "pred_offsets" in model_args:
        model_args["pred_horizons"] = len(model_args["pred_offsets"])
        del model_args["pred_offsets"]
        del model_args["model_name"]
    
    # get num_nodes for model also
    model_args["num_nodes"] = edge_index.max().item() + 1 
    
    logger.info(f"🔧 using patched model_args: {model_args}")
    
    if model_class == "DistributedDataParallel":
        # if wrapped, use real class name stored in args
        model_class = args.model_arch 
        
    if model_class == "GCNGRU":
        model = GCNGRU(**model_args)
    else:
        raise ValueError(f"unknown model: {model_class}")
    
    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()
    
    logger.info(f"loaded model {model_class} with args: {model_args}")
    
    # load noaa data via api
    with timer("Load NOAA data"):
        df, latest_ts = fetch_last_hours(args)
        
    logger.info(f"\n====================== ✅ extraction successful ======================")

    logger.info(f"🧪 loaded df with shape: {df.shape}")
    logger.info(f"total rows extracted: {len(df)}")
    # logger.info(f"\nunique timestamps: {df['datetime'].unique()}")
    logger.info(f"latest timestamp: {latest_ts}")

    unique_locs = df[['lat', 'lon']].drop_duplicates()
    logger.info(f"num unique locations: {len(unique_locs)}")
    
    # logger.info("\ndf head:\n")
    # logger.info(df.head())
    logger.info(f"timestamps: {df['datetime'].min()} → {df['datetime'].max()}")

    # preprocess noaa data before predicting
    with timer("Preprocessing"):
        (X_seq, timestamps, in_channels, num_nodes) = preprocess_for_inference(
                                                                                data = df,
                                                                                feature_scaler = feature_scaler,
                                                                                args = args
                                                                              )
        
    logger.info(f"📐 model input shape: {X_seq.shape}")
    X_test = X_seq.to(device)

    # run model pred
    with timer("Model run"):
        with torch.no_grad():
            preds_scaled = model(X_test, edge_index, edge_weight)
        
    preds_np = preds_scaled.cpu().numpy()
    
    preds_unscaled = target_scaler.inverse_transform(preds_np.reshape(-1, preds_np.shape[-1])).reshape(preds_np.shape)
        
    logger.info(f"preds shape: {preds_unscaled.shape}")
    logger.info(f"sample preds: {preds_unscaled[0, :10, 0]}")
    
    # save preds just in case we need them for the dahsboard later
    out_csv = os.path.join(args.run_dir, "predictions.csv")
    rows = []

    for h_idx, horizon in enumerate(args.pred_offsets):
        forecast_time = latest_ts + timedelta(hours=horizon)
    
        for node_idx in range(num_nodes):
            lat = nodes_df.iloc[node_idx]["lat"]
            lon = nodes_df.iloc[node_idx]["lon"]
    
            pred_val = preds_unscaled[0, h_idx, node_idx, 0]
    
            rows.append([
                forecast_time,
                horizon,
                lat,
                lon,
                pred_val
            ])
    
    out_df = pd.DataFrame(
        rows,
        columns=["forecast_time", "horizon_hours", "lat", "lon", "predicted_temp"]
    )
    out_df.to_csv(out_csv, index=False)
    logger.info(f"saved inference results → {out_csv}")

    # plot results
    if args.make_plots.lower() in ["y", "yes", "1"]:
        plot_dir = os.path.join(args.run_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        logger.info("generating gnn-style inference maps...")
        plot_inference_maps(
            preds_unscaled=preds_unscaled,
            nodes_df=nodes_df,
            pred_offsets=args.pred_offsets,
            out_dir=os.path.join(plot_dir, "maps"),
            latest_timestamp=latest_ts
        )
        
        logger.info("generating timeseries plots...")
        plot_prediction_timeseries(
            preds_unscaled=preds_unscaled,
            pred_offsets=args.pred_offsets,
            out_dir=os.path.join(plot_dir, "timeseries"),
            latest_timestamp=latest_ts
        )
        logger.info("🖼️ plots generated!")

    
    logger.info("🎉 inference completed successfully!")


# start
if __name__ == "__main__":
    main()
