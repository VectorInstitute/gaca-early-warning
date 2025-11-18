# 🚨 GACA Early Warning System
### **NOAA URMA → Graph Neural Network (GCN-GRU) Temperature Forecasting**

This repository contains the **modular inference pipeline** for the **GACA Early Warning System**, a high-resolution Graph Neural Network framework that generates localized temperature forecasts from **NOAA URMA** meteorological data.

The pipeline takes the most recent URMA hourly fields, preprocesses them into graph-structured tensors, and produces **multi-horizon (1–48h)** temperature predictions across thousands of spatial nodes in Southwestern Ontario.

---

## Repository Structure

- `model/` → trained GNN model (.pth) + feature/target scalers
- `data/` → inference graph (edge_index, edge_weight, node lat/lon)
- `util/` → preprocessing, plotting, config loader, extraction utilities
- `main.py` → end-to-end inference runner (fetch → preprocess → predict → save)
- `gen_maps_from_preds.ipynb` → notebook for testing + visualizing predictions
- `config.yaml` → main configuration for paths + model settings

⚠️ **Important:** Update the paths inside **`config.yaml`** before running `main.py`, otherwise results will be saved inside the repo.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires **Python 3.12+**.

### Prerequisites

If you don't have `uv` installed, install it using:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
# Clone the repository
git clone https://github.com/VectorInstitute/gaca-early-warning.git
cd gaca-early-warning

# Install all dependencies
uv sync --dev --group docs

# Activate the virtual environment
source .venv/bin/activate
```

---

## Author

**Joud El-Shawa** - Vector Institute for AI & Western University

*This repository is intended for early deployment testing of the GACA forecasting pipeline.*
