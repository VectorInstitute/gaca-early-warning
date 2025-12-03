<div align="center">

<img src="app/public/vector-logo.webp" alt="Vector Institute" width="200"/>

# Global AI Alliance for Climate Action

## High-Resolution Temperature Forecasting

### NOAA URMA → Graph Neural Network (GCN-GRU) Temperature Forecasting

</div>

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/gaca-early-warning/actions/workflows/docs.yml)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/gaca-early-warning)

Automated production forecasting system for Southwestern Ontario. Features hourly GCNGRU predictions stored in BigQuery with rolling 30-day evaluation metrics.

**Key Features:**
- **Automated Forecasts**: Hourly execution at :15 past each hour
- **Multi-Horizon**: 1, 6, 12, 18, 24, 36, 48-hour predictions
- **Live Dashboard**: Real-time forecast visualization with auto-refresh
- **CLI Tools**: Manual inference and batch prediction capabilities
- **Evaluation**: Daily rolling 30-day RMSE/MAE metrics

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

## Automated Forecasting System

The backend runs automated hourly forecasts and daily evaluations using APScheduler:

### Production Deployment

**Forecasting Schedule:**
- Runs hourly at :15 (after NOAA data availability)
- Automatically stores predictions to BigQuery
- CSV outputs saved to configurable directory

**Evaluation Schedule:**
- Runs daily at 00:30 UTC
- Computes rolling 30-day metrics (RMSE/MAE)
- SQL-based aggregation for efficiency

**Dashboard:**
- Auto-refreshes every 30 minutes
- Smart polling checks for new data before fetching
- Displays scheduler status and last update time
- Manual refresh button available

**Environment Variables:**
```bash
FORECAST_OUTPUT_DIR=forecasts/         # Output directory for CSVs
GCP_PROJECT_ID=your-project-id         # BigQuery project (optional)
BIGQUERY_DATASET=gaca_evaluation       # BigQuery dataset name
```

### CLI Usage

The CLI remains unchanged and works independently of the automated system:

```bash
# Single forecast
gaca-ews predict --config config.yaml --output results/

# Batch predictions for date range
gaca-ews batch-predict --start-date "2024-02-06 12:00" --end-date "2024-02-10 12:00" --interval 24

# Model information
gaca-ews info

# Version
gaca-ews version
```

---

## Author

**Joud El-Shawa** - Vector Institute for AI & Western University

*This repository is intended for early deployment testing of the GACA forecasting pipeline.*
