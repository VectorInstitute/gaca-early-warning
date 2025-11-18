# GACA Early Warning System - Developer Guide

## Overview

The project has been restructured into a professional package with a Next.js dashboard and FastAPI backend for production deployment.

## Directory Structure

```
gaca-early-warning/
├── src/gaca_ews/              # Main Python package
│   ├── __init__.py
│   ├── core/                  # Core utilities
│   │   ├── config.py         # Configuration management
│   │   ├── data_extraction.py # NOAA data fetching
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── logger.py          # Logging utilities
│   │   └── plotting.py        # Visualization
│   ├── model/                 # Model definitions
│   │   ├── gcngru.py         # GCNGRU architecture
│   │   ├── final_model.pth    # Trained model checkpoint
│   │   ├── feature_scaler.pkl # Feature scaler
│   │   └── target_scaler.pkl  # Target scaler
│   └── cli/                   # Command-line interface
│       └── main.py            # Inference pipeline CLI
├── backend/                   # FastAPI backend
│   ├── app/
│   │   ├── main.py           # API routes and endpoints
│   │   └── __init__.py
│   ├── test_api.py           # Full API test suite
│   └── requirements.txt
├── app/                       # Next.js dashboard
│   ├── app/                   # App router pages
│   │   ├── layout.tsx        # Root layout
│   │   ├── page.tsx          # Home page
│   │   └── globals.css       # Global styles
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── next.config.ts
├── data/                      # Graph and node data
│   ├── edge_index.pt
│   ├── edge_weight.pt
│   ├── nodes_latlon.csv
│   └── G_base.pkl
├── model/                     # Legacy model artifacts
├── tests/                     # Unit tests
├── config.yaml                # Pipeline configuration
├── pyproject.toml             # Package configuration
└── README.md

```

## Package Installation

The project is now a proper Python package called `gaca-ews`:

```bash
# Install with development dependencies
uv sync --all-extras --dev

# The package provides a CLI command
gaca-ews --config config.yaml
```

## API Endpoints

The FastAPI backend provides the following endpoints:

### GET /
- Root endpoint with API information
- Returns: `{message, version, status}`

### GET /health
- Health check endpoint
- Returns: `{status, model_loaded}`

### GET /model/info
- Model information and configuration
- Returns: Model architecture, features, horizons, region

### POST /predict
- Run full inference pipeline
- Request body: `{num_hours: 24}`  (optional, defaults to 24)
- Process:
  1. Fetches 24 hours of NOAA URMA data
  2. Preprocesses features with spatial consistency
  3. Runs GCNGRU model for 7 forecast horizons
  4. Returns predictions for all 14,005 grid points
- Returns: Array of predictions with structure:
  ```json
  {
    "forecast_time": "2025-11-18T09:00:00",
    "horizon_hours": 1,
    "lat": 42.00008,
    "lon": -80.84195,
    "predicted_temp": 2.97
  }
  ```

### GET /predictions/latest
- Retrieve most recent predictions from file
- Returns: Predictions from last inference run

## Testing Results

### API Validation ✅

**Total Predictions**: 98,035 (14,005 locations × 7 horizons)

**Forecast Horizons**: [1, 6, 12, 18, 24, 36, 48] hours

**Predictions per Horizon**:
- 1h: 14,005 locations
- 6h: 14,005 locations
- 12h: 14,005 locations
- 18h: 14,005 locations
- 24h: 14,005 locations
- 36h: 14,005 locations
- 48h: 14,005 locations

**Geographic Coverage**:
- Latitude: 42.00° to 45.00° N
- Longitude: -81.00° to -78.00° W
- Region: Southwestern Ontario

**Temperature Statistics** (November 18, 2025):
- Minimum: -7.3°C
- Maximum: 8.7°C
- Mean: 2.6°C

### Data Quality Checks ✅

- ✓ All required fields present in predictions
- ✓ Geographic bounds match configuration
- ✓ Temperature values within reasonable range
- ✓ Forecast horizons match model configuration
- ✓ Spatial consistency maintained across timestamps
- ✓ No NaN values in predictions

## Running the System

### 1. Start the FastAPI Backend

```bash
# Option A: Using uvicorn directly
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Option B: Using the run script
bash backend/run_server.sh

# Option C: With auto-reload for development
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. Start the Next.js Dashboard (Optional)

```bash
cd app
npm run dev
```

Dashboard will be available at: http://localhost:3000

### 3. Run Inference

```bash
# Via API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"num_hours": 24}'

# Via CLI
gaca-ews --config config.yaml
```

## Testing

### Run Unit Tests
```bash
uv run pytest -m "not integration_test" --cov . tests
```

### Run Full API Test
```bash
uv run python backend/test_api.py
# Note: Will prompt before running expensive inference test
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Run inference (takes ~45 seconds)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"num_hours": 24}' \
  -o predictions.json
```

## Dependencies

### Core Package
- Python >=3.12
- torch, torch-geometric (GNN framework)
- scikit-learn==1.7.1 (pinned to match trained models)
- pandas, numpy (data processing)
- boto3 (NOAA data fetching)
- pygrib (GRIB2 file parsing)
- fastapi, uvicorn (API framework)

### Dashboard
- Next.js 16
- React 19
- TypeScript 5
- Tailwind CSS 4

## Key Features

1. **Modular Package Structure**: Clean separation of concerns with `core`, `model`, and `cli` modules
2. **FastAPI Backend**: Production-ready REST API with automatic documentation
3. **NOAA Integration**: Automatic fetching of real-time meteorological data
4. **GNN-based Forecasting**: GCNGRU model for spatiotemporal predictions
5. **Multi-horizon Predictions**: 1, 6, 12, 18, 24, 36, and 48-hour forecasts
6. **Spatial Consistency**: Graph-based preprocessing ensures valid spatial relationships
7. **Type Safety**: Full type hints and validation with Pydantic
8. **CORS Support**: Ready for frontend integration
9. **Next.js Dashboard**: Modern UI framework for visualization

## Configuration

The `config.yaml` file controls all pipeline parameters:

```yaml
model:
  feature_scaler_path: "./model/feature_scaler.pkl"
  target_scaler_path: "./model/target_scaler.pkl"
  model_path: "./model/final_model.pth"

graph:
  edge_index_path: "./data/edge_index.pt"
  edge_weight_path: "./data/edge_weight.pt"
  nodes_csv_path: "./data/nodes_latlon.csv"
  G_base_path: "./data/G_base.pkl"

region:
  lat_min: 42.0  # DO NOT CHANGE - tied to trained model
  lat_max: 45.0
  lon_min: -81.0
  lon_max: -78.0

pred_offsets: [1, 6, 12, 18, 24, 36, 48]
features: ["t2m", "d2m", "u10", "v10", "sp", "orog"]
model_arch: "GCNGRU"
```

## Next Steps

1. **Dashboard Development**: Implement visualization of predictions in Next.js
2. **Caching**: Add Redis or similar for prediction caching
3. **Monitoring**: Add Prometheus metrics and health monitoring
4. **Authentication**: Implement API key or OAuth for production
5. **Rate Limiting**: Add rate limiting for API endpoints
6. **Docker**: Containerize the application
7. **CI/CD**: Set up automated testing and deployment

## Notes

- The scikit-learn version is pinned to 1.7.1 to match the trained scaler artifacts
- NOAA data fetching requires internet access and may take 15-45 seconds
- Model inference takes ~5 seconds on CPU for all 14,005 nodes
- Predictions are automatically saved to `{run_dir}/predictions.csv`
- The API server loads all model artifacts on startup (~5 seconds)
