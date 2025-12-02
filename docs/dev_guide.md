# GACA Early Warning System - Developer Guide

## Overview

The GACA EWS is a professional Python package with three interfaces:
1. **Modern CLI** - Built with Typer and Rich for beautiful terminal output
2. **FastAPI Backend** - Automated forecasting with REST API
3. **Next.js Dashboard** - Real-time forecast visualization

All interfaces share the same core inference engine, ensuring code reuse and consistency.

**Production Mode**: Backend runs automated hourly forecasts (at :15) and daily evaluations (00:30 UTC) using APScheduler. Predictions are stored in BigQuery. Dashboard auto-refreshes every 30 minutes with smart polling to minimize costs.

## Architecture: Code Reuse & Modularity

The system follows DRY (Don't Repeat Yourself) principles with a **single source of truth** for inference logic:

```
src/gaca_ews/
├── core/
│   ├── inference.py       ← InferenceEngine: Single source of truth
│   ├── data_extraction.py
│   ├── preprocessing.py
│   ├── config.py
│   ├── logger.py
│   └── plotting.py
├── model/
│   ├── gcngru.py
│   └── *.pkl, *.pth       ← Model artifacts
└── cli/
    └── main.py            ← CLI wrapper using InferenceEngine

backend/
└── app/
    └── main.py            ← FastAPI wrapper using InferenceEngine

app/                       ← Next.js dashboard (future)
```

### InferenceEngine Class

All model loading and prediction logic is centralized in `src/gaca_ews/core/inference.py`:

```python
from gaca_ews.core.inference import InferenceEngine

# Initialize once
engine = InferenceEngine("config.yaml")
engine.load_artifacts()  # Load model, scalers, graph

# Run inference
predictions, latest_ts = engine.run_full_pipeline()

# Or step-by-step
data, latest_ts = engine.fetch_data()
X = engine.preprocess(data)
predictions = engine.predict(X)

# Save results
engine.save_predictions(predictions, latest_ts, "output.csv")
engine.generate_plots(predictions, latest_ts, "plots/")
```

**Key Benefits:**
- ✅ **Zero code duplication** - CLI and API both use InferenceEngine
- ✅ **Single point of change** - Bug fixes/updates apply everywhere
- ✅ **Consistent behavior** - Same predictions from CLI and API
- ✅ **Easy testing** - Test one class, all interfaces work
- ✅ **Type-safe** - Full Python 3.12+ type hints

## Directory Structure

```
gaca-early-warning/
├── src/gaca_ews/              # Main Python package
│   ├── __init__.py
│   ├── core/                  # Core inference logic
│   │   ├── inference.py      # ← InferenceEngine (shared by all)
│   │   ├── config.py
│   │   ├── data_extraction.py
│   │   ├── preprocessing.py
│   │   ├── logger.py
│   │   └── plotting.py
│   ├── model/                 # Model and artifacts
│   │   ├── gcngru.py
│   │   ├── final_model.pth
│   │   ├── feature_scaler.pkl
│   │   └── target_scaler.pkl
│   └── cli/                   # Command-line interface
│       └── main.py
├── backend/                   # FastAPI backend
│   ├── app/
│   │   ├── main.py           # API routes (uses InferenceEngine)
│   │   └── __init__.py
│   └── test_api.py
├── app/                       # Next.js dashboard
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── package.json
│   └── tsconfig.json
├── data/                      # Graph and node data
│   ├── edge_index.pt
│   ├── edge_weight.pt
│   ├── nodes_latlon.csv
│   └── G_base.pkl
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── config.yaml                # Pipeline configuration
└── pyproject.toml             # Package configuration
```

## Installation

```bash
# Install with all dependencies
uv sync --all-extras --dev

# CLI is automatically installed
gaca-ews --help
```

## CLI Usage

### Commands

#### 1. `predict` - Run Inference

```bash
# Basic usage
gaca-ews predict

# With options
gaca-ews predict --config config.yaml --output results/ --verbose

# Without plots (faster)
gaca-ews predict --no-plots

# Full options
gaca-ews predict \
  --config config.yaml \
  --output predictions/ \
  --plots \
  --csv \
  --verbose
```

**Options:**
- `-c, --config PATH` - Configuration file (default: `config.yaml`)
- `-o, --output PATH` - Output directory (default: from config)
- `--plots/--no-plots` - Generate visualizations (default: yes)
- `--csv/--no-csv` - Save CSV file (default: yes)
- `-v, --verbose` - Detailed output

**Example Output:**
```
╭───────────────────────────────────╮
│ 🌡️  GACA Early Warning System     │
│ Temperature Forecasting Pipeline │
╰───────────────────────────────────╯

⠋ Loading model artifacts...  ━━━━━━  0:00:05
⠙ Fetching NOAA meteorological data...
  Fetched 336,120 rows • Latest: 2025-11-18 08:00 UTC
⠹ Preprocessing features...
  Input shape: torch.Size([1, 24, 14005, 8])
⠸ Running model inference...
  Output shape: (1, 7, 14005, 1)
⠼ Saving predictions to CSV...
⠴ Generating visualization plots...

╭─────────────────────────────────────╮
│ ✓ Prediction Complete!              │
│                                     │
│ 📊 98,035 predictions               │
│ 📍 14,005 locations                 │
│ 🕐 7 time horizons                  │
│ 🌡️  -7.3°C to 8.7°C                 │
│ 📂 predictions/                     │
╰─────────────────────────────────────╯
```

#### 2. `info` - Model Information

```bash
gaca-ews info
```

**Output:**
```
╭───────────────────────────╮
│ 🌡️  GACA Model Information │
╰───────────────────────────╯

         Model Configuration
 Architecture       GCNGRU
 Device             cpu
 Number of Nodes    14,005
 Input Features     t2m, d2m, u10, v10, sp, orog
 Forecast Horizons  [1, 6, 12, 18, 24, 36, 48]

       Coverage Region
 Latitude   42.0° to 45.0°
 Longitude  -81.0° to -78.0°
 Region     Southwestern Ontario
```

#### 3. `version` - Version Information

```bash
gaca-ews version
# or
gaca-ews --version
```

## API Backend

### Starting the Server

```bash
# Development with auto-reload
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### GET /
Root endpoint with API information

#### GET /health
Health check and model loaded status

#### GET /model/info
Model configuration and metadata

#### GET /forecasts/latest-timestamp
Lightweight endpoint to check for new data (minimizes BigQuery costs)

**Response:**
```json
{
  "last_run_timestamp": "2024-12-02T15:15:00",
  "has_data": true
}
```

**Use Case**: Frontend polls this every 30 min to check if new forecast available

#### GET /forecasts/latest
Fetch latest predictions from BigQuery

**Response:**
```json
[
  {
    "forecast_time": "2025-11-18T09:00:00",
    "horizon_hours": 1,
    "lat": 42.00008,
    "lon": -80.84195,
    "predicted_temp": 2.97,
    "run_timestamp": "2025-11-18T08:15:00"
  },
  ...
]
```

#### GET /forecasts/status
Scheduler status and last forecast run information

**Response:**
```json
{
  "scheduler": {
    "is_running": false,
    "scheduler_active": true,
    "last_run_timestamp": "2025-11-18T08:15:00",
    "next_scheduled_run": "2025-11-18T09:15:00"
  },
  "last_forecast": {
    "run_timestamp": "2025-11-18T08:15:00",
    "prediction_count": 98035
  }
}
```

#### GET /scheduler/status
Status of forecast and evaluation schedulers

#### GET /evaluation/static
Static evaluation metrics (Feb-Jul 2024)

**Response:**
```json
{
  "evaluation_period": {
    "start": "2024-02-06T12:00:00",
    "end": "2024-07-19T17:00:00"
  },
  "metrics": {
    "overall": {
      "rmse": 1.234,
      "mae": 0.987,
      "sample_count": 150000
    },
    "by_horizon": {
      "1": {"rmse": 0.8, "mae": 0.6, "sample_count": 21000},
      "6": {"rmse": 1.1, "mae": 0.85, "sample_count": 21000}
    }
  },
  "computed_at": "2025-11-18T10:00:00"
}
```

#### GET /evaluation/dynamic
Rolling 30-day evaluation metrics

**Response:**
```json
{
  "evaluation_window": {
    "start": "2025-10-19T00:00:00",
    "end": "2025-11-18T00:00:00",
    "days": 30
  },
  "metrics": {
    "overall": {
      "rmse": 1.156,
      "mae": 0.923,
      "sample_count": 42000
    },
    "by_horizon": {
      "1": {"rmse": 0.75, "mae": 0.58, "sample_count": 6000}
    }
  },
  "computed_at": "2025-11-18T00:30:00"
}
```

### Setting Up Dynamic Evaluation

Dynamic evaluation requires both predictions and ground truth data for the last 30 days.

#### Initial Setup

1. **Create BigQuery tables** (if not already done):
```bash
./bigquery/setup.sh your-project-id
```

2. **Populate last 30 days of data**:
```bash
# Set your GCP project ID
export GCP_PROJECT_ID=your-project-id

# Populate predictions and ground truth for last 30 days
python scripts/populate_dynamic_evaluation.py

# Or specify a different time window (e.g., last 7 days)
python scripts/populate_dynamic_evaluation.py --days 7

# Use custom interval (e.g., every 12 hours instead of 24)
python scripts/populate_dynamic_evaluation.py --interval 12
```

**What the script does:**
- Generates predictions for the last N days using batch inference
- Stores predictions to BigQuery `predictions` table
- Extracts ground truth from NOAA cache (t2m field)
- Stores ground truth to BigQuery `ground_truth` table

**Options:**
- `--days N` - Number of days to populate (default: 30)
- `--interval N` - Hours between predictions (default: 24)
- `--config PATH` - Path to config file (default: config.yaml)
- `--output DIR` - Output directory for temporary files (default: ./dynamic_eval_data)
- `--skip-inference` - Only load existing CSVs and ground truth
- `-v, --verbose` - Enable verbose output

#### Ongoing Updates

Once set up, the system automatically maintains the rolling window:

- **Hourly Forecasts** (at :15): Add new predictions to BigQuery
- **Daily Evaluation** (at 00:30 UTC): Compute metrics for last 30 days
- **Frontend Auto-refresh** (every 30 min): Fetch updated metrics

The rolling window logic ensures that:
- Old predictions beyond 30 days are still stored but not included in metrics
- New predictions are automatically incorporated into the rolling window
- No manual intervention needed after initial setup

#### Troubleshooting

**No data showing on dashboard:**
```bash
# Check if BigQuery tables exist
bq ls --project_id=your-project-id gaca_evaluation

# Check prediction count
bq query --project_id=your-project-id \
  "SELECT COUNT(*) FROM gaca_evaluation.predictions"

# Check ground truth count
bq query --project_id=your-project-id \
  "SELECT COUNT(*) FROM gaca_evaluation.ground_truth"

# Re-run population script if needed
python scripts/populate_dynamic_evaluation.py --days 30
```

**Metrics not updating daily:**
```bash
# Check evaluation scheduler status
curl http://localhost:8000/scheduler/status

# Manually trigger evaluation via Python
python -c "
from gaca_ews.evaluation.storage import EvaluationStorage
from datetime import datetime, timedelta
storage = EvaluationStorage()
end = datetime.now()
start = end - timedelta(days=30)
metrics = storage.compute_metrics_for_period(start, end)
print(metrics)
"
```

## Testing

### Unit Tests
```bash
# Run all unit tests
uv run pytest -m "not integration_test" --cov . tests

# Specific test file
uv run pytest tests/test_imports.py -v
```

### API Testing
```bash
# Quick health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/model/info

# Run inference (takes ~45 seconds)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"num_hours": 24}' \
  -o predictions.json

# Analyze results
python3 << 'EOF'
import json
with open('predictions.json') as f:
    preds = json.load(f)
print(f"Total: {len(preds):,} predictions")
temps = [p['predicted_temp'] for p in preds]
print(f"Range: {min(temps):.1f}°C to {max(temps):.1f}°C")
EOF
```

### CLI Testing
```bash
# Test info command
gaca-ews info

# Test predict with verbose output
gaca-ews predict --no-plots --verbose
```

## Configuration

The `config.yaml` file controls all pipeline parameters:

```yaml
model:
  feature_scaler_path: "./src/gaca_ews/model/feature_scaler.pkl"
  target_scaler_path: "./src/gaca_ews/model/target_scaler.pkl"
  model_path: "./src/gaca_ews/model/final_model.pth"

graph:
  edge_index_path: "./data/edge_index.pt"
  edge_weight_path: "./data/edge_weight.pt"
  nodes_csv_path: "./data/nodes_latlon.csv"
  G_base_path: "./data/G_base.pkl"

run_dir: "./predictions"
make_plots: "y"

# DO NOT CHANGE - tied to trained model
region:
  lat_min: 42.0
  lat_max: 45.0
  lon_min: -81.0
  lon_max: -78.0

num_hours_to_fetch: 24
pred_offsets: [1, 6, 12, 18, 24, 36, 48]
features: ["t2m", "d2m", "u10", "v10", "sp", "orog"]
model_arch: "GCNGRU"
```

## Development

### Adding New Features

To extend the inference engine:

```python
# In src/gaca_ews/core/inference.py

class InferenceEngine:
    def your_new_method(self) -> Any:
        """Add new functionality here."""
        # Both CLI and API can use it immediately
        pass
```

To add a new CLI command:

```python
# In src/gaca_ews/cli/main.py

@app.command(name="new-command")
def new_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
) -> None:
    """Your command description."""
    engine = InferenceEngine(config)
    # Use engine methods
    console.print("[green]Done![/green]")
```

To add a new API endpoint:

```python
# In backend/app/main.py

@app.get("/new-endpoint")
async def new_endpoint() -> dict:
    """Your endpoint description."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    # Use engine methods
    result = engine.your_new_method()
    return {"result": result}
```

### Code Quality

The project follows strict quality standards:

- **Type hints**: All functions have full type annotations (Python 3.12+ syntax)
- **Docstrings**: NumPy-style docstrings for all public functions
- **Linting**: `ruff` for code formatting and linting
- **Type checking**: `mypy` in strict mode
- **Testing**: `pytest` with coverage tracking

```bash
# Run all quality checks
pre-commit run --all-files

# Individual tools
ruff check .              # Linting
ruff format .             # Formatting
mypy .                    # Type checking
pytest --cov . tests      # Testing with coverage
```

## Dependencies

### Core Package
- Python >=3.12
- **ML**: torch, torch-geometric, scikit-learn==1.7.1
- **Data**: pandas, numpy, joblib
- **API**: fastapi, uvicorn, pydantic
- **CLI**: typer, rich
- **NOAA**: boto3, pygrib
- **Visualization**: matplotlib, networkx

### Dashboard
- Next.js 16
- React 19
- TypeScript 5
- Tailwind CSS 4

## Performance Metrics

### Tested on November 18, 2025

**Inference Pipeline:**
- NOAA data fetch: ~15-45 seconds (network dependent)
- Preprocessing: ~2 seconds
- Model inference: ~5 seconds (CPU)
- Total: ~25-55 seconds

**Output:**
- 98,035 predictions (14,005 locations × 7 horizons)
- Temperature range: -7.3°C to 8.7°C
- CSV file: ~3.5 MB
- Plots: 7 spatial maps + timeseries

**Data Quality:**
- ✅ All required fields present
- ✅ Geographic bounds validated
- ✅ Temperature values reasonable
- ✅ Spatial consistency maintained
- ✅ No NaN values

## Troubleshooting

### Model Not Loading

```bash
# Check config paths
gaca-ews info

# Verify files exist
ls -la src/gaca_ews/model/
ls -la data/
```

### NOAA Data Fetch Failing

```bash
# Run with verbose output
gaca-ews predict --verbose

# Check network
curl -I https://noaa-urma-pds.s3.amazonaws.com
```

### API Server Issues

```bash
# Check if server is running
curl http://localhost:8000/health

# View server logs
uvicorn backend.app.main:app --log-level debug

# Kill existing servers
lsof -ti:8000 | xargs kill -9
```

## Best Practices

1. **Always use InferenceEngine** - Never duplicate inference logic
2. **Type everything** - Use Python 3.12+ type hints (`|` instead of `Union`)
3. **Test before committing** - Run `pre-commit run --all-files`
4. **Update docs** - Keep this guide in sync with code changes
5. **Version pin carefully** - scikit-learn is pinned to match artifacts

## Next Steps

1. **Dashboard Development** - Connect Next.js to FastAPI backend
2. **Caching** - Add Redis for prediction caching
3. **Monitoring** - Add Prometheus metrics
4. **Authentication** - Implement API keys/OAuth
5. **Docker** - Containerize for deployment
6. **CI/CD** - Automate testing and deployment
7. **Horizontal Scaling** - Add load balancing

## Notes

- scikit-learn pinned to 1.7.1 to match trained scaler artifacts
- NOAA data may lag real-time by 1-2 hours
- Model artifacts total ~32 MB
- Graph data is pre-computed and fixed
- Region boundaries are tied to the trained model and cannot be changed
- Positional encodings (lat/lon × 0.01) must remain consistent with training
