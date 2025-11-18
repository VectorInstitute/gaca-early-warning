# GACA Early Warning System - Developer Guide

## Overview

The GACA EWS is a professional Python package with three interfaces:
1. **Modern CLI** - Built with Typer and Rich for beautiful terminal output
2. **FastAPI Backend** - REST API for programmatic access
3. **Next.js Dashboard** - (In progress) Web UI for visualization

All interfaces share the same core inference engine, ensuring code reuse and consistency.

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

**Response:**
```json
{
  "message": "GACA Early Warning System API",
  "version": "0.1.0",
  "status": "online"
}
```

#### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET /model/info
Model configuration and information

**Response:**
```json
{
  "model_architecture": "GCNGRU",
  "num_nodes": 14005,
  "input_features": ["t2m", "d2m", "u10", "v10", "sp", "orog"],
  "prediction_horizons": [1, 6, 12, 18, 24, 36, 48],
  "region": {
    "lat_min": 42.0,
    "lat_max": 45.0,
    "lon_min": -81.0,
    "lon_max": -78.0
  },
  "status": "loaded"
}
```

#### POST /predict
Run full inference pipeline

**Request:**
```json
{
  "num_hours": 24
}
```

**Process:**
1. Fetches 24 hours of NOAA URMA data
2. Preprocesses features with spatial consistency
3. Runs GCNGRU model for 7 forecast horizons
4. Returns predictions for all 14,005 grid points

**Response:** Array of 98,035 predictions (14,005 nodes × 7 horizons)
```json
[
  {
    "forecast_time": "2025-11-18T09:00:00",
    "horizon_hours": 1,
    "lat": 42.00008,
    "lon": -80.84195,
    "predicted_temp": 2.97
  },
  ...
]
```

#### GET /predictions/latest
Retrieve most recent predictions from file

**Response:**
```json
{
  "count": 98035,
  "predictions": [...],
  "file_timestamp": "2025-11-18T10:00:00"
}
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
