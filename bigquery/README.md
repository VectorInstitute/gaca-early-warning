# BigQuery Evaluation Database

Schema definitions and setup for GACA evaluation storage in BigQuery.

## Setup

```bash
./bigquery/setup.sh coderd
```

This creates:
- `gaca_evaluation.predictions` - Model predictions (partitioned by run_timestamp, clustered by horizon_hours)
- `gaca_evaluation.ground_truth` - NOAA observations (partitioned by timestamp)
- `gaca_evaluation.evaluation_metrics` - Computed RMSE/MAE metrics

## Why BigQuery?

- **Fast**: Aggregations over millions of rows in seconds
- **Cheap**: ~$0.02/GB/month storage, $6.25/TB queries
- **Optimized**: Partitioned tables scan only relevant dates
- **Scalable**: Bulk load 100K+ rows in seconds

## Schema Files

- `schemas/bigquery_schema_predictions.json` - Predictions table
- `schemas/bigquery_schema_ground_truth.json` - Ground truth table
- `schemas/bigquery_schema_metrics.json` - Metrics table

## Usage

The evaluation system automatically loads data to BigQuery during batch predictions:

```bash
python scripts/generate_historical_predictions.py
```

View results at `/evaluation` endpoint.
