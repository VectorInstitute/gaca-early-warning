#!/bin/bash
# BigQuery setup script for GACA Early Warning System evaluation database
#
# This script creates the BigQuery dataset and tables needed for storing
# predictions, ground truth, and evaluation metrics.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - BigQuery API enabled in GCP project
#   - Appropriate permissions (BigQuery Admin or Data Editor)
#
# Usage:
#   ./bigquery/setup.sh <project_id>
#
# Example:
#   ./bigquery/setup.sh coderd

set -e  # Exit on error

if [ -z "$1" ]; then
    echo "Error: Project ID required"
    echo "Usage: $0 <project_id>"
    exit 1
fi

PROJECT_ID=$1
DATASET="gaca_evaluation"
LOCATION="us-central1"

echo "========================================="
echo "BigQuery Setup for GACA Evaluation"
echo "========================================="
echo "Project: $PROJECT_ID"
echo "Dataset: $DATASET"
echo "Location: $LOCATION"
echo ""

# Create dataset
echo "[1/4] Creating dataset..."
bq --project_id="$PROJECT_ID" mk --dataset --location="$LOCATION" "$DATASET" || echo "Dataset already exists"

# Create predictions table (partitioned by run_timestamp, clustered by horizon_hours)
echo "[2/4] Creating predictions table..."
bq --project_id="$PROJECT_ID" mk --table \
    --time_partitioning_field=run_timestamp \
    --time_partitioning_type=DAY \
    --clustering_fields=horizon_hours \
    "${DATASET}.predictions" \
    bigquery/schemas/bigquery_schema_predictions.json \
    || echo "Predictions table already exists"

# Create ground_truth table (partitioned by timestamp)
echo "[3/4] Creating ground_truth table..."
bq --project_id="$PROJECT_ID" mk --table \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    "${DATASET}.ground_truth" \
    bigquery/schemas/bigquery_schema_ground_truth.json \
    || echo "Ground truth table already exists"

# Create evaluation_metrics table
echo "[4/4] Creating evaluation_metrics table..."
bq --project_id="$PROJECT_ID" mk --table \
    "${DATASET}.evaluation_metrics" \
    bigquery/schemas/bigquery_schema_metrics.json \
    || echo "Evaluation metrics table already exists"

echo ""
echo "========================================="
echo "✓ BigQuery setup complete!"
echo "========================================="
echo ""
echo "Tables created:"
echo "  - ${DATASET}.predictions (partitioned by run_timestamp)"
echo "  - ${DATASET}.ground_truth (partitioned by timestamp)"
echo "  - ${DATASET}.evaluation_metrics"
echo ""
echo "Next steps:"
echo "  1. Run batch predictions: python scripts/generate_historical_predictions.py"
echo "  2. Predictions will be automatically loaded to BigQuery"
echo "  3. View evaluation metrics at /evaluation endpoint"
echo ""
