"""Populate predictions and ground truth for dynamic evaluation (last 30 days).

This script generates predictions and fetches ground truth for the rolling
30-day window used in dynamic evaluation. Run this to initialize the dynamic
evaluation system or to backfill missing data.

The script:
1. Generates predictions for the last N days using batch-predict CLI
2. Stores prediction CSVs to BigQuery
3. Extracts ground truth from the NOAA cache and stores to BigQuery

Usage:
    # Populate last 30 days (default)
    GCP_PROJECT_ID=your-project python scripts/populate_dynamic_evaluation.py

    # Populate last 7 days only
    python scripts/populate_dynamic_evaluation.py --days 7

    # Skip inference and only load existing CSVs + ground truth
    python scripts/populate_dynamic_evaluation.py --skip-inference
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from rich.console import Console

from gaca_ews.evaluation.storage import EvaluationStorage


console = Console()


def run_batch_predictions(
    start_date: str,
    end_date: str,
    interval_hours: int,
    config: str,
    output_dir: Path,
    verbose: bool = False,
) -> bool:
    """Run batch predictions using the CLI.

    Parameters
    ----------
    start_date : str
        Start date in format "YYYY-MM-DD HH:MM"
    end_date : str
        End date in format "YYYY-MM-DD HH:MM"
    interval_hours : int
        Interval between predictions in hours
    config : str
        Path to config file
    output_dir : Path
        Output directory for CSV files
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "gaca-ews",
        "batch-predict",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--interval",
        str(interval_hours),
        "--config",
        config,
        "--output",
        str(output_dir),
        "--csv",
    ]

    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Batch prediction failed: {e}[/red]")
        return False
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Cancelled by user[/yellow]")
        sys.exit(1)


def store_predictions_to_bigquery(
    output_dir: Path,
    storage: EvaluationStorage,
    verbose: bool = False,
) -> tuple[int, int]:
    """Store generated predictions to BigQuery.

    Parameters
    ----------
    output_dir : Path
        Directory containing prediction CSV files
    storage : EvaluationStorage
        BigQuery storage instance
    verbose : bool
        Enable verbose output

    Returns
    -------
    tuple[int, int]
        (successful_count, failed_count)
    """
    csv_files = sorted(output_dir.glob("predictions_*.csv"))

    if len(csv_files) == 0:
        console.print("[yellow]⚠ No prediction files found[/yellow]")
        return 0, 0

    console.print(
        f"\n[bold]Storing {len(csv_files)} prediction runs to BigQuery...[/bold]\n"
    )

    successful = 0
    failed = 0

    for csv_file in csv_files:
        # Extract timestamp from filename: predictions_YYYYMMDD_HHMM.csv
        try:
            timestamp_str = csv_file.stem.replace("predictions_", "")
            run_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")

            # Read predictions
            df = pd.read_csv(csv_file)

            # Store to BigQuery
            rows_loaded = storage.store_predictions(df, run_timestamp)

            console.print(
                f"[green]✓[/green] {csv_file.name} → BigQuery ({rows_loaded:,} rows)"
            )
            successful += 1

        except Exception as e:
            console.print(f"[red]✗[/red] {csv_file.name}: {str(e)[:50]}")
            failed += 1
            if verbose:
                console.print_exception()

    return successful, failed


def store_ground_truth_to_bigquery(
    cache_file: Path, storage: EvaluationStorage, verbose: bool = False
) -> int:
    """Extract ground truth from NOAA cache and store to BigQuery.

    Parameters
    ----------
    cache_file : Path
        Path to NOAA cache Parquet file
    storage : EvaluationStorage
        BigQuery storage instance
    verbose : bool
        Enable verbose output

    Returns
    -------
    int
        Number of ground truth rows stored
    """
    if not cache_file.exists():
        console.print(f"[yellow]⚠ Cache file not found: {cache_file}[/yellow]")
        console.print("Ground truth will not be loaded.")
        return 0

    console.print("\n[bold]Loading ground truth from cache...[/bold]\n")
    console.print(f"Reading from: {cache_file}")

    try:
        # Read NOAA data using Polars
        console.print("[cyan]Reading Parquet cache...[/cyan]")
        df_pl = pl.read_parquet(cache_file)

        console.print(f"[green]✓[/green] Loaded {len(df_pl):,} rows from cache")

        # Extract ground truth: t2m converted from Kelvin to Celsius
        ground_truth_pl = df_pl.select(
            [
                pl.col("time").alias("timestamp"),
                pl.col("lat"),
                pl.col("lon"),
                (pl.col("t2m") - 273.15).alias("actual_temp"),  # K to °C
            ]
        ).unique()

        console.print(
            f"[cyan]Extracted {len(ground_truth_pl):,} unique ground truth observations[/cyan]"
        )

        # Convert to pandas for BigQuery compatibility
        ground_truth_df = ground_truth_pl.to_pandas()

        # Store to BigQuery
        rows_loaded = storage.store_ground_truth(ground_truth_df)

        console.print(f"[green]✓[/green] Loaded {rows_loaded:,} ground truth records")

        return rows_loaded

    except Exception as e:
        console.print(f"[red]✗ Failed to load ground truth: {e}[/red]")
        if verbose:
            console.print_exception()
        return 0


def main() -> None:
    """Populate predictions and ground truth for dynamic evaluation."""
    parser = argparse.ArgumentParser(
        description="Populate predictions and ground truth for dynamic evaluation"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to populate (default: 30)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval between predictions in hours (default: 1 for unbiased evaluation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dynamic_eval_data",
        help="Output directory for predictions and cache",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip batch prediction (only store existing CSVs and ground truth)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "noaa_data_cache.parquet"

    # Calculate date range for last N days
    # Important: end_date is the last PREDICTION RUN time, not the last FORECAST time
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    console.print("\n[bold cyan]Dynamic Evaluation Data Population[/bold cyan]\n")
    console.print(
        f"Prediction runs: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}"
    )
    console.print(f"Days: {args.days}")
    console.print(f"Interval: {args.interval} hours")
    console.print(f"Output: {output_dir}")
    console.print(
        "\n[dim]Note: Ground truth will be fetched up to end_date + max_forecast_horizon "
        "(+48h)[/dim]\n"
    )

    # Warn about temporal bias
    if args.interval > 1:
        console.print(
            f"[bold yellow]⚠ WARNING: Using interval={args.interval}h may introduce temporal bias![/bold yellow]"
        )
        console.print(
            "[yellow]  Each forecast horizon will be evaluated at only specific times of day,[/yellow]"
        )
        console.print(
            "[yellow]  causing diurnal effects to bias error metrics (e.g., 48h < 36h errors).[/yellow]"
        )
        console.print(
            "[yellow]  For unbiased evaluation, use --interval 1 (hourly predictions).[/yellow]\n"
        )

    # Step 1: Run batch predictions (unless skipped)
    if not args.skip_inference:
        console.print("=" * 70)
        console.print("[bold cyan]Step 1: Running batch predictions[/bold cyan]")
        console.print("=" * 70 + "\n")

        success = run_batch_predictions(
            start_date=start_date.strftime("%Y-%m-%d %H:%M"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M"),
            interval_hours=args.interval,
            config=args.config,
            output_dir=output_dir,
            verbose=args.verbose,
        )

        if not success:
            console.print("[red]✗ Batch prediction failed[/red]")
            sys.exit(1)
    else:
        console.print("[yellow]⊘ Skipping batch prediction[/yellow]\n")

    # Step 2: Store predictions to BigQuery
    console.print("=" * 70)
    console.print("[bold cyan]Step 2: Storing predictions to BigQuery[/bold cyan]")
    console.print("=" * 70)

    try:
        storage = EvaluationStorage()
        console.print(f"[green]✓[/green] Connected to BigQuery: {storage.project_id}\n")

        successful, failed = store_predictions_to_bigquery(
            output_dir, storage, args.verbose
        )

        # Step 3: Store ground truth to BigQuery
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]Step 3: Storing ground truth to BigQuery[/bold cyan]")
        console.print("=" * 70)

        gt_rows = store_ground_truth_to_bigquery(cache_file, storage, args.verbose)

        # Summary
        console.print("\n" + "=" * 70)
        console.print("[bold green]✓ Complete![/bold green]")
        console.print(f"  Predictions stored: {successful}")
        console.print(f"  Predictions failed: {failed}")
        console.print(f"  Ground truth rows: {gt_rows:,}")
        console.print("=" * 70 + "\n")

        if successful > 0 and gt_rows > 0:
            console.print("[green]✓ Dynamic evaluation is now ready![/green]")
            console.print("View at: http://localhost:3000/evaluation\n")
        else:
            console.print(
                "[yellow]⚠ Some data may be missing. Check logs above.[/yellow]\n"
            )

        sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
