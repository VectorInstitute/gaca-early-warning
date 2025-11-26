"""Load ground truth temperature data from NOAA cache to BigQuery.

This script extracts actual temperature observations from the cached NOAA data
and stores them in BigQuery for evaluation purposes.

Usage:
    GCP_PROJECT_ID=coderd python scripts/load_ground_truth.py
"""

from pathlib import Path

import polars as pl
from rich.console import Console

from gaca_ews.evaluation.storage import EvaluationStorage


console = Console()


def main() -> None:
    """Load ground truth from NOAA Parquet cache to BigQuery."""
    cache_file = Path("historical_predictions/noaa_data_cache.parquet")

    if not cache_file.exists():
        console.print("[red]✗ Cache file not found:[/red]", cache_file)
        console.print("Run batch predictions first to generate the cache.")
        return

    console.print("\n[bold cyan]Loading Ground Truth to BigQuery[/bold cyan]\n")
    console.print(f"Reading from: {cache_file}")

    # Read NOAA data using Polars (fast!)
    console.print("[cyan]Reading Parquet cache...[/cyan]")
    df_pl = pl.read_parquet(cache_file)

    console.print(f"[green]✓[/green] Loaded {len(df_pl):,} rows from cache")

    # Extract ground truth: actual temperature observations
    # The NOAA data contains t2m (2m temperature) which is the "actual" temperature
    # Convert from Kelvin to Celsius to match predictions
    ground_truth_pl = df_pl.select(
        [
            pl.col("time").alias("timestamp"),
            pl.col("lat"),
            pl.col("lon"),
            (pl.col("t2m") - 273.15).alias("actual_temp"),  # Convert K to °C
        ]
    ).unique()  # Remove duplicates if any

    console.print(
        f"[cyan]Extracted {len(ground_truth_pl):,} unique ground truth observations[/cyan]"
    )

    # Convert to pandas for BigQuery compatibility
    ground_truth_df = ground_truth_pl.to_pandas()

    # Store to BigQuery
    console.print("[cyan]Storing to BigQuery...[/cyan]")
    storage = EvaluationStorage()

    rows_loaded = storage.store_ground_truth(ground_truth_df)

    console.print("\n[bold green]✓ Complete![/bold green]")
    console.print(f"Loaded {rows_loaded:,} ground truth records to BigQuery")
    console.print("\nYou can now compute evaluation metrics!\n")


if __name__ == "__main__":
    main()
