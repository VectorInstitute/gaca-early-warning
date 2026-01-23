#!/usr/bin/env python3
"""Backfill missing ground truth data for dynamic evaluation.

This script fetches actual NOAA temperatures for forecast times that
have predictions but no ground truth, enabling evaluation metrics to
be computed.

Usage:
    export GCP_PROJECT_ID=your-project-id
    python scripts/backfill_ground_truth.py --days 45
"""

import argparse
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaca_ews.core.data_extraction import fetch_historical_hours
from gaca_ews.evaluation.storage import EvaluationStorage


console = Console()

# Region boundaries (must match trained model)
LAT_MIN = 42.0
LAT_MAX = 45.0
LON_MIN = -81.0
LON_MAX = -78.0


def get_missing_timestamps(storage: EvaluationStorage, days: int) -> list[datetime]:
    """Get forecast times that need ground truth data.

    Parameters
    ----------
    storage : EvaluationStorage
        BigQuery storage instance
    days : int
        Number of days to look back

    Returns
    -------
    list[datetime]
        List of timestamps needing ground truth
    """
    # Use hours instead of days for the lookback
    lookback_hours = days * 24
    return storage.get_forecast_times_needing_ground_truth(lookback_hours)


def fetch_and_store_ground_truth(
    timestamp: datetime,
    storage: EvaluationStorage,
) -> int:
    """Fetch NOAA data for a timestamp and store as ground truth.

    Parameters
    ----------
    timestamp : datetime
        The timestamp to fetch ground truth for
    storage : EvaluationStorage
        BigQuery storage instance

    Returns
    -------
    int
        Number of rows stored, or 0 if failed
    """
    try:
        # Fetch NOAA data for this specific hour
        data = fetch_historical_hours(
            target_datetime=timestamp,
            hours_back=1,
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
        )

        if data is None or data.empty:
            return 0

        # Extract ground truth (t2m in Celsius)
        # Column is 'datetime' from fetch_historical_hours
        ground_truth_df = data[["datetime", "lat", "lon", "t2m"]].copy()
        ground_truth_df = ground_truth_df.rename(
            columns={"datetime": "timestamp", "t2m": "actual_temp"}
        )

        # Convert from Kelvin to Celsius
        ground_truth_df["actual_temp"] = ground_truth_df["actual_temp"] - 273.15

        # Store to BigQuery
        return storage.store_ground_truth(ground_truth_df)

    except Exception as e:
        console.print(f"[red]Failed to fetch {timestamp}: {e}[/red]")
        return 0


def main() -> None:
    """Backfill missing ground truth data."""
    parser = argparse.ArgumentParser(
        description="Backfill missing ground truth for dynamic evaluation"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=45,
        help="Number of days to look back for missing ground truth (default: 45)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Number of timestamps to process before showing progress (default: 24)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't actually fetch/store",
    )

    args = parser.parse_args()

    console.print("\n[bold cyan]Ground Truth Backfill[/bold cyan]\n")
    console.print(f"Looking back: {args.days} days")

    # Check for GCP project
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        console.print("[red]Error: GCP_PROJECT_ID environment variable not set[/red]")
        console.print("Run: export GCP_PROJECT_ID=your-project-id")
        sys.exit(1)

    console.print(f"GCP Project: {project_id}")

    # Initialize storage (uses GCP_PROJECT_ID from environment)
    try:
        storage = EvaluationStorage()
        console.print(
            f"[green]✓[/green] BigQuery connection established (project: {storage.project_id})"
        )
    except Exception as e:
        console.print(f"[red]Failed to connect to BigQuery: {e}[/red]")
        sys.exit(1)

    # Get missing timestamps
    console.print("\n[cyan]Checking for missing ground truth...[/cyan]")
    missing_timestamps = get_missing_timestamps(storage, args.days)

    if not missing_timestamps:
        console.print("[green]✓ No missing ground truth found![/green]")
        return

    console.print(
        f"[yellow]Found {len(missing_timestamps)} timestamps needing ground truth[/yellow]"
    )

    # Show date range
    if missing_timestamps:
        earliest = min(missing_timestamps)
        latest = max(missing_timestamps)
        console.print(f"  Range: {earliest.isoformat()} to {latest.isoformat()}")

    if args.dry_run:
        console.print("\n[yellow]Dry run - no data will be fetched or stored[/yellow]")
        return

    # Process timestamps with progress bar
    console.print("\n[cyan]Fetching and storing ground truth...[/cyan]")

    total_rows = 0
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing 0/{len(missing_timestamps)}...", total=len(missing_timestamps)
        )

        for i, ts in enumerate(missing_timestamps):
            progress.update(
                task,
                description=f"Processing {i + 1}/{len(missing_timestamps)}: {ts.isoformat()}",
            )

            rows = fetch_and_store_ground_truth(ts, storage)

            if rows > 0:
                total_rows += rows
                successful += 1
            else:
                failed += 1

            progress.advance(task)

            # Periodic status update
            if (i + 1) % args.batch_size == 0:
                console.print(
                    f"  Progress: {i + 1}/{len(missing_timestamps)} "
                    f"({successful} OK, {failed} failed, {total_rows:,} rows)"
                )

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Timestamps processed: {successful + failed}")
    console.print(f"  Successful: {successful}")
    console.print(f"  Failed: {failed}")
    console.print(f"  Total rows stored: {total_rows:,}")

    if successful > 0:
        console.print("\n[green]✓ Ground truth backfill complete![/green]")
        console.print("The dynamic evaluation should now show metrics.")
    else:
        console.print(
            "\n[yellow]⚠ No ground truth was stored. Check NOAA data availability.[/yellow]"
        )


if __name__ == "__main__":
    main()
