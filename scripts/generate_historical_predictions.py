"""Generate predictions for historical validation period and store to Firestore.

This script wraps the batch-predict CLI command and stores results to Firestore
for evaluation purposes.

Usage:
    python scripts/generate_historical_predictions.py --help
    python scripts/generate_historical_predictions.py --start-date "2024-02-06 12:00"
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
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


def main() -> None:
    """Generate historical predictions and store to BigQuery."""
    parser = argparse.ArgumentParser(
        description="Generate historical predictions and store to Firestore"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-02-06 12:00",
        help="Start date (YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-07-19 17:00",
        help="End date (YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Interval between predictions in hours",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./historical_predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip batch prediction (only store existing CSVs)",
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

    # Step 1: Run batch predictions (unless skipped)
    if not args.skip_inference:
        console.print("\n[bold cyan]Step 1: Running batch predictions[/bold cyan]\n")

        success = run_batch_predictions(
            start_date=args.start_date,
            end_date=args.end_date,
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

    # Step 2: Store to BigQuery
    console.print("\n[bold cyan]Step 2: Storing predictions to BigQuery[/bold cyan]\n")

    try:
        storage = EvaluationStorage()
        successful, failed = store_predictions_to_bigquery(
            output_dir, storage, args.verbose
        )

        # Summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Complete![/bold green]")
        console.print(f"  Stored to BigQuery: {successful}")
        console.print(f"  Failed: {failed}")
        console.print("=" * 60 + "\n")

        sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
