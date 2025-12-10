#!/usr/bin/env python
"""Clean dynamic evaluation data before repopulating.

This script deletes predictions and ground truth data from the rolling 30-day
window to prepare for fresh data population. Use this when switching from
biased (24h interval) to unbiased (1h interval) predictions.

Usage:
    # Interactive mode (asks for confirmation)
    GCP_PROJECT_ID=your-project python scripts/clean_dynamic_eval.py

    # Force mode (no confirmation)
    python scripts/clean_dynamic_eval.py --force

    # Custom day range
    python scripts/clean_dynamic_eval.py --days 30 --force
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

from google.cloud import bigquery
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table


console = Console()


def get_data_counts(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    start_date: datetime,
    end_date: datetime,
) -> tuple[int, int]:
    """Get counts of predictions and ground truth to be deleted.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client
    project_id : str
        GCP project ID
    dataset_id : str
        BigQuery dataset ID
    start_date : datetime
        Start of deletion range
    end_date : datetime
        End of deletion range

    Returns
    -------
    tuple[int, int]
        (prediction_count, ground_truth_count)
    """
    # Count predictions
    pred_query = f"""
    SELECT COUNT(*) as count
    FROM `{project_id}.{dataset_id}.predictions`
    WHERE run_timestamp >= @start_date
      AND run_timestamp <= @end_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
        ]
    )

    pred_job = client.query(pred_query, job_config=job_config)
    pred_results = list(pred_job.result())
    pred_count = pred_results[0]["count"] if pred_results else 0

    # Count ground truth (extended by 48h for forecast horizon)
    gt_end = end_date + timedelta(hours=48)
    gt_query = f"""
    SELECT COUNT(*) as count
    FROM `{project_id}.{dataset_id}.ground_truth`
    WHERE timestamp >= @start_date
      AND timestamp <= @end_date
    """

    job_config_gt = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", gt_end),
        ]
    )

    gt_job = client.query(gt_query, job_config=job_config_gt)
    gt_results = list(gt_job.result())
    gt_count = gt_results[0]["count"] if gt_results else 0

    return pred_count, gt_count


def delete_predictions(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    start_date: datetime,
    end_date: datetime,
) -> int:
    """Delete predictions in the specified date range.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client
    project_id : str
        GCP project ID
    dataset_id : str
        BigQuery dataset ID
    start_date : datetime
        Start of deletion range
    end_date : datetime
        End of deletion range

    Returns
    -------
    int
        Number of rows deleted
    """
    query = f"""
    DELETE FROM `{project_id}.{dataset_id}.predictions`
    WHERE run_timestamp >= @start_date
      AND run_timestamp <= @end_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
        ]
    )

    console.print("[cyan]Deleting predictions...[/cyan]")
    job = client.query(query, job_config=job_config)
    job.result()

    console.print("[green]✓[/green] Deleted predictions")
    return job.num_dml_affected_rows or 0


def delete_ground_truth(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    start_date: datetime,
    end_date: datetime,
) -> int:
    """Delete ground truth in the specified date range.

    Extends end_date by 48h to include ground truth for forecast horizons.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client
    project_id : str
        GCP project ID
    dataset_id : str
        BigQuery dataset ID
    start_date : datetime
        Start of deletion range
    end_date : datetime
        End of deletion range

    Returns
    -------
    int
        Number of rows deleted
    """
    # Extend by 48h to cover forecast ground truth
    gt_end = end_date + timedelta(hours=48)

    query = f"""
    DELETE FROM `{project_id}.{dataset_id}.ground_truth`
    WHERE timestamp >= @start_date
      AND timestamp <= @end_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", gt_end),
        ]
    )

    console.print("[cyan]Deleting ground truth...[/cyan]")
    job = client.query(query, job_config=job_config)
    job.result()

    console.print("[green]✓[/green] Deleted ground truth")
    return job.num_dml_affected_rows or 0


def main() -> None:
    """Clean dynamic evaluation data."""
    parser = argparse.ArgumentParser(
        description="Clean dynamic evaluation data before repopulating"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to clean (default: 30)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Delete only predictions, keep ground truth",
    )

    args = parser.parse_args()

    # Get project ID
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        console.print("[red]✗ GCP_PROJECT_ID environment variable not set[/red]")
        console.print("Set it with: export GCP_PROJECT_ID=your-project-id")
        sys.exit(1)

    dataset_id = os.getenv("BIGQUERY_DATASET", "gaca_evaluation")

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    console.print()
    console.print(
        Panel.fit(
            "[bold red]Clean Dynamic Evaluation Data[/bold red]\n"
            "[dim]This will permanently delete data from BigQuery[/dim]",
            border_style="red",
        )
    )
    console.print()

    # Display deletion info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column()
    info_table.add_row("Project", project_id)
    info_table.add_row("Dataset", dataset_id)
    info_table.add_row("Start Date", start_date.strftime("%Y-%m-%d %H:%M UTC"))
    info_table.add_row("End Date", end_date.strftime("%Y-%m-%d %H:%M UTC"))
    info_table.add_row("Days", str(args.days))
    console.print(info_table)
    console.print()

    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)

        # Get counts before deletion
        console.print("[cyan]Checking existing data...[/cyan]")
        pred_count, gt_count = get_data_counts(
            client, project_id, dataset_id, start_date, end_date
        )

        if pred_count == 0 and gt_count == 0:
            console.print("[yellow]⚠ No data found in the specified range[/yellow]")
            console.print(
                f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
            sys.exit(0)

        # Display what will be deleted
        console.print()
        delete_table = Table(title="Data to be Deleted", box=None)
        delete_table.add_column("Table", style="cyan")
        delete_table.add_column("Rows", justify="right", style="yellow")
        delete_table.add_row("Predictions", f"{pred_count:,}")
        if not args.predictions_only:
            delete_table.add_row(
                "Ground Truth", f"{gt_count:,} (includes +48h forecast data)"
            )
        console.print(delete_table)
        console.print()

        # Confirmation
        if not args.force:
            confirmed = Confirm.ask(
                "[bold red]⚠ This will permanently delete the data above. Continue?[/bold red]"
            )
            if not confirmed:
                console.print("[yellow]⊘ Cancelled by user[/yellow]")
                sys.exit(0)

        console.print()
        console.print("=" * 70)
        console.print("[bold]Deleting data...[/bold]")
        console.print("=" * 70)
        console.print()

        # Delete predictions
        deleted_pred = delete_predictions(
            client, project_id, dataset_id, start_date, end_date
        )

        # Delete ground truth (unless predictions-only)
        deleted_gt = 0
        if not args.predictions_only:
            deleted_gt = delete_ground_truth(
                client, project_id, dataset_id, start_date, end_date
            )

        # Summary
        console.print()
        console.print("=" * 70)
        console.print("[bold green]✓ Deletion Complete![/bold green]")
        console.print(f"  Predictions deleted: {deleted_pred:,}")
        if not args.predictions_only:
            console.print(f"  Ground truth deleted: {deleted_gt:,}")
        console.print("=" * 70)
        console.print()

        console.print("[green]✓ Ready to repopulate with unbiased data![/green]\n")
        console.print("[dim]Next steps:[/dim]")
        console.print(
            f"  python scripts/populate_dynamic_evaluation.py --days {args.days} --interval 1 --verbose\n"
        )

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
