"""Clear static evaluation predictions from BigQuery.

Deletes all predictions for the static evaluation period (Feb 6, 2024 - July 19, 2024)
to allow for regeneration with proper hourly intervals.

Usage:
    GCP_PROJECT_ID=coderd python scripts/clear_static_predictions.py
"""

import os
from datetime import datetime

from google.cloud import bigquery
from rich.console import Console


console = Console()


def main() -> None:
    """Clear static evaluation predictions from BigQuery."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        console.print("[red]✗ GCP_PROJECT_ID environment variable not set[/red]")
        return

    console.print(
        "\n[bold cyan]Clearing Static Evaluation Predictions from BigQuery[/bold cyan]\n"
    )

    # Static evaluation period
    start_date = datetime(2024, 2, 6, 12, 0, 0)
    end_date = datetime(2024, 7, 19, 17, 0, 0)

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.gaca_evaluation.predictions"

    console.print("[yellow]Target period:[/yellow]")
    console.print(f"  Start: {start_date}")
    console.print(f"  End: {end_date}")
    console.print()

    # Count rows to be deleted
    count_query = f"""
    SELECT COUNT(*) as count
    FROM `{table_id}`
    WHERE run_timestamp BETWEEN @start_date AND @end_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
        ]
    )

    console.print("[cyan]Counting rows in static period...[/cyan]")
    count_job = client.query(count_query, job_config=job_config)
    result = list(count_job.result())
    row_count = result[0]["count"] if result else 0

    console.print(f"[yellow]Found {row_count:,} rows to delete[/yellow]\n")

    if row_count == 0:
        console.print("[green]✓ No rows to delete[/green]\n")
        return

    # Confirm deletion
    console.print(
        "[bold red]WARNING:[/bold red] This will permanently delete "
        f"{row_count:,} prediction rows!"
    )
    response = input("Continue? (yes/no): ").strip().lower()

    if response != "yes":
        console.print("\n[yellow]⚠ Cancelled[/yellow]\n")
        return

    # Delete rows
    console.print(f"\n[cyan]Deleting rows from {table_id}...[/cyan]")
    delete_query = f"""
    DELETE FROM `{table_id}`
    WHERE run_timestamp BETWEEN @start_date AND @end_date
    """

    delete_job = client.query(delete_query, job_config=job_config)
    delete_job.result()  # Wait for completion

    console.print(
        f"\n[bold green]✓ Deleted {row_count:,} predictions from static period![/bold green]\n"
    )


if __name__ == "__main__":
    main()
