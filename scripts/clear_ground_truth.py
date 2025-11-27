"""Clear ground truth data from BigQuery.

Usage:
    GCP_PROJECT_ID=coderd python scripts/clear_ground_truth.py
"""

import os

from google.cloud import bigquery
from rich.console import Console


console = Console()


def main() -> None:
    """Clear ground truth table in BigQuery."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        console.print("[red]✗ GCP_PROJECT_ID environment variable not set[/red]")
        return

    console.print("\n[bold cyan]Clearing Ground Truth from BigQuery[/bold cyan]\n")

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.gaca_evaluation.ground_truth"

    # Delete all rows
    console.print(f"[cyan]Deleting all rows from {table_id}...[/cyan]")
    query = f"DELETE FROM `{table_id}` WHERE TRUE"

    query_job = client.query(query)
    query_job.result()  # Wait for completion

    console.print("[bold green]✓ Ground truth table cleared![/bold green]\n")


if __name__ == "__main__":
    main()
