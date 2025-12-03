"""GACA Early Warning System - Command Line Interface.

A modern CLI for running temperature forecasts using the GCNGRU model.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from numpy.typing import NDArray
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typing_extensions import Annotated

from gaca_ews import __version__
from gaca_ews.core.batch_data import extract_window, fetch_historical_range
from gaca_ews.core.inference import InferenceEngine
from gaca_ews.core.logger import get_logger, setup_console_logging


# Initialize Typer app and Rich console
app = typer.Typer(
    name="gaca-ews",
    help="🌡️  GACA Early Warning System - Temperature Forecasting for Southwestern Ontario",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console(stderr=True)

# Configure logger to use the same console as CLI for clean output
setup_console_logging(console=console)


def _display_verbose_config(
    engine: InferenceEngine, config: Path, output: Path
) -> None:
    """Display configuration table in verbose mode."""
    info = engine.get_model_info()
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column(style="cyan")
    config_table.add_column()
    config_table.add_row("Config", str(config))
    config_table.add_row("Output", str(output))
    config_table.add_row("Device", info["device"])
    config_table.add_row("Nodes", f"{info['num_nodes']:,}")
    config_table.add_row("Model", info["model_architecture"])
    console.print(config_table)
    console.print()


def _run_pipeline_steps(
    engine: InferenceEngine,
    progress: Progress,
    output: Path,
    save_csv: bool,
    plots: bool,
    verbose: bool,
) -> tuple:
    """Run all pipeline steps with progress tracking.

    Returns
    -------
    tuple
        (predictions, latest_ts, csv_path)
    """
    # Fetch data
    fetch_task = progress.add_task(
        "[cyan]Fetching NOAA meteorological data...", total=None
    )
    data, latest_ts = engine.fetch_data()
    progress.remove_task(fetch_task)
    if verbose:
        console.print(
            f"  [dim]Fetched {len(data):,} rows • "
            f"Latest: {latest_ts.strftime('%Y-%m-%d %H:%M UTC')}[/dim]"
        )

    # Preprocess
    preprocess_task = progress.add_task("[cyan]Preprocessing features...", total=None)
    X = engine.preprocess(data)
    progress.remove_task(preprocess_task)
    if verbose:
        console.print(f"  [dim]Input shape: {X.shape}[/dim]")

    # Run inference
    inference_task = progress.add_task("[cyan]Running model inference...", total=None)
    predictions = engine.predict(X)
    progress.remove_task(inference_task)
    if verbose:
        console.print(f"  [dim]Output shape: {predictions.shape}[/dim]")

    # Save predictions
    csv_path = None
    if save_csv:
        save_task = progress.add_task("[cyan]Saving predictions to CSV...", total=None)
        csv_path = engine.save_predictions(
            predictions, latest_ts, output / "predictions.csv"
        )
        progress.remove_task(save_task)
        if verbose:
            console.print(f"  [dim]Saved: {csv_path}[/dim]")

    # Generate plots
    if plots:
        plot_task = progress.add_task(
            "[cyan]Generating visualization plots...", total=None
        )
        engine.generate_plots(predictions, latest_ts, output / "plots")
        progress.remove_task(plot_task)
        if verbose:
            console.print(f"  [dim]Plots saved to: {output / 'plots'}[/dim]")

    return predictions, latest_ts, csv_path


def _display_completion_summary(
    predictions: NDArray[Any], engine: InferenceEngine, output: Path
) -> None:
    """Display completion summary panel."""
    temps = predictions.flatten()
    num_predictions = predictions.size
    num_nodes = predictions.shape[2]
    horizons = engine.config["pred_offsets"]

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✓ Prediction Complete![/bold green]\n\n"
            f"📊 Generated [bold]{num_predictions:,}[/bold] predictions\n"
            f"📍 Covering [bold]{num_nodes:,}[/bold] locations\n"
            f"🕐 For [bold]{len(horizons)}[/bold] time horizons: {horizons}\n"
            f"🌡️  Temperature range: [bold]{temps.min():.1f}°C[/bold] to [bold]{temps.max():.1f}°C[/bold]\n"
            f"📂 Output: [cyan]{output}[/cyan]",
            border_style="green",
        )
    )
    console.print()


@app.command(name="predict")
def predict(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("config.yaml"),
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for predictions and plots",
        ),
    ] = None,
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots",
            help="Generate visualization plots",
        ),
    ] = True,
    save_csv: Annotated[
        bool,
        typer.Option(
            "--csv/--no-csv",
            help="Save predictions to CSV file",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed logging output",
        ),
    ] = False,
) -> None:
    """Run temperature prediction inference pipeline.

    Fetches meteorological data from NOAA, preprocesses features,
    runs the GCNGRU model, and generates multi-horizon predictions.

    Example:
        gaca-ews predict --config config.yaml --output results/
        gaca-ews predict -c config.yaml --no-plots
    """
    try:
        # Set logger level based on verbose mode
        log_level = logging.INFO if verbose else logging.WARNING
        logger = get_logger()
        logger.setLevel(log_level)

        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]GACA Early Warning System[/bold cyan]\n"
                "[dim]Temperature Forecasting Pipeline[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        # Initialize inference engine
        engine = InferenceEngine(config)

        # Set output directory
        if output is None:
            output = Path(engine.config.get("run_dir", "./predictions"))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Load artifacts
            load_task = progress.add_task(
                "[cyan]Loading model artifacts...", total=None
            )
            engine.load_artifacts()
            progress.remove_task(load_task)

            # Show config if verbose
            if verbose:
                _display_verbose_config(engine, config, output)

            # Run pipeline steps
            predictions, latest_ts, _ = _run_pipeline_steps(
                engine, progress, output, save_csv, plots, verbose
            )

        # Show summary
        _display_completion_summary(predictions, engine, output)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Operation cancelled by user[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command(name="batch-predict")
def batch_predict(  # noqa: PLR0912, PLR0915
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("config.yaml"),
    start_date: Annotated[
        str,
        typer.Option(
            "--start-date",
            help="Start date for batch predictions (YYYY-MM-DD HH:MM)",
        ),
    ] = "2024-02-06 12:00",
    end_date: Annotated[
        str,
        typer.Option(
            "--end-date",
            help="End date for batch predictions (YYYY-MM-DD HH:MM)",
        ),
    ] = "2024-07-19 17:00",
    interval_hours: Annotated[
        int,
        typer.Option(
            "--interval",
            help="Interval between predictions in hours",
        ),
    ] = 24,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for batch predictions",
        ),
    ] = None,
    save_csv: Annotated[
        bool,
        typer.Option(
            "--csv/--no-csv",
            help="Save each prediction to separate CSV files",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed logging output",
        ),
    ] = False,
) -> None:
    """Run batch predictions over a historical time period.

    Generates predictions for multiple timestamps within a date range,
    useful for validation, evaluation, and backtesting.

    Example:
        gaca-ews batch-predict --start-date "2024-02-06 12:00" \\
            --end-date "2024-02-10 12:00"
        gaca-ews batch-predict --start-date "2024-02-06 12:00" --interval 12
    """
    try:
        # Set logger level
        log_level = logging.INFO if verbose else logging.WARNING
        logger = get_logger()
        logger.setLevel(log_level)

        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
        except ValueError:
            console.print(
                "[bold red]✗ Error:[/bold red] Invalid date format. Use YYYY-MM-DD HH:MM"
            )
            raise typer.Exit(1) from None

        if start_dt >= end_dt:
            console.print(
                "[bold red]✗ Error:[/bold red] Start date must be before end date"
            )
            raise typer.Exit(1)

        # Generate timestamps
        timestamps = []
        current = start_dt
        while current <= end_dt:
            timestamps.append(current)
            current += timedelta(hours=interval_hours)

        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]GACA Batch Prediction[/bold cyan]\n"
                "[dim]Historical Temperature Forecasting[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        # Display batch info
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column(style="cyan")
        info_table.add_column()
        info_table.add_row("Start Date", start_dt.strftime("%Y-%m-%d %H:%M"))
        info_table.add_row("End Date", end_dt.strftime("%Y-%m-%d %H:%M"))
        info_table.add_row("Interval", f"{interval_hours} hours")
        info_table.add_row("Total Runs", str(len(timestamps)))
        console.print(info_table)
        console.print()

        # Initialize engine
        engine = InferenceEngine(config)

        # Set output directory
        if output is None:
            output = Path(engine.config.get("run_dir", "./batch_predictions"))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        # Load artifacts once
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task(
                "[cyan]Loading model artifacts...", total=None
            )
            engine.load_artifacts()
            progress.remove_task(load_task)

        # Fetch all historical data once (optimized approach)
        console.print("[bold]Step 1: Fetching historical NOAA data...[/bold]\n")

        # Add buffer to ensure we have enough data for the first window
        data_start = start_dt - timedelta(hours=engine.config["num_hours_to_fetch"])

        # IMPORTANT: Extend end date to include ground truth for all forecast horizons
        # Example: if last prediction run is at T and max horizon is 48h,
        # we need ground truth up to T+48h
        max_forecast_horizon = max(engine.config["pred_offsets"])
        data_end = end_dt + timedelta(hours=max_forecast_horizon)

        cache_file = output / "noaa_data_cache.parquet"

        try:
            full_data = fetch_historical_range(
                start_date=data_start,
                end_date=data_end,  # Extended to include forecast ground truth
                lat_min=engine.config["region"]["lat_min"],
                lat_max=engine.config["region"]["lat_max"],
                lon_min=engine.config["region"]["lon_min"],
                lon_max=engine.config["region"]["lon_max"],
                cache_file=cache_file,
            )
        except Exception as e:
            console.print(f"[red]✗ Failed to fetch historical data: {e}[/red]")
            raise typer.Exit(1) from e

        # Run batch predictions using cached data
        console.print("\n[bold]Step 2: Running batch predictions...[/bold]\n")

        successful = 0
        failed = 0
        failed_timestamps = []

        for i, ts in enumerate(timestamps, 1):
            console.print(
                f"[cyan]({i}/{len(timestamps)})[/cyan] {ts.strftime('%Y-%m-%d %H:%M')}...",
                end=" ",
            )

            try:
                # Extract time window from cached data
                data = extract_window(
                    full_data,
                    target_datetime=ts,
                    hours_back=engine.config["num_hours_to_fetch"],
                )

                if data is None or len(data) == 0:
                    console.print("[yellow]⚠ No data[/yellow]")
                    failed += 1
                    failed_timestamps.append((ts, "Insufficient data in window"))
                    continue

                # Preprocess and predict
                X = engine.preprocess(data)
                predictions = engine.predict(X)

                # Save if requested
                if save_csv:
                    csv_filename = f"predictions_{ts.strftime('%Y%m%d_%H%M')}.csv"
                    engine.save_predictions(predictions, ts, output / csv_filename)

                console.print("[green]✓[/green]")
                successful += 1

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠ Cancelled by user[/yellow]")
                raise typer.Exit(1) from None
            except Exception as e:
                console.print(f"[red]✗ {str(e)[:50]}[/red]")
                failed += 1
                failed_timestamps.append((ts, str(e)))
                if verbose:
                    logger.exception(f"Failed for {ts}")

        # Display summary
        console.print()
        summary = (
            f"[bold green]✓ Batch Complete![/bold green]\n\n"
            f"[green]Successful:[/green] {successful}/{len(timestamps)}\n"
            f"[red]Failed:[/red] {failed}/{len(timestamps)}\n"
            f"📂 Output: [cyan]{output}[/cyan]"
        )

        if failed_timestamps and verbose:
            summary += "\n\n[bold]Failed Timestamps:[/bold]"
            for ts, error in failed_timestamps[:5]:  # Show first 5
                summary += f"\n  • {ts.strftime('%Y-%m-%d %H:%M')}: {error[:50]}"
            if len(failed_timestamps) > 5:
                summary += f"\n  ... and {len(failed_timestamps) - 5} more"

        console.print(
            Panel.fit(summary, border_style="green" if failed == 0 else "yellow")
        )
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Operation cancelled by user[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command(name="info")
def info(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("config.yaml"),
) -> None:
    """Display model configuration and system information.

    Shows details about the loaded model, region coverage,
    forecast horizons, and available features.

    Example:
        gaca-ews info
        gaca-ews info --config custom-config.yaml
    """
    try:
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]GACA-EWS model information[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()

        # Load engine and get info
        engine = InferenceEngine(config)
        info_data = engine.get_model_info()

        # Display configuration
        table = Table(title="Model Configuration", show_header=False, box=None)
        table.add_column(style="cyan", width=20)
        table.add_column()

        table.add_row("Architecture", info_data["model_architecture"])
        table.add_row("Device", info_data["device"])
        table.add_row("Number of Nodes", f"{info_data['num_nodes']:,}")
        table.add_row("Input Features", ", ".join(info_data["input_features"]))
        table.add_row("Forecast Horizons", str(info_data["prediction_horizons"]))

        console.print(table)
        console.print()

        # Display region
        region_table = Table(title="Coverage Region", show_header=False, box=None)
        region_table.add_column(style="cyan", width=20)
        region_table.add_column()

        region = info_data["region"]
        region_table.add_row(
            "Latitude", f"{region['lat_min']}° to {region['lat_max']}°"
        )
        region_table.add_row(
            "Longitude", f"{region['lon_min']}° to {region['lon_max']}°"
        )
        region_table.add_row("Region", "Southwestern Ontario")

        console.print(region_table)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command(name="version")
def version() -> None:
    """Show version information."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]GACA Early Warning System[/bold cyan]\n"
            f"Version: [bold]{__version__}[/bold]\n"
            f"Model: GCNGRU (Graph Convolutional GRU)",
            border_style="cyan",
        )
    )
    console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """GACA Early Warning System - Temperature Forecasting CLI.

    A GNN-based forecasting system for Southwestern Ontario using NOAA data.
    """
    if version_flag:
        version()
        raise typer.Exit

    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
