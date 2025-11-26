"""Optimized batch data fetching for historical predictions using Polars + Parquet.

This module provides high-performance bulk fetching of historical NOAA data,
using Polars for fast in-memory operations and Parquet for efficient caching.

Performance improvements over pandas + CSV:
- Parquet I/O: 10-50x faster than CSV
- Polars operations: 10-100x faster than pandas for large datasets
- Multi-threaded by default
- Lazy evaluation for query optimization
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import botocore
import pandas as pd
import polars as pl
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from gaca_ews.core.data_extraction import extract_grib_file
from gaca_ews.core.logger import logger


console = Console()


def _fetch_single_file(
    ts: datetime,
    bucket: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame | None:
    """Fetch a single GRIB file for a specific timestamp.

    Parameters
    ----------
    ts : datetime
        Target timestamp
    bucket : str
        S3 bucket name
    lat_min, lat_max, lon_min, lon_max : float
        Region boundaries

    Returns
    -------
    pd.DataFrame | None
        Data with time column added, or None if failed
    """
    # Create a new S3 client for this thread
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=boto3.session.Config(signature_version=botocore.UNSIGNED),
    )

    date_str = ts.strftime("%Y%m%d")
    hour_str = ts.strftime("%H")
    prefix = f"urma2p5.{date_str}/"

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" not in resp:
            return None

        # Find the specific file for this hour
        for obj in resp["Contents"]:
            key = obj["Key"]
            if (
                key.endswith(".grb2_wexp")
                and f".t{hour_str}z" in key
                and "2dvaranl" in key
            ):
                df = extract_grib_file(
                    key, s3, bucket, lat_min, lat_max, lon_min, lon_max
                )
                if df is not None:
                    # Add timestamp column
                    df["time"] = ts
                    return df
                break

        return None

    except Exception as e:
        logger.debug(f"Failed to fetch {ts}: {e}")
        return None


def fetch_historical_range(
    start_date: datetime,
    end_date: datetime,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_file: Path | None = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Fetch NOAA URMA data using parallel downloads with Parquet caching.

    Optimized with Polars for fast in-memory operations and Parquet for
    efficient storage.

    Parameters
    ----------
    start_date : datetime
        Start of the period
    end_date : datetime
        End of the period
    lat_min, lat_max, lon_min, lon_max : float
        Region boundaries
    cache_file : Path | None
        Optional path to cache the data (uses Parquet format)
    max_workers : int
        Number of parallel download threads (default: 10)

    Returns
    -------
    pd.DataFrame
        Historical data with columns: time, lat, lon, t2m, d2m, u10, v10, sp, orog
    """
    # Check if cached file exists
    if cache_file and cache_file.exists():
        console.print(f"[cyan]Loading cached data from {cache_file}...[/cyan]")

        # Use Polars for fast Parquet reading
        df_pl = pl.read_parquet(cache_file)

        # Convert to pandas for compatibility with existing pipeline
        df = df_pl.to_pandas()

        console.print(f"[green]✓[/green] Loaded {len(df):,} rows from cache (Parquet)")
        return df

    console.print(f"[cyan]Fetching historical data: {start_date} to {end_date}[/cyan]")

    bucket = "noaa-urma-pds"

    # Generate all hourly timestamps in the range
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(hours=1)

    console.print(
        f"Fetching {len(timestamps)} hourly files with {max_workers} parallel workers..."
    )

    frames = []
    failed_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Downloading NOAA data...",
            total=len(timestamps),
        )

        # Use ThreadPoolExecutor for parallel I/O
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_ts = {
                executor.submit(
                    _fetch_single_file,
                    ts,
                    bucket,
                    lat_min,
                    lat_max,
                    lon_min,
                    lon_max,
                ): ts
                for ts in timestamps
            }

            # Process results as they complete
            for future in as_completed(future_to_ts):
                ts = future_to_ts[future]
                try:
                    result_df: pd.DataFrame | None = future.result()
                    if result_df is not None:
                        frames.append(result_df)
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.debug(f"Exception fetching {ts}: {e}")
                    failed_count += 1

                progress.update(task, advance=1)

    if len(frames) == 0:
        raise RuntimeError("No NOAA data could be fetched for the specified period")

    # Convert to Polars for fast concatenation and operations
    console.print(f"[cyan]Concatenating {len(frames)} dataframes with Polars...[/cyan]")

    # Convert pandas frames to Polars
    pl_frames = [pl.from_pandas(df) for df in frames]

    # Concatenate using Polars (much faster than pandas)
    full_data_pl = pl.concat(pl_frames)

    # Sort by time
    full_data_pl = full_data_pl.sort("time")

    console.print(
        f"[green]✓[/green] Fetched {len(full_data_pl):,} rows "
        f"({len(frames)}/{len(timestamps)} hours, {failed_count} failed)"
    )

    # Cache to Parquet if requested (much faster than CSV)
    if cache_file:
        console.print(f"[cyan]Caching data to {cache_file} (Parquet format)...[/cyan]")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Write Parquet with compression
        full_data_pl.write_parquet(
            cache_file,
            compression="zstd",  # Fast compression with good ratio
            statistics=True,  # Enable column statistics for faster queries
        )

        file_size_mb = cache_file.stat().st_size / (1024 * 1024)
        console.print(
            f"[green]✓[/green] Data cached to Parquet ({file_size_mb:.1f} MB compressed)"
        )

    # Convert back to pandas for compatibility
    return full_data_pl.to_pandas()


def extract_window(
    full_data: pd.DataFrame,
    target_datetime: datetime,
    hours_back: int,
) -> pd.DataFrame | None:
    """Extract a time window from bulk historical data using fast Polars operations.

    Parameters
    ----------
    full_data : pd.DataFrame
        Full historical dataset with 'time' column
    target_datetime : datetime
        Target datetime (end of window)
    hours_back : int
        Number of hours to include before target

    Returns
    -------
    pd.DataFrame | None
        Windowed data or None if insufficient data
    """
    # Convert to Polars for fast filtering
    df_pl = pl.from_pandas(full_data)

    # Define window boundaries
    start_time = target_datetime - timedelta(hours=hours_back - 1)
    end_time = target_datetime

    # Filter using Polars (much faster than pandas)
    window_pl = df_pl.filter(
        (pl.col("time") >= start_time) & (pl.col("time") <= end_time)
    )

    # Check if we have all required hours
    expected_hours = {start_time + timedelta(hours=i) for i in range(hours_back)}

    # Get unique timestamps using Polars
    actual_hours = set(
        window_pl.select(pl.col("time").dt.truncate("1h"))
        .unique()
        .to_series()
        .to_list()
    )

    if len(actual_hours) < len(expected_hours):
        logger.warning(
            f"Incomplete window for {target_datetime}: "
            f"expected {len(expected_hours)} hours, got {len(actual_hours)}"
        )
        return None

    # Convert back to pandas for compatibility with existing pipeline
    return window_pl.to_pandas()


def get_prediction_timestamps(
    start_date: datetime,
    end_date: datetime,
    interval_hours: int,
) -> list[datetime]:
    """Generate list of prediction timestamps.

    Parameters
    ----------
    start_date : datetime
        Start of prediction period
    end_date : datetime
        End of prediction period
    interval_hours : int
        Interval between predictions

    Returns
    -------
    list[datetime]
        List of timestamps for predictions
    """
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(hours=interval_hours)
    return timestamps
