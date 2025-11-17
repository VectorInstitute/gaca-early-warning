"""NOAA URMA data extraction utilities.

This module provides functions for extracting meteorological data from NOAA's
Unrestricted Mesoscale Analysis (URMA) datasets stored in AWS S3. The data includes
surface weather variables such as temperature, dewpoint, wind components, surface
pressure, and orography.

Functions
---------
parse_timestamp_from_key
    Extract timestamp from NOAA S3 object key.
get_most_recent_noaa_ts
    Identify the most recent available timestamp in the NOAA bucket.
list_recent_files
    List GRIB2 files covering the last N hours.
extract_grib_file
    Download and parse a single GRIB2 file, returning a DataFrame.
fetch_last_hours
    Main wrapper function to fetch and concatenate recent hours of data.

Notes
-----
The module uses boto3 for S3 access and pygrib for parsing GRIB2 files.
Data is automatically cropped to a specified geographic region and returned as
pandas DataFrames with hourly temporal resolution.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to extract data for inference using noaa api
###########################################################################################

import re
import tempfile
from datetime import datetime, timedelta
from typing import Any

import boto3
import botocore
import numpy as np
import pandas as pd
import pygrib
from numpy.typing import NDArray

from util.logger import logger


def parse_timestamp_from_key(key: str) -> datetime | None:
    """Parse and return timestamp from NOAA S3 object key."""
    f = re.search(r"\.t(\d{2})z", key)
    if f:
        hour = int(f.group(1))  # get hour

        # now get date from prefix: urma2p5.YYYYMMDD/
        f2 = re.search(r"urma2p5\.(\d{8})", key)
        if f2:
            date = f2.group(1)  # YYYYMMDD
            try:
                return datetime.strptime(date + f"{hour:02d}", "%Y%m%d%H")
            except Exception:
                logger.info("invalid timestamp!")
    return None


def get_most_recent_noaa_ts(
    s3_client: Any, bucket: str, hours_back: int
) -> tuple[datetime, list[datetime]]:
    """Get the most recent timestamp in NOAA bucket and generate sequence."""
    today = datetime.utcnow().strftime("%Y%m%d")
    prefix = f"urma2p5.{today}/"

    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in resp:
        raise RuntimeError("no urma files found for today :/")

    timestamps = []

    for obj in resp["Contents"]:
        key = obj["Key"]
        if key.endswith(".grb2_wexp") and "2dvaranl" in key:
            ts = parse_timestamp_from_key(key)
            if ts:
                timestamps.append(ts)

    if not timestamps:
        raise RuntimeError("no valid 2dvar hourly timestamps found today :(")

    latest = max(timestamps)

    logger.info(f"🌟 latest available urma timestamp: {latest}")

    # now get all timestamps starting from latest + working way backwards
    timestamps = [latest - timedelta(hours=i) for i in range(hours_back)]

    return (
        latest,
        timestamps,
    )  # return latest also bc we would need an indicator of when the predictions are relative to


def list_recent_files(
    s3_client: Any, bucket: str, hours_back: int = 24
) -> tuple[list[str], datetime]:
    """Return list of GRIB2 file keys covering the specified number of recent hours."""
    latest, timestamps = get_most_recent_noaa_ts(s3_client, bucket, hours_back)

    # logger.info(f"timestamps looking at: {timestamps}\n")

    keys = []
    for ts in timestamps:
        date_str = ts.strftime("%Y%m%d")
        hour_str = ts.strftime("%H")

        prefix = f"urma2p5.{date_str}/"
        # logger.info(f"prefix for extraction: {prefix}, date_str: {date_str},
        # hr str: {hour_str}")
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in resp:
            continue

        for obj in resp["Contents"]:
            key = obj["Key"]
            if (
                key.endswith(".grb2_wexp")
                and f".t{hour_str}z" in key
                and "2dvaranl" in key
            ):
                # logger.info(f"\tappending {key}")
                keys.append(key)

    keys = sorted(set(keys))
    return keys, latest


def extract_grib_file(
    key: str,
    s3_client: Any,
    bucket: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame | None:
    """Download and parse GRIB2 file, returning cropped DataFrame."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        by = obj["Body"].read()

        with tempfile.NamedTemporaryFile(delete=True, suffix=".grb2") as f:
            f.write(by)
            f.flush()

            grbs = pygrib.open(f.name)
            base_shape = grbs[1].values.shape

            def safe(name: str) -> NDArray[Any]:
                try:
                    return grbs.select(name=name)[0].values
                except Exception:
                    return np.full(base_shape, np.nan)

            # variables needed
            t2m = safe("2 metre temperature")
            d2m = safe("2 metre dewpoint temperature")
            u10 = safe("10 metre U wind component")
            v10 = safe("10 metre V wind component")
            sp = safe("Surface pressure")
            orog = safe("Orography")

            lats, lons = grbs[1].latlons()
            grbs.close()

        date_match = re.search(r"(\d{8})", key)
        hour_match = re.search(r"\.t(\d{2})z", key)
        if not date_match or not hour_match:
            raise ValueError(f"Could not parse date/hour from key: {key}")
        date = date_match.group(1)
        hour = hour_match.group(1)
        dt = datetime.strptime(date + hour, "%Y%m%d%H")

        df = pd.DataFrame(
            {
                "lat": lats.flatten(),
                "lon": lons.flatten(),
                "t2m": t2m.flatten(),
                "d2m": d2m.flatten(),
                "u10": u10.flatten(),
                "v10": v10.flatten(),
                "sp": sp.flatten(),
                "orog": orog.flatten(),
                "datetime": dt,
            }
        )

        region = df[
            (df.lat >= lat_min)
            & (df.lat <= lat_max)
            & (df.lon >= lon_min)
            & (df.lon <= lon_max)
        ]

        return region.reset_index(drop=True)

    except Exception as e:
        logger.info("⚠️ failed:", key, e)
        return None


def fetch_last_hours(cfg: Any) -> tuple[pd.DataFrame, datetime]:
    """Fetch and concatenate meteorological data for the last N hours as configured."""
    bucket = "noaa-urma-pds"
    hours = cfg.num_hours_to_fetch

    lat_min = cfg.region.lat_min
    lat_max = cfg.region.lat_max
    lon_min = cfg.region.lon_min
    lon_max = cfg.region.lon_max

    logger.info(f"in main extraction func, hours: {hours}, lat min: {lat_min}")

    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=boto3.session.Config(signature_version=botocore.UNSIGNED),
    )

    keys, latest = list_recent_files(s3, bucket, hours_back=hours)
    # logger.info(f"returned keys: {keys}, latest timestamp available: {latest}")
    frames = []

    for k in keys:
        df = extract_grib_file(k, s3, bucket, lat_min, lat_max, lon_min, lon_max)
        if df is not None:
            frames.append(df)

    if len(frames) == 0:
        raise RuntimeError("no urma files found for recent hours! :(")

    return pd.concat(frames).reset_index(drop=True), latest
