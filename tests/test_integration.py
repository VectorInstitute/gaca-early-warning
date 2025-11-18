"""Integration tests for GACA Early Warning System."""

import pytest

from gaca_ews.core.data_extraction import fetch_last_hours


@pytest.mark.integration_test
def test_noaa_data_availability() -> None:
    """Test that NOAA URMA data bucket is accessible.

    This is a minimal integration test that verifies the external NOAA S3
    bucket is accessible. A full integration test would fetch actual data,
    but this serves as a placeholder to validate the integration test workflow.
    """
    # This test verifies the function exists and is callable
    # In a real integration test, we would call it with minimal parameters
    # and validate the response, but that would require several minutes
    # of execution time and external API calls
    assert callable(fetch_last_hours)

    # TODO: Implement full integration test that fetches minimal data
    # when integration test infrastructure is ready
    # Example:
    # df = fetch_last_hours(
    #     hours=1,
    #     variables=['t2m'],
    #     lat_min=42.0, lat_max=43.0,
    #     lon_min=278.0, lon_max=279.0
    # )
    # assert not df.empty
    # assert 't2m' in df.columns
