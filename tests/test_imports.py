"""Basic import tests to verify package structure."""

from argparse import Namespace

from model import gcngru
from util import config, data_extraction, logger, plotting, preprocessing


def test_import_modules() -> None:
    """Test that all main modules can be imported successfully."""
    # Verify key classes and functions are accessible
    assert hasattr(config, "get_args")
    assert hasattr(data_extraction, "fetch_last_hours")
    assert hasattr(logger, "logger")
    assert hasattr(plotting, "plot_inference_maps")
    assert hasattr(preprocessing, "preprocess_for_inference")
    assert hasattr(gcngru, "GCNGRU")


def test_config_namespace() -> None:
    """Test that config returns proper Namespace type."""
    # Verify Namespace type is available
    assert Namespace is not None
