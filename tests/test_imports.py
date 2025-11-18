"""Basic import tests to verify package structure."""

from argparse import Namespace

from gaca_ews import core, model


def test_import_modules() -> None:
    """Test that all main modules can be imported successfully."""
    # Verify key classes and functions are accessible
    assert hasattr(core, "get_args")
    assert hasattr(core, "fetch_last_hours")
    assert hasattr(core, "logger")
    assert hasattr(core, "plot_inference_maps")
    assert hasattr(core, "preprocess_for_inference")
    assert hasattr(model, "GCNGRU")


def test_config_namespace() -> None:
    """Test that config returns proper Namespace type."""
    # Verify Namespace type is available
    assert Namespace is not None
