"""Tests for logger configuration and rich-based logging."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from gaca_ews.core import setup_logging
from gaca_ews.core.logger import (
    get_file_handler,
    get_logger,
    get_rich_handler,
    setup_console_logging,
    setup_file_logging,
)


def test_get_rich_handler() -> None:
    """Test creating a rich logging handler."""
    handler = get_rich_handler()
    assert isinstance(handler, RichHandler)
    assert handler.level == logging.INFO


def test_get_rich_handler_with_custom_console() -> None:
    """Test creating a rich handler with custom console."""
    console = Console()
    handler = get_rich_handler(console=console, level=logging.DEBUG, show_path=True)
    assert isinstance(handler, RichHandler)
    assert handler.level == logging.DEBUG


def test_get_file_handler(tmp_path: Path) -> None:
    """Test creating a file logging handler."""
    log_file = tmp_path / "test.log"
    handler = get_file_handler(log_file)

    assert isinstance(handler, logging.FileHandler)
    assert handler.level == logging.INFO
    assert Path(handler.baseFilename) == log_file


def test_get_logger() -> None:
    """Test getting the main logger instance."""
    logger = get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "forecast_logger"
    assert logger.level == logging.INFO
    assert not logger.propagate


def test_get_logger_singleton() -> None:
    """Test that get_logger returns the same instance."""
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_setup_console_logging() -> None:
    """Test setting up console logging."""
    logger = logging.getLogger("test_console_logger")
    setup_console_logging(logger)

    # Check that a RichHandler was added
    rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) > 0


def test_setup_file_logging(tmp_path: Path) -> None:
    """Test setting up file logging."""
    logger = logging.getLogger("test_file_logger")
    log_path = setup_file_logging(tmp_path, logger, filename="test_log.txt")

    assert log_path.exists()
    assert log_path.name == "test_log.txt"
    assert log_path.parent == tmp_path

    # Check that a FileHandler was added
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) > 0


def test_file_handler_auto_flush(tmp_path: Path) -> None:
    """Test that file handler automatically flushes."""
    log_file = tmp_path / "flush_test.log"
    logger = logging.getLogger("test_flush_logger")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Add file handler
    handler = get_file_handler(log_file)
    logger.addHandler(handler)

    # Write a log message
    test_message = "Test auto-flush message"
    logger.info(test_message)

    # Read the file immediately (should contain the message due to auto-flush)
    content = log_file.read_text()
    assert test_message in content


def test_logger_rich_markup() -> None:
    """Test that logger supports rich markup."""
    logger = get_logger()
    console = Console()

    # This should not raise an error
    setup_console_logging(logger, console=console)
    logger.info("[cyan]Test message[/cyan]")
    logger.info("[bold green]✓[/bold green] Success")


def test_setup_file_logging_creates_directory(tmp_path: Path) -> None:
    """Test that setup_file_logging creates directories if they don't exist."""
    nested_dir = tmp_path / "nested" / "logs"
    log_path = setup_file_logging(nested_dir)

    assert nested_dir.exists()
    assert log_path.exists()
    assert log_path.parent == nested_dir


def test_backward_compatibility_setup_logging(tmp_path: Path) -> None:
    """Test backward compatibility with old setup_logging function."""
    # Should work the same as setup_file_logging
    log_path = setup_logging(tmp_path)
    assert log_path.exists()
    assert log_path.name == "log.txt"
