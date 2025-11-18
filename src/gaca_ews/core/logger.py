"""Logging configuration and setup utilities.

This module configures rich-based logging for the GACA early warning pipeline.
It provides reusable utilities for setting up console and file-based logging
with styled output using the rich library for consistency with the CLI.

Functions
---------
get_logger
    Get or create the main forecast logger instance.
setup_file_logging
    Configure file-based logging in the specified run directory.
setup_console_logging
    Configure rich-based console logging with styled output.
get_rich_handler
    Create a rich logging handler with consistent styling.
get_file_handler
    Create a file logging handler with auto-flushing.

Attributes
----------
logger : logging.Logger
    Main logger instance used throughout the pipeline.

Notes
-----
The logger uses rich's RichHandler for console output with styled formatting
including:
- Colored log levels
- Timestamps
- Log paths with dimmed styling
- Consistent markup with the CLI

File logging uses a custom handler with immediate flushing for real-time
log file updates.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# edited nov 18, 2025 to integrate rich-based logging
# purpose: to set up logger for process with rich styling
###########################################################################################

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Module-level logger instance
_logger: Optional[logging.Logger] = None


def get_rich_handler(
    console: Optional[Console] = None,
    level: int = logging.INFO,
    show_path: bool = False,
) -> RichHandler:
    """Create a rich logging handler with consistent styling.

    Parameters
    ----------
    console : Console, optional
        Rich console instance. If None, creates a new console.
    level : int, default=logging.INFO
        Logging level for the handler.
    show_path : bool, default=False
        Whether to show the file path in log messages.

    Returns
    -------
    RichHandler
        Configured rich logging handler.
    """
    if console is None:
        console = Console(stderr=True)

    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    handler.setLevel(level)
    return handler


def get_file_handler(
    log_path: Path | str,
    level: int = logging.INFO,
    mode: str = "w",
) -> logging.FileHandler:
    """Create a file logging handler with auto-flushing.

    Parameters
    ----------
    log_path : Path | str
        Path to the log file.
    level : int, default=logging.INFO
        Logging level for the handler.
    mode : str, default="w"
        File mode ('w' for overwrite, 'a' for append).

    Returns
    -------
    FlushFileHandler
        File handler with automatic flushing after each log message.
    """

    class FlushFileHandler(logging.FileHandler):
        """File handler that flushes after each emit for real-time updates."""

        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    handler = FlushFileHandler(str(log_path), mode=mode, encoding="utf-8")
    handler.setLevel(level)

    # Use simpler formatting for file output (no rich markup)
    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler


def get_logger(
    name: str = "forecast_logger",
    level: int = logging.INFO,
    console: Optional[Console] = None,
) -> logging.Logger:
    """Get or create the main forecast logger instance.

    This function returns a singleton logger instance. If the logger hasn't been
    created yet, it initializes it with a rich console handler.

    Parameters
    ----------
    name : str, default="forecast_logger"
        Name of the logger.
    level : int, default=logging.INFO
        Logging level.
    console : Console, optional
        Rich console instance for the handler. If None, creates a new console.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _logger  # noqa: PLW0603

    if _logger is not None:
        return _logger

    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    _logger.propagate = False

    # Add rich handler if no handlers exist
    if not _logger.hasHandlers():
        setup_console_logging(_logger, console=console, level=level)

    return _logger


def setup_console_logging(
    logger_instance: Optional[logging.Logger] = None,
    console: Optional[Console] = None,
    level: int = logging.INFO,
    show_path: bool = False,
) -> None:
    """Configure rich-based console logging with styled output.

    Parameters
    ----------
    logger_instance : logging.Logger, optional
        Logger to configure. If None, uses the module-level logger.
    console : Console, optional
        Rich console instance. If None, creates a new console.
    level : int, default=logging.INFO
        Logging level.
    show_path : bool, default=False
        Whether to show file paths in log messages.
    """
    if logger_instance is None:
        logger_instance = get_logger()

    # Remove existing console handlers to avoid duplicates
    logger_instance.handlers = [
        h for h in logger_instance.handlers if not isinstance(h, RichHandler)
    ]

    handler = get_rich_handler(console=console, level=level, show_path=show_path)
    logger_instance.addHandler(handler)


def setup_file_logging(
    run_dir: Path | str,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    filename: str = "log.txt",
) -> Path:
    """Configure file-based logging in the specified run directory.

    Parameters
    ----------
    run_dir : Path | str
        Directory where log file will be created.
    logger_instance : logging.Logger, optional
        Logger to configure. If None, uses the module-level logger.
    level : int, default=logging.INFO
        Logging level for the file handler.
    filename : str, default="log.txt"
        Name of the log file.

    Returns
    -------
    Path
        Path to the created log file.
    """
    if logger_instance is None:
        logger_instance = get_logger()

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / filename

    # Remove existing file handlers to avoid duplicates
    logger_instance.handlers = [
        h for h in logger_instance.handlers if not isinstance(h, logging.FileHandler)
    ]

    handler = get_file_handler(log_path, level=level)
    logger_instance.addHandler(handler)

    return log_path


# Initialize module-level logger with rich console handler
logger = get_logger()
