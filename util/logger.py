"""Logging configuration and setup utilities.

This module configures logging for the GACA early warning pipeline. It sets up both
console and file-based logging handlers with consistent formatting. The logger is
configured to output to stdout for real-time monitoring and to a log file for
persistent record-keeping.

Functions
---------
setup_logging
    Configure file-based logging in the specified run directory.

Attributes
----------
logger : logging.Logger
    Main logger instance used throughout the pipeline.

Notes
-----
The logger is configured with INFO level and includes timestamps, log levels, and
messages. File logging uses a custom handler with immediate flushing for real-time
log file updates.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to set up logger for process
###########################################################################################

import logging
import os
import sys


logger = logging.getLogger("forecast_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

# formatter for both stdout + file
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

if not logger.hasHandlers():
    # stream handler (live stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def setup_logging(run_dir: str) -> None:
    """Add file handler to logger to persist logs in the specified run directory."""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")

    # class to ensure real-time log updates
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
