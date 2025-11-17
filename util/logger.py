###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# edited nov 13, 2025 to adjust for inference + deployment
# purpose: to set up logger for process
###########################################################################################

import logging
import sys
import os

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

def setup_logging(run_dir):
    """
    func that adds file handler to logger to also save to a file in run_dir
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")
    
    # class to ensure real-time log updates
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
            
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
