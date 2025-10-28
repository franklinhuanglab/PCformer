# src/__init__.py
import os
import sys
import yaml
import time
import logging
import functools
import psutil

"""
Logging and Performance Tracking Functions
"""

# Global config and logger
config = {}
logger = None

def setup_logging(log_filename, debug=False):
    """
    Configures a named logger to output messages to both a specified file and the console.
    Also suppresses unnecessary logs from external libraries.

    Args:
        log_filename (str): The path to the log file.
        debug (bool, optional): If True, sets logging level to DEBUG; otherwise, INFO.

    Returns:
        logging.Logger: The configured logger.
    """
    global logger
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    fh = logging.FileHandler(log_filename, mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(ch)

    # Suppress noisy loggers
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    return logger
    
class TeeLogger:
    """
    Redirects standard output and standard error to both the console and a log file.
    """
    def __init__(self, log_file):
        """
        Initializes the TeeLogger with a log file.

        Args:
            log_file (str): Path to the log file.
        """
        self.log_file = open(log_file, "a")
        self.terminal = sys.stdout

    def write(self, message):
        """
        Writes a message to both the console and the log file.

        Args:
            message (str): The message to write.
        """
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):  # For sys.stdout
        """
        Flushes the output buffers of both the console and the log file.
        """
        self.terminal.flush()
        self.log_file.flush()

    def debug(self, msg):
        """Writes a debug message."""
        self.write(f"DEBUG: {msg}\\n")
    def info(self, msg):
        """Writes an informational message."""
        self.write(f"INFO: {msg}\\n")
    def error(self, msg):
        """Writes an error message."""
        self.write(f"ERROR: {msg}\\n")
    def close(self):
        """Closes the log file."""
        self.log_file.close()

def debug_log(msg):
    """
    Logs a debug message if the global configuration is set to debug mode.
    
    Args:
        msg (str): The debug message to print and log.
    """
    if config.get("debug", False):
        # print(msg)
        if logger:
            logger.debug(msg)

def track_performance(func):
    """
    Decorator that measures the execution time and change in memory usage of a function.
    
    Args:
        func (callable): The function to wrap.
    
    Returns:
        callable: The wrapped function that logs performance details upon execution.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if getattr(wrapper, '_tracking', False):
            return func(*args, **kwargs)
        wrapper._tracking = True
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            start_time = time.time()
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss / (1024 ** 2)
            elapsed_time = time.time() - start_time
            msg = f"Function `{func.__name__}` took {elapsed_time:.2f}s and used {mem_after - mem_before:.2f}MB memory."
            print(msg)
            if logger:
                logger.debug(msg)
            return result
        finally:
            wrapper._tracking = False
    return wrapper

def initialize_logging(prefix, context="run"):
    global logger

    """
    # Example: "cache/pretrain/pc_atlas/embed_1024"
    parts = os.path.normpath(prefix).split(os.sep)
    
    try:
        stage = parts[-3]        # "pretrain"
        model_name = parts[-2]   # "pc_atlas"
        embed_size = parts[-1]   # "embed_1024"
        run_id = f"{model_name}_{embed_size}_{stage}"
    except IndexError:
        run_id = os.path.basename(prefix)  # fallback
    
    log_filename = os.path.join(runs_dir, f"{run_id}_debug_{context}_{timestamp}.log")
    """

    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runs_dir = os.path.join(project_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    filename = os.path.basename(prefix)
    prefix = os.path.splitext(filename)[0]

    log_filename = os.path.join(runs_dir, f"{prefix}_debug_{context}_{timestamp}.log")

    logger = setup_logging(log_filename, debug=config.get("debug", False))
    sys.stdout = TeeLogger(log_filename)
    sys.stderr = TeeLogger(log_filename)

    return log_filename

def load_config(path=None):
    """
    Loads a YAML configuration file.

    Args:
        config_file (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing configuration parameters.
    """
    global config

    if not path:  # If path == [None, ""]
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config.update(yaml.safe_load(f))

__all__ = [
    "config",
    "debug_log",
    "initialize_logging",
    "load_config",
    "logger",
    "setup_logging",
    "TeeLogger",
    "track_performance"
]
