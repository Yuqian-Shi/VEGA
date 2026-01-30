import argparse
import logging
import time
from pathlib import Path
from config_manager import ConfigManager

def initialize_logger():
    """Initialize logger with console and file handlers."""
    log_dir = "log_files"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{log_dir}/{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}.log"
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    return logger, log_file

def load_configuration():
    """Load and merge configuration from YAML.

    Args:
        cli_args: Parsed CLI arguments (only config and profile)

    Returns:
        argparse.Namespace with merged configuration
    """
    # Get config file path from CLI or use default
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark (using unified config)"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="all_config.yaml",
        help="Path to config YAML file",
    )
    cli_args = parser.parse_args()
    config_file = cli_args.config

    # Load base config
    config_manager = ConfigManager(config_file)

    # Convert to Namespace
    logger, log_file = initialize_logger()
    config_manager.log_path = log_file
    return config_manager, logger