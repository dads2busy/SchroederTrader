import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from schroeder_trader.config import LOG_DIR


def setup_logging(log_dir: Path = LOG_DIR) -> None:
    """Configure logging with console (INFO) and file (DEBUG) handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("schroeder_trader")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler — INFO
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — DEBUG with rotation
    file_handler = RotatingFileHandler(
        log_dir / "schroeder_trader.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
