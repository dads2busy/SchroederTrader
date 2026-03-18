import logging

from schroeder_trader.logging_config import setup_logging


def test_setup_logging_configures_root_logger(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)

    logger = logging.getLogger("schroeder_trader")
    assert logger.level == logging.DEBUG

    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types

    logger.handlers.clear()


def test_log_file_created(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)

    logger = logging.getLogger("schroeder_trader")
    logger.info("test message")

    log_file = log_dir / "schroeder_trader.log"
    assert log_file.exists()

    logger.handlers.clear()
