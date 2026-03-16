"""Centralized loguru configuration."""

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Remove the default loguru handler and re-add one at *level*."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper())
