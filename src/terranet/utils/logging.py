"""Rich console + file logging with a single call."""
from __future__ import annotations

import logging
from pathlib import Path

try:
    from rich.logging import RichHandler as _Handler
    _HANDLER_KW = {"rich_tracebacks": True, "show_path": False}
except ImportError:  # rich is optional; fall back to std logging
    _Handler = logging.StreamHandler
    _HANDLER_KW = {}

_FMT = "%(asctime)s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.addHandler(_Handler(**_HANDLER_KW))
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)
    return logger
