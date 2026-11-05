"""Logging: console + rotating file with IST timestamps."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler

from .config import data_dir

IST = timezone(timedelta(hours=5, minutes=30))


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, "INFO", logging.INFO)
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(ISTFormatter("[%(asctime)s] %(levelname)-7s %(name)-22s %(message)s"))
    logger.addHandler(console)

    try:
        fh = RotatingFileHandler(data_dir() / "logs" / "bot.log", maxBytes=2_000_000,
                                 backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(ISTFormatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(fh)
    except Exception:
        pass

    logger.propagate = False
    return logger
