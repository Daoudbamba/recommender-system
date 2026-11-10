"""Journalisation homogene pour les scripts et les modules."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retourne un logger configure une seule fois (pas de handlers dupliques)."""
    root = logging.getLogger("recsys")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(level)
    return logging.getLogger(name if name.startswith("recsys") else f"recsys.{name}")
