"""Centralized logging utilities for data pipelines."""

from __future__ import annotations

import logging
import sys
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union


LogFile = Optional[Union[str, PathLike[str]]]


class PipelineLogger:
    """Configurable logger shared by pipeline modules.

    Handlers are identified by destination, so creating the same logger more
    than once does not duplicate each log message.
    """

    DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    _HANDLER_MARKER = "_pipeline_logger_handler"
    _HANDLER_KEY = "_pipeline_logger_key"

    def __init__(
        self,
        name: str = "data_pipeline",
        level: int = logging.INFO,
        log_file: LogFile = None,
        console: bool = True,
        fmt: str = DEFAULT_FORMAT,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._formatter = logging.Formatter(fmt)
        self._configure_console(console)
        self._configure_file(log_file)

    @property
    def logger(self) -> logging.Logger:
        """Return the underlying standard-library logger."""
        return self._logger

    def set_level(self, level: int) -> None:
        """Change the minimum level accepted by this logger."""
        self._logger.setLevel(level)

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.error(message, *args, **kwargs)

    def exception(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(message, *args, **kwargs)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(message, *args, **kwargs)

    def log(self, level: int, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.log(level, message, *args, **kwargs)

    def _configure_console(self, enabled: bool) -> None:
        if not enabled:
            return
        handler = logging.StreamHandler(sys.stdout)
        self._add_handler(handler, ("console",))

    def _configure_file(self, log_file: LogFile) -> None:
        if log_file is None:
            return
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        self._add_handler(handler, ("file", str(path.resolve())))

    def _add_handler(
        self,
        handler: logging.Handler,
        key: tuple[str, ...],
    ) -> None:
        for existing in self._logger.handlers:
            if getattr(existing, self._HANDLER_MARKER, False) and getattr(
                existing, self._HANDLER_KEY, None
            ) == key:
                handler.close()
                return

        handler.setFormatter(self._formatter)
        setattr(handler, self._HANDLER_MARKER, True)
        setattr(handler, self._HANDLER_KEY, key)
        self._logger.addHandler(handler)
