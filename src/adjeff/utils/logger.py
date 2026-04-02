"""Define a custom logger for Adjeff and the associated configuration class.

The logger can :
- Filter redondant messages,

"""

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import colorlog


@dataclass
class LoggerConfig:
    """Configuration for the global Adjeff logger."""

    name: str = "AdjeffLogging"
    level: int = logging.INFO
    file: Optional[str] = None
    formatter: Optional[str] = None


class RedundantMessageFilter(logging.Filter):
    """Filter to avoid logging the same message multiple times in a row."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._last_key: Optional[tuple[str, int, str]] = None

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter the record if it matches the precedent.

        Parameters
        ----------
        record : logging.LogRecord
            The record to be logged.

        Returns
        -------
        bool
            Whether the record should be logger or not.
        """
        key = (record.name, record.levelno, record.getMessage())
        if key == self._last_key:
            return False
        self._last_key = key
        return True


class AdjeffLogging:
    """Singleton-style logger for the Adjeff library.

    Parameters
    ----------
    config : Optional[LoggerConfig]
        The configuration of the logger.
    """

    _instance: Optional["AdjeffLogging"] = None

    def __new__(cls, config: Optional[LoggerConfig] = None) -> "AdjeffLogging":
        """Ensure a single instance of the singleton class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        """Initialize the singleton logger instance.

        Parameters
        ----------
        config : Optional[LoggerConfig]
            Configuration of the logger.
        """
        if getattr(self, "_initialized", False):
            return

        self.config = config or LoggerConfig()
        self.logger = logging.getLogger(self.config.name)
        self.logger.propagate = False

        self._apply_config(self.config)

        self._initialized = True

    def _apply_config(self, config: LoggerConfig) -> None:
        """Apply a configuration to the logger.

        Parameters
        ----------
        config : LoggerConfig
            Configuration to apply.
        """
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        self.logger.setLevel(config.level)
        self.logger.propagate = False

        # Console formatter (colored)
        console_formatter = colorlog.ColoredFormatter(
            fmt=config.formatter
            or "[%(asctime)s][ADJEFF][%(filename)s:%(funcName)s]"
            "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt="%Y/%m/%d-%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(config.level)
        console.setFormatter(console_formatter)
        console.addFilter(RedundantMessageFilter())
        self.logger.addHandler(console)

        # File handler (optional, non-colored)
        if config.file:
            file_formatter = logging.Formatter(
                config.formatter
                or "[%(asctime)s][ADJEFF][%(filename)s:%(funcName)s]"
                "[%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(config.file, encoding="utf-8")
            file_handler.setLevel(config.level)
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(RedundantMessageFilter())
            self.logger.addHandler(file_handler)

    def set_config(self, config: LoggerConfig) -> None:
        """Update the logger configuration on the fly.

        Parameters
        ----------
        config : LoggerConfig
            New configuration for the logger.
        """
        self.config = config
        self._apply_config(config)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger.

        Returns
        -------
        logging.Logger
            The underlying logger instance.
        """
        return self.logger

    def log_execution_time(
        self, func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Log the execution time of a method.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to be decorated.

        Returns
        -------
        Callable[..., Any]
            Decorated function that logs execution time.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.logger.info("%s took %.3f seconds.", func.__name__, duration)
            return result

        return wrapper


class MultilineConsoleRenderer:
    """Structlog renderer that prints each context field on its own line.

    Produces output like::

        2026-04-02 08:52:22 [debug    ] Aggregated parameters.
            das     = {'wl': ['wl'], 'aot': ['x', 'y']}
            scalars = []
            vectors = ['wl', 'aot', 'rh']

    Parameters
    ----------
    indent : int
        Number of spaces to indent field lines.
    pad_keys : bool
        If True, align the ``=`` signs by padding key names to the same width.
    colors : bool
        If True, colorize the level token using ANSI codes.
    """

    _LEVEL_COLORS = {
        "debug": "\033[36m",  # cyan
        "info": "\033[32m",  # green
        "warning": "\033[33m",  # yellow
        "error": "\033[31m",  # red
        "critical": "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(
        self,
        indent: int = 4,
        pad_keys: bool = True,
        colors: bool = True,
    ) -> None:
        self._indent = " " * indent
        self._pad_keys = pad_keys
        self._colors = colors

    def __call__(
        self,
        logger: Any,
        method: str,
        event_dict: dict[str, Any],
    ) -> str:
        """Process the log and applies a multi-line rendering."""
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", method)
        event = event_dict.pop("event", "")

        level_str = f"{level:<8}"
        if self._colors:
            color = self._LEVEL_COLORS.get(level.lower(), "")
            level_str = f"{color}{level_str}{self._RESET}"

        header = f"{timestamp} [{level_str}] {event}"

        if not event_dict:
            return header

        if self._pad_keys:
            width = max(len(k) for k in event_dict)
            lines = [
                f"{self._indent}{k:<{width}} = {v}"
                for k, v in event_dict.items()
            ]
        else:
            lines = [f"{self._indent}{k} = {v}" for k, v in event_dict.items()]

        return header + "\n" + "\n".join(lines)


# Default logger
adjeff_logging: AdjeffLogging = AdjeffLogging(
    LoggerConfig(level=logging.INFO, file="adjeff.log")
)
