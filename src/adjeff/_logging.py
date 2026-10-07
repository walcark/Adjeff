"""Logging of adjeff, through the standard library.

Loggers are structlog wrappers around :func:`logging.getLogger`, so that
one level governs adjeff and its dependencies, and the application's
handlers receive everything.  Nothing prints until :func:`setup_logging`
is called; :func:`structlog.configure` is never touched.

Functions
---------
    get_logger
        Logger of an adjeff module.
    setup_logging
        Send adjeff's and its dependencies' logs to the console.
    timed
        Log the start, end and duration of a block.
    run_context
        Bind fields onto every line logged inside a block.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

__all__ = ["get_logger", "run_context", "setup_logging", "timed"]

#: Root of adjeff's logger namespace.
ROOT = "adjeff"

#: Loggers that follow adjeff's level: they report on the same work.
COMPANIONS = ("xsweep",)

#: Loggers capped at *noisy_level*: chatty about their own internals.
NOISY = (
    "zarr",
    "numcodecs",
    "matplotlib",
    "asyncio",
    "h5py",
    "trimesh",
    "PIL",
    "fsspec",
)

#: Applied before the record reaches :mod:`logging`; the rendering is
#: left to the handler.
_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
]

#: Name of the handler :func:`setup_logging` installs, so it is replaced.
_HANDLER_NAME = "adjeff-console"

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> Any:
    """Return a structlog logger writing through :mod:`logging`.

    *name*, usually ``__name__``, is placed under ``adjeff`` if outside.
    """
    if name != ROOT and not name.startswith(ROOT + "."):
        name = f"{ROOT}.{name}"
    return structlog.wrap_logger(
        logging.getLogger(name),
        processors=_PROCESSORS,
        wrapper_class=structlog.stdlib.BoundLogger,
    )


def setup_logging(
    level: str | int = "info",
    *,
    json: bool = False,
    noisy_level: str | int = "warning",
    quiet: tuple[str, ...] = NOISY,
) -> None:
    """Send adjeff's and its dependencies' logs, and warnings, to the console.

    A second call replaces the handler rather than adding one.

    Parameters
    ----------
    level : str or int, optional
        Level of adjeff and :data:`COMPANIONS`, ``"info"`` by default.
    json : bool, optional
        One JSON object per line instead of the console format.
    noisy_level : str or int, optional
        Level of the loggers in *quiet* and of warnings, ``"warning"``.
    quiet : tuple of str, optional
        Loggers capped at *noisy_level*, :data:`NOISY` by default.

    Examples
    --------
    >>> adjeff.setup_logging(level="debug", json=True)  # doctest: +SKIP
    """
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=not json)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.set_name(_HANDLER_NAME)

    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.name == _HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    for name in (ROOT, *COMPANIONS):
        logging.getLogger(name).setLevel(_as_level(level))
    for name in quiet:
        logging.getLogger(name).setLevel(_as_level(noisy_level))

    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(_as_level(noisy_level))


def _as_level(level: str | int) -> int:
    """Return *level* as the integer :mod:`logging` works in."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(
            f"unknown log level {level!r}; expected one of debug, info, "
            "warning, error, critical, or an integer"
        )
    return resolved


@contextmanager
def timed(log: Any, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log ``<event>.start``, then ``<event>.done`` with ``duration_s``.

    The yielded dict is added to the done line, for what is only known
    at the end.  On failure, ``<event>.failed`` is logged and the
    exception re-raised.
    """
    log.info(f"{event}.start", **fields)
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    except BaseException as exc:
        log.warning(
            f"{event}.failed",
            **fields,
            **extra,
            error=type(exc).__name__,
            duration_s=round(time.perf_counter() - started, 3),
        )
        raise
    log.info(
        f"{event}.done",
        **fields,
        **extra,
        duration_s=round(time.perf_counter() - started, 3),
    )


@contextmanager
def run_context(**fields: Any) -> Iterator[None]:
    """Bind *fields* onto every line logged inside the block; nestable."""
    tokens = structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
