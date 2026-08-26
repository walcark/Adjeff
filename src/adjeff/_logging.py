"""How adjeff emits log lines, and why it goes through the standard library.

structlog is the writing API here, not the transport.  Left to its
defaults it is both: ``PrintLoggerFactory`` writes straight to stdout,
outside :mod:`logging` altogether, and ``BoundLoggerFilteringAtNotset``
filters nothing, so every ``debug`` line prints and no level setting can
stop it.  That is a pipeline of its own, disjoint from the one every
dependency uses, and it has two consequences that were measured rather
than guessed.

A caller who raises adjeff's level does not quiet ``zarr``, and a caller
who quiets ``zarr`` does not raise ``xsweep``: over one forward run,
``xsweep`` emitted twenty-one ``INFO`` records carrying the point counts,
the cache decisions and the elapsed time of every sweep, and adjeff
showed none of them.  And there being no level to set at all, the six
notebooks that document this package all begin by redirecting structlog
to ``/dev/null``.

So the loggers built here wrap :func:`logging.getLogger`, and adjeff
speaks through the same transport as everything it depends on.  One level
governs the lot, a caller's own handlers receive adjeff's lines in their
own format, and importing adjeff prints nothing until someone asks it to.
:func:`adjeff.setup_logging` is that asking.

Nothing here touches :func:`structlog.configure`.  Configuring the
process globally is the application's decision, and a library that makes
it for them takes away the one they were entitled to.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

__all__ = ["get_logger", "setup_logging"]

#: Root of adjeff's logger namespace.  Everything the package emits hangs
#: below it, so one call sets the level for all of it.
ROOT = "adjeff"

#: Loggers that report on the same work adjeff is doing, and follow its
#: level.  ``xsweep`` runs the sweeps, and says over one forward run what
#: this package was asked to start saying: point counts, cache decisions
#: and the elapsed time of every call.
COMPANIONS = ("xsweep",)

#: Loggers that talk about their own internals rather than about the run.
#: Over one forward run they accounted for a hundred and seventy of the
#: two hundred and five records that reached the standard library, all of
#: them ``DEBUG``, none of them about the science.  Capped rather than
#: silenced: a warning from any of them is worth reading.
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

#: Run before the record reaches :mod:`logging`.  ``merge_contextvars``
#: is what lets a caller bind a run id once and have every line below
#: carry it, and ``ProcessorFormatter.wrap_for_formatter`` hands the
#: event dict on intact so the final rendering is the handler's choice,
#: made in :func:`adjeff.setup_logging` or by the application.
_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
]

#: Name carried by the handler :func:`setup_logging` installs, so
#: that calling it twice replaces rather than doubles.
_HANDLER_NAME = "adjeff-console"

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> Any:
    """Return the logger a module of adjeff writes through.

    Parameters
    ----------
    name : str
        Usually ``__name__``.  Names outside adjeff's namespace are
        re-homed under it, so that one level setting reaches everything
        the package emits.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Bound logger delegating to :mod:`logging`.  It takes key-value
        pairs and :meth:`bind` like any structlog logger, and its records
        reach whatever handlers the application has installed.
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
    """Send adjeff's log lines, and its dependencies', to the console.

    A library has no business configuring logging for the process it is
    imported into, so adjeff does not, and prints nothing until this is
    called.  What it owes the caller instead is a way to turn logging on
    that takes one line rather than twenty, which is what this is.

    Calling it twice replaces the handler rather than adding a second, so
    a notebook cell can be re-run without doubling every line.

    Parameters
    ----------
    level : str or int, optional
        Level for adjeff and for the loggers listed in :data:`COMPANIONS`,
        which report on the same work (default ``"info"``).
    json : bool, optional
        Render one JSON object per line instead of the console format
        (default ``False``).  For a run whose output is collected rather
        than watched.
    noisy_level : str or int, optional
        Ceiling for the loggers in *quiet* (default ``"warning"``).
    quiet : tuple of str, optional
        Loggers to cap at *noisy_level*.  Defaults to :data:`NOISY`, the
        ones measured to talk about their own internals; pass your own
        tuple to widen or narrow it, or an empty one to cap nothing.

    Notes
    -----
    Python's :mod:`warnings` are routed here too, so a
    ``ResourceWarning`` from a Smart-G call lands in the same stream as
    everything else rather than on stderr in another format.

    Examples
    --------
    >>> import adjeff
    >>> adjeff.setup_logging(level="info")  # doctest: +SKIP
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
