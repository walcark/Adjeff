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

__all__ = ["get_logger"]

#: Root of adjeff's logger namespace.  Everything the package emits hangs
#: below it, so one call sets the level for all of it.
ROOT = "adjeff"

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
