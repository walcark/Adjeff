"""adjeff must be silent on import and reachable through `logging`.

Left to its defaults structlog writes to stdout outside `logging`
altogether and filters nothing, which is why all six notebooks begin by
redirecting it to /dev/null.  These tests pin the two properties that
replaced it: nothing is printed unless asked, and everything the package
emits is an ordinary `logging` record.
"""

import io
import logging
from contextlib import redirect_stderr, redirect_stdout

import pytest

from adjeff._logging import ROOT, get_logger
from adjeff.core import S2Band, gaussian_image_dict


def test_the_package_logger_carries_only_a_null_handler():
    """A library installs a NullHandler and no more; handlers are the app's."""
    handlers = logging.getLogger(ROOT).handlers

    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.NullHandler)


def test_producing_a_scene_prints_nothing():
    """Real work, with no logging configured, must reach neither stream.

    This is the property the notebooks had to fake by silencing structlog
    themselves.
    """
    out, err = io.StringIO(), io.StringIO()

    with redirect_stdout(out), redirect_stderr(err):
        gaussian_image_dict(sigma=0.4, res_km=0.01, n=9, bands=[S2Band.B02])

    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_a_log_line_arrives_as_an_ordinary_record(caplog):
    """`caplog` sees adjeff's lines, which means any handler does."""
    with caplog.at_level(logging.DEBUG, logger=ROOT):
        gaussian_image_dict(sigma=0.4, res_km=0.01, n=9, bands=[S2Band.B02])

    assert caplog.records
    assert all(r.name.startswith(ROOT) for r in caplog.records)


def test_a_module_outside_the_namespace_is_re_homed():
    """One level setting must reach everything the package emits."""
    assert get_logger("adjeff.core.thing").name == "adjeff.core.thing"
    assert get_logger("thing").name == "adjeff.thing"
    assert get_logger(ROOT).name == ROOT


def test_key_values_survive_the_trip(caplog):
    """Structlog's whole point: the data stays data, not a formatted string."""
    log = get_logger("adjeff.test")

    with caplog.at_level(logging.INFO, logger=ROOT):
        log.info("probe.event", band="B03", n=1999)

    (record,) = caplog.records
    assert record.msg["event"] == "probe.event"
    assert record.msg["band"] == "B03"
    assert record.msg["n"] == 1999


def test_bind_still_binds(caplog):
    """`SceneModule` binds its own name and cache key; that must keep working."""
    log = get_logger("adjeff.test").bind(module="Probe")

    with caplog.at_level(logging.INFO, logger=ROOT):
        log.bind(key="abc123").info("probe.event")

    (record,) = caplog.records
    assert record.msg["module"] == "Probe"
    assert record.msg["key"] == "abc123"


def test_context_variables_reach_the_record(caplog):
    """A run id bound once must appear on every line below it."""
    import structlog

    log = get_logger("adjeff.test")
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id="r-1")
    try:
        with caplog.at_level(logging.INFO, logger=ROOT):
            log.info("probe.event")
    finally:
        structlog.contextvars.clear_contextvars()

    (record,) = caplog.records
    assert record.msg["run_id"] == "r-1"


def test_the_dead_renderer_is_gone():
    """`MultilineConsoleRenderer` was configured nowhere, in src or out."""
    with pytest.raises(ImportError):
        import adjeff.utils.logger  # noqa: F401
