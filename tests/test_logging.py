"""adjeff must be silent on import and reachable through `logging`.

Left to its defaults structlog writes to stdout outside `logging`
altogether and filters nothing, which is why all six notebooks begin by
redirecting it to /dev/null.  These tests pin the two properties that
replaced it: nothing is printed unless asked, and everything the package
emits is an ordinary `logging` record.
"""

import io
import logging
import re
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


# --- setup_logging: one call, one level, the dependencies included ---


@pytest.fixture
def clean_logging():
    """Undo whatever `setup_logging` did, so tests do not leak into each other."""
    root = logging.getLogger()
    before = list(root.handlers), root.level
    levels = {
        name: logging.getLogger(name).level
        for name in (ROOT, "xsweep", "zarr", "py.warnings")
    }
    yield
    root.handlers, root.level = before
    for name, lvl in levels.items():
        logging.getLogger(name).setLevel(lvl)
    logging.captureWarnings(False)


#: The console renderer colours its output, so a plain substring check
#: fails on `n=3` where the stream holds `\x1b[36mn\x1b[0m=\x1b[35m3\x1b[0m`.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def rendered(capture: io.StringIO) -> str:
    """Return what the installed handler wrote, without the colour codes."""
    return ANSI.sub("", capture.getvalue())


def test_setup_logging_shows_adjeff(clean_logging):
    """The point of the call."""
    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf):
        setup_logging(level="info")
        get_logger("adjeff.test").info("probe.event", n=3)

    assert "probe.event" in rendered(buf)
    assert "n=3" in rendered(buf)


def test_setup_logging_lifts_the_companion_loggers(clean_logging):
    """`xsweep` reports on the same work and must follow adjeff's level.

    Over one forward run it emitted twenty-one INFO records that adjeff
    used to discard: point counts, cache decisions, elapsed time.
    """
    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf):
        setup_logging(level="info")
        logging.getLogger("xsweep").info("sweep done ok=1 elapsed=2.2s")

    assert "sweep done" in rendered(buf)


def test_setup_logging_caps_the_noisy_ones(clean_logging):
    """170 of 205 records over one run came from these, all of them debug."""
    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf):
        setup_logging(level="debug")
        logging.getLogger("zarr.group").debug("opening group")
        logging.getLogger("numcodecs").debug("registering codec")

    assert rendered(buf) == ""


def test_a_capped_logger_still_gets_through_when_it_warns(clean_logging):
    """Capped, not silenced: a warning from zarr is worth reading."""
    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf):
        setup_logging(level="info")
        logging.getLogger("zarr.group").warning("store is read-only")

    assert "store is read-only" in rendered(buf)


def test_warnings_join_the_same_stream(clean_logging):
    """A ResourceWarning per Smart-G call used to land on stderr unformatted."""
    import warnings

    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf), warnings.catch_warnings():
        warnings.simplefilter("always")
        setup_logging(level="info")
        warnings.warn("unclosed resource", UserWarning, stacklevel=1)

    assert "unclosed resource" in rendered(buf)


def test_calling_setup_twice_does_not_double_every_line(clean_logging):
    """A notebook cell gets re-run; the output must not grow with it."""
    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf):
        setup_logging(level="info")
        setup_logging(level="info")
        get_logger("adjeff.test").info("probe.event")

    assert rendered(buf).count("probe.event") == 1


def test_json_output_is_one_object_per_line(clean_logging):
    """For a run whose output is collected rather than watched."""
    import json as json_mod

    from adjeff import setup_logging

    buf = io.StringIO()
    with redirect_stderr(buf):
        setup_logging(level="info", json=True)
        get_logger("adjeff.test").info("probe.event", band="B03", n=1999)

    payload = json_mod.loads(rendered(buf).strip())
    assert payload["event"] == "probe.event"
    assert payload["band"] == "B03"
    assert payload["n"] == 1999


def test_an_unknown_level_says_what_the_known_ones_are(clean_logging):
    """`logging.getLevelName` answers a string for garbage, so check it."""
    from adjeff import setup_logging

    with pytest.raises(ValueError, match="unknown log level 'verbose'"):
        setup_logging(level="verbose")
