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


# --- durations and run context ---


def test_timed_brackets_the_work(caplog):
    """A start line, a done line, and the duration nobody used to record."""
    from adjeff._logging import timed

    log = get_logger("adjeff.test")
    with caplog.at_level(logging.INFO, logger=ROOT):
        with timed(log, "probe", n=3) as outcome:
            outcome["cached"] = False

    start, done = caplog.records
    assert start.msg["event"] == "probe.start"
    assert start.msg["n"] == 3
    assert done.msg["event"] == "probe.done"
    assert done.msg["cached"] is False
    assert isinstance(done.msg["duration_s"], float)


def test_timed_says_where_a_failure_happened(caplog):
    """A run that dies mid-way must still say where and how long it got."""
    from adjeff._logging import timed

    log = get_logger("adjeff.test")
    with caplog.at_level(logging.INFO, logger=ROOT):
        with pytest.raises(ValueError):
            with timed(log, "probe", n=3):
                raise ValueError("boom")

    _, failed = caplog.records
    assert failed.msg["event"] == "probe.failed"
    assert failed.msg["error"] == "ValueError"
    assert "duration_s" in failed.msg


def test_run_context_unbinds_only_what_it_bound(caplog):
    """Nesting must work: an inner context cannot drop an outer one."""
    from adjeff._logging import run_context

    log = get_logger("adjeff.test")
    with caplog.at_level(logging.INFO, logger=ROOT):
        with run_context(run_id="r-1"):
            with run_context(band="B03"):
                log.info("probe.inner")
            log.info("probe.outer")

    inner, outer = caplog.records
    assert inner.msg["run_id"] == "r-1"
    assert inner.msg["band"] == "B03"
    assert outer.msg["run_id"] == "r-1"
    assert "band" not in outer.msg


def test_every_scene_module_brackets_its_work(caplog):
    """The coverage rule, checked rather than trusted.

    Each `SceneModule` must emit an entry line and an exit line carrying
    a duration.  This is inherited from `forward`, so it holds for a
    module written later too, but nothing enforced it before.
    """
    from _test_module import TestModule

    scene = gaussian_image_dict(sigma=0.4, res_km=0.01, n=9, bands=[S2Band.B02])
    module = TestModule()

    with caplog.at_level(logging.INFO, logger=ROOT):
        module(scene)

    events = [r.msg["event"] for r in caplog.records]
    assert "module.start" in events
    assert "module.done" in events

    (done,) = [r.msg for r in caplog.records if r.msg["event"] == "module.done"]
    assert done["module"] == "TestModule"
    assert done["bands"] == 1
    assert "duration_s" in done
    assert done["cached"] is False


# --- the object.action convention, checked rather than trusted ---

#: `object.action`, lowercase, dot-separated, no space and no full stop.
EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")

#: `_logging` builds `<event>.start` and `<event>.done` from its argument,
#: so its f-strings are the mechanism rather than a violation of it.
CONVENTION_EXEMPT = {"_logging.py"}

LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def log_calls():
    """Yield every log call in the package as (file, line, first argument)."""
    import ast
    from pathlib import Path

    import adjeff

    root = Path(adjeff.__file__).parent
    for path in sorted(root.rglob("*.py")):
        if path.name in CONVENTION_EXEMPT:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
                continue
            if not isinstance(func.value, ast.Name):
                continue
            if func.value.id not in {"logger", "log", "_log"}:
                continue
            first = node.args[0] if node.args else None
            yield path.relative_to(root), node.lineno, first


def test_the_package_actually_logs_something():
    """A guard on the guards below: an empty walk would pass everything."""
    assert len(list(log_calls())) >= 15


def test_no_log_message_is_built_by_interpolation():
    """A pre-formatted string cannot be filtered, plotted, or serialised.

    It is the one thing that undoes structlog, and `optim/fit.py` and both
    optimisers used to do it on the hottest path in the package.
    """
    import ast

    offenders = [
        f"{path}:{line}"
        for path, line, arg in log_calls()
        if isinstance(arg, (ast.JoinedStr, ast.BinOp))
    ]

    assert not offenders, f"f-string log messages at {offenders}"


def test_every_log_message_follows_the_convention():
    """`objet.action`: lowercase, dotted, no space, no full stop."""
    import ast

    offenders = [
        f"{path}:{line} {arg.value!r}"
        for path, line, arg in log_calls()
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and not EVENT_NAME.match(arg.value)
    ]

    assert not offenders, f"messages off convention: {offenders}"
