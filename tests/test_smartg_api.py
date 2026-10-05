"""Every keyword adjeff passes to Smart-G exists in Smart-G.

The 1.2 to 2.0 migration renamed most of the API, in capitals to
snake_case, and a rename done call site by call site misses the ones no
unit test reaches: `Environment(ENV=...)` and `Sensor(POSX=...)` both
survived a first pass and only surfaced on a GPU run minutes long.
Reading the signatures costs milliseconds and covers every call.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _targets() -> dict[str, object]:
    """Return the Smart-G callables adjeff passes keywords to."""
    from smartg.albedo import AlbedoCst, AlbedoMap
    from smartg.atmosphere import AerOPAC, Atm1D
    from smartg.objects3d import Entity, Plane, Transformation
    from smartg.sensor import Sensor
    from smartg.smartg import Smartg
    from smartg.surface import Environment, LambSurface, RTLSSurface

    return {
        "Sensor": Sensor.__init__,
        "Environment": Environment.__init__,
        "LambSurface": LambSurface.__init__,
        "RTLSSurface": RTLSSurface.__init__,
        "AlbedoCst": AlbedoCst.__init__,
        "AlbedoMap": AlbedoMap.__init__,
        "AerOPAC": AerOPAC.__init__,
        "Atm1D": Atm1D.__init__,
        "Entity": Entity.__init__,
        "Plane": Plane.__init__,
        "Transformation": Transformation.__init__,
        "Smartg": Smartg.__init__,
        "run": Smartg.run,
        "calc": Atm1D.calc,
    }


def _calls() -> list[tuple[pathlib.Path, int, str, str]]:
    """Return every (file, line, callee, keyword) adjeff writes."""
    names = set(_targets())
    out = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name not in names:
                continue
            for kw in node.keywords:
                if kw.arg is not None:
                    out.append((path, node.lineno, name, kw.arg))
    return out


@pytest.mark.integration
def test_every_keyword_adjeff_passes_exists_in_smartg():
    """A renamed Smart-G parameter must not reach a GPU run to be caught."""
    allowed = {
        name: set(inspect.signature(fn).parameters) for name, fn in _targets().items()
    }
    unknown = [
        f"{path.name}:{line} {callee}(..., {kw}=...)"
        for path, line, callee, kw in _calls()
        if kw not in allowed[callee]
    ]
    assert not unknown, "keywords Smart-G no longer accepts:\n" + "\n".join(unknown)


@pytest.mark.integration
def test_the_check_actually_sees_the_call_sites():
    """Guard against the walk silently matching nothing."""
    assert len(_calls()) > 40
