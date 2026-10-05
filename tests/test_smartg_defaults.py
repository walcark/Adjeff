"""The Smart-G defaults adjeff bets on by not passing them.

A signature check catches a renamed parameter.  It cannot catch a
parameter that keeps its name and changes its default, which is invisible
in the diff of every call site and silent at runtime.  Smart-G 2.0 moved
``Sensor.th_deg`` from 0 to 180: the two sensors of ``tdif_up`` and
``tdif_down`` did not pass it, turned from zenith to nadir, and returned
zeros.  Nothing but the 6S reciprocity invariant found it, after three
hours and several two-minute GPU runs.

Every parameter below is one adjeff leaves out of a call and whose value
changes a result.  Freezing them turns the next such move into a one
second failure naming the parameter.

When one of these fails, the question is not how to make it pass: it is
whether the new default is the one adjeff wants.  If it is, update the
value here.  If it is not, pass the parameter explicitly at the call site
and delete its line.
"""

from __future__ import annotations

import inspect

import pytest

#: ``(callable, parameter): default adjeff relies on``.
#:
#: Restricted on purpose to the parameters that carry physics or shape an
#: output.  Freezing all 108 defaults of the Smart-G API would fail on
#: every unrelated release and teach the reader to update the file
#: without thinking.
EXPECTED: dict[tuple[str, str], object] = {
    # Geometry of a sensor.  adjeff builds several and leaves the rest of
    # the placement to these.
    ("Sensor", "th_deg"): 180.0,
    ("Sensor", "ph_deg"): 180.0,
    ("Sensor", "pos_x"): 0.0,
    ("Sensor", "pos_y"): 0.0,
    ("Sensor", "pos_z"): 0.0,
    ("Sensor", "loc"): "SURF0P",
    ("Sensor", "fov"): 0.0,
    ("Sensor", "sensor_type"): 0,
    # Mode of the engine.  adjeff only ever passes `autoinit` and
    # `obj3d`, so a plane-parallel, double-precision, forward run is
    # assumed everywhere.
    ("Smartg", "pp"): True,
    ("Smartg", "double"): True,
    ("Smartg", "back"): False,
    ("Smartg", "alis"): False,
    ("Smartg", "thermal"): False,
    # Physics of a run.
    ("run", "depol"): 0.0279,
    ("run", "earth_radius"): 6371.0,
    ("run", "surface"): None,
    ("run", "environment"): None,
    ("run", "water"): None,
    # Shape of a run's output.  `adapt_smartg_output` reads the result by
    # dimension name and size, so these two set what it receives.
    ("run", "n_theta"): 45,
    ("run", "n_phi"): 90,
    ("run", "output_layers"): 0,
    ("run", "flux"): None,
    ("run", "le"): None,
    ("run", "stdev"): False,
    # Scene description.
    ("Environment", "env"): 0,
    ("Environment", "env_size"): 1.0e6,
    ("Atm1D", "lat"): 45.0,
    ("Atm1D", "no2"): True,
    ("Atm1D", "tau_r"): None,
    ("Atm1D.calc", "n_theta"): "native",
}


def _callables() -> dict[str, object]:
    """Return the Smart-G callables the frozen defaults belong to."""
    from smartg.atmosphere import Atm1D
    from smartg.sensor import Sensor
    from smartg.smartg import Smartg
    from smartg.surface import Environment

    return {
        "Sensor": Sensor.__init__,
        "Smartg": Smartg.__init__,
        "run": Smartg.run,
        "Environment": Environment.__init__,
        "Atm1D": Atm1D.__init__,
        "Atm1D.calc": Atm1D.calc,
    }


@pytest.mark.integration
@pytest.mark.parametrize(("where", "expected"), sorted(EXPECTED.items(), key=str))
def test_the_default_adjeff_relies_on_has_not_moved(where, expected):
    """A default that moves is silent at the call site and at runtime."""
    name, parameter = where
    signature = inspect.signature(_callables()[name])

    assert parameter in signature.parameters, (
        f"{name} no longer has a {parameter!r} parameter; adjeff was "
        "relying on its default"
    )
    actual = signature.parameters[parameter].default
    assert actual == expected, (
        f"{name}.{parameter} now defaults to {actual!r}, not {expected!r}. "
        "Decide whether adjeff wants the new value, or pass the parameter "
        "explicitly at the call site and drop it from EXPECTED."
    )


@pytest.mark.integration
def test_every_frozen_default_names_a_real_parameter():
    """Guard against a stale entry that can no longer fail."""
    calls = _callables()
    stale = [
        f"{name}.{parameter}"
        for (name, parameter) in EXPECTED
        if parameter not in inspect.signature(calls[name]).parameters
    ]
    assert not stale, "frozen defaults naming nothing:\n" + "\n".join(stale)
