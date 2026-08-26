"""The public API's docstrings must agree with the functions they document.

Nothing enforced this, and the drift was real: forty-four parameters
carried a default the docstring did not call ``optional``, and the same
``cache`` argument was described eight different ways across eight
functions.
"""

import inspect
import re

import pytest

import adjeff.api as api

PUBLIC = [
    (name, getattr(api, name))
    for name in api.__all__
    if inspect.isfunction(getattr(api, name))
]


def documented_parameters(func) -> dict[str, str]:
    """Return the ``name -> type`` line of every parameter a docstring lists."""
    doc = inspect.getdoc(func) or ""
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Parameters")
    except StopIteration:
        return {}
    found = {}
    for line in lines[start + 2 :]:
        if re.fullmatch(r"[A-Z][a-z]+", line.strip()):
            break
        match = re.fullmatch(r"(\w+) : (.+)", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def defaulted_parameters(func) -> set[str]:
    """Return the names of the parameters that carry a default value."""
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.default is not inspect.Parameter.empty
    }


@pytest.mark.parametrize(("name", "func"), PUBLIC, ids=[n for n, _ in PUBLIC])
def test_a_parameter_with_a_default_is_documented_as_optional(name, func):
    """Numpydoc spells a default as ``optional`` on the type line."""
    documented = documented_parameters(func)
    optional = defaulted_parameters(func)

    wrong = sorted(
        p
        for p, kind in documented.items()
        if (p in optional) is not kind.rstrip().endswith("optional")
    )
    assert not wrong, f"{name}: type line disagrees with the signature for {wrong}"


@pytest.mark.parametrize(("name", "func"), PUBLIC, ids=[n for n, _ in PUBLIC])
def test_every_documented_parameter_exists(name, func):
    """A docstring must not describe an argument the function does not take."""
    taken = set(inspect.signature(func).parameters)
    ghosts = sorted(set(documented_parameters(func)) - taken)

    assert not ghosts, f"{name}: documents {ghosts}, which it does not accept"


def test_the_shared_cache_argument_reads_the_same_everywhere():
    """One argument, one description: eight wordings was seven too many."""
    descriptions = set()
    for _, func in PUBLIC:
        doc = inspect.getdoc(func) or ""
        match = re.search(r"^cache : (.+)\n((?:[ ]+.+\n?)+)", doc, re.M)
        if match:
            descriptions.add((match.group(1), " ".join(match.group(2).split())))

    assert len(descriptions) == 1, f"cache is described {len(descriptions)} ways"
