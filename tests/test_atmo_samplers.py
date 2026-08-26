"""The six radiative samplers must still say what they used to say.

They now share a base class that builds their static arguments from a
declaration.  A declaration can be wrong in a way a signature cannot, so
these tests read the physics functions and check the declarations against
them.
"""

import inspect
import re

import pytest

import adjeff.atmosphere as atmo
from adjeff.core import S2Band
from adjeff.modules.samplers import (
    RhoAtmSampler,
    SphAlbSampler,
    TdifDownSampler,
    TdifUpSampler,
    TdirDownSampler,
    TdirUpSampler,
)

WITH_GEOMETRY = [
    TdirDownSampler,
    TdirUpSampler,
    TdifDownSampler,
    TdifUpSampler,
    RhoAtmSampler,
]
ALL_SAMPLERS = [*WITH_GEOMETRY, SphAlbSampler]

#: Statics every radiative sampler passes, geometry aside.
ATMOSPHERIC_STATICS = {"species", "afgl_type", "remove_rayleigh", "n_ph"}


@pytest.fixture
def configs():
    """Return an atmosphere, a geometry and a spectral config to build with."""
    return (
        atmo.AtmoConfig(
            aot=[0.1, 0.3], rh=50.0, h=2.0, href=1.0, species={"sulphate": 1.0}
        ),
        atmo.GeoConfig(
            sza=[30.0, 40.0], vza=5.0, saa=100.0, vaa=200.0, sat_height=800.0
        ),
        atmo.SpectralConfig.from_bands([S2Band.B02, S2Band.B03]),
    )


def build(cls, configs, **kwargs):
    """Return an instance of *cls*, with or without a geometry as it needs."""
    atmo_config, geo_config, spectral_config = configs
    if cls is SphAlbSampler:
        return cls(
            atmo_config=atmo_config,
            spectral_config=spectral_config,
            remove_rayleigh=False,
            **kwargs,
        )
    return cls(
        atmo_config=atmo_config,
        geo_config=geo_config,
        spectral_config=spectral_config,
        remove_rayleigh=False,
        **kwargs,
    )


def swept_names(contract: str) -> set[str]:
    """Return the parameter names a contract sweeps over."""
    names: set[str] = set()
    for clause in re.findall(r"(?:batch|vec|loop|const)\(([^)]*)\)", contract):
        names.update(part.strip() for part in clause.split(",") if part.strip())
    return names


@pytest.mark.parametrize("cls", ALL_SAMPLERS, ids=lambda c: c.__name__)
def test_the_declared_statics_are_exactly_what_the_physics_asks_for(cls):
    """A sampler must pass every argument its point function needs, and no other.

    ``geo_statics`` is a hand-written tuple; nothing but this stops it
    drifting from the Smart-G function it feeds.
    """
    expected = set(inspect.signature(cls.point_fn).parameters)
    declared = swept_names(cls.contract) | ATMOSPHERIC_STATICS | set(cls.geo_statics)

    assert declared == expected


@pytest.mark.parametrize("cls", ALL_SAMPLERS, ids=lambda c: c.__name__)
def test_the_statics_a_sampler_builds_match_its_declaration(cls, configs):
    """What ``_statics`` returns must be what the declaration promised."""
    sampler = build(cls, configs)

    assert set(sampler._statics()) == ATMOSPHERIC_STATICS | set(cls.geo_statics)


@pytest.mark.parametrize("cls", ALL_SAMPLERS, ids=lambda c: c.__name__)
def test_a_sampler_produces_the_variable_its_contract_names(cls):
    """The output clause of the contract and ``_output_vars`` must agree."""
    produced = re.search(r"->\s*(\w+)", cls.contract)

    assert produced is not None
    assert cls._output_vars == [produced.group(1)]
    assert cls._required_vars == []


@pytest.mark.parametrize(
    ("cls", "n_ph"),
    [
        (TdirDownSampler, int(1e2)),
        (TdirUpSampler, int(1e9)),
        (TdifDownSampler, int(3e7)),
        (TdifUpSampler, int(3e7)),
        (RhoAtmSampler, int(2e7)),
        (SphAlbSampler, int(2e7)),
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_each_sampler_keeps_its_own_photon_budget(cls, n_ph, configs):
    """Photon counts differ by seven orders of magnitude across the six.

    A direct transmittance is an extinction along one line; a path
    reflectance is a scattering integral.  Sharing a base class must not
    flatten that.
    """
    assert build(cls, configs).n_ph == n_ph
    assert build(cls, configs, n_ph=1234)._statics()["n_ph"] == 1234


def test_the_spherical_albedo_needs_no_geometry(configs):
    """It is the one quantity of the six that has no direction in it."""
    sampler = build(SphAlbSampler, configs)

    assert sampler.geo_config is None
    assert [type(c).__name__ for c in sampler._get_configs()] == [
        "SpectralConfig",
        "AtmoConfig",
    ]


@pytest.mark.parametrize("cls", WITH_GEOMETRY, ids=lambda c: c.__name__)
def test_a_geometric_sampler_reads_its_geometry(cls, configs):
    """The geometry must be part of the space its cache key is drawn from."""
    sampler = build(cls, configs)

    assert sampler.geo_config is not None
    assert "GeoConfig" in [type(c).__name__ for c in sampler._get_configs()]


def test_asking_for_a_geometry_value_without_a_geometry_says_so(configs):
    """The failure has to name the sampler and the value, not raise on None."""
    atmo_config, _, spectral_config = configs
    sampler = RhoAtmSampler(
        atmo_config=atmo_config,
        geo_config=None,
        spectral_config=spectral_config,
        remove_rayleigh=False,
    )

    with pytest.raises(ValueError, match="RhoAtmSampler needs 'saa'"):
        sampler._statics()
