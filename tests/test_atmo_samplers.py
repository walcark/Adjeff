"""The six radiative samplers must still say what they used to say.

They now share a base class that builds their static arguments from a
declaration.  A declaration can be wrong in a way a signature cannot, so
these tests read the physics functions and check the declarations against
them.
"""

import inspect
import re

import numpy as np
import pytest
import xarray as xr
from xsweep.policy import UNSET

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


class TestDeclarations:
    """What each sampler declares, against what it computes."""

    @pytest.mark.parametrize("cls", ALL_SAMPLERS, ids=lambda c: c.__name__)
    def test_the_declared_statics_are_exactly_what_the_physics_asks_for(self, cls):
        """A sampler must pass every argument its point function needs, and no other.

        ``geo_statics`` is a hand-written tuple; nothing but this stops it
        drifting from the Smart-G function it feeds.
        """
        expected = set(inspect.signature(cls.point_fn).parameters)
        declared = (
            swept_names(cls.contract) | ATMOSPHERIC_STATICS | set(cls.geo_statics)
        )

        assert declared == expected

    @pytest.mark.parametrize("cls", ALL_SAMPLERS, ids=lambda c: c.__name__)
    def test_the_statics_a_sampler_builds_match_its_declaration(self, cls, configs):
        """What ``_statics`` returns must be what the declaration promised."""
        sampler = build(cls, configs)

        assert set(sampler._statics()) == ATMOSPHERIC_STATICS | set(cls.geo_statics)

    @pytest.mark.parametrize("cls", ALL_SAMPLERS, ids=lambda c: c.__name__)
    def test_a_sampler_produces_the_variable_its_contract_names(self, cls):
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
    def test_each_sampler_keeps_its_own_photon_budget(self, cls, n_ph, configs):
        """Photon counts differ by seven orders of magnitude across the six.

        A direct transmittance is an extinction along one line; a path
        reflectance is a scattering integral.  Sharing a base class must not
        flatten that.
        """
        assert build(cls, configs).n_ph == n_ph
        assert build(cls, configs, n_ph=1234)._statics()["n_ph"] == 1234


class TestGeometryDependence:
    """Which samplers need a geometry, and which do not."""

    def test_the_spherical_albedo_needs_no_geometry(self, configs):
        """It is the one quantity of the six that has no direction in it."""
        sampler = build(SphAlbSampler, configs)

        assert sampler.geo_config is None
        assert [type(c).__name__ for c in sampler._get_configs()] == [
            "SpectralConfig",
            "AtmoConfig",
        ]

    @pytest.mark.parametrize("cls", WITH_GEOMETRY, ids=lambda c: c.__name__)
    def test_a_geometric_sampler_reads_its_geometry(self, cls, configs):
        """The geometry must be part of the space its cache key is drawn from."""
        sampler = build(cls, configs)

        assert sampler.geo_config is not None
        assert "GeoConfig" in [type(c).__name__ for c in sampler._get_configs()]

    def test_asking_for_a_geometry_value_without_a_geometry_says_so(self, configs):
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


class TestSweepPolicy:
    """Which sweeps are written through to the store."""

    def test_a_spatial_sweep_is_chunked_on_its_vectorised_dims(self, tmp_path):
        from adjeff.modules.samplers import RhoToaSymSampler
        from adjeff.utils import CacheStore

        sampler = RhoToaSymSampler(
            atmo_config=atmo.AtmoConfig(
                aot=xr.DataArray([0.1, 0.2], dims=["aot"]),
                rh=xr.DataArray([50.0], dims=["rh"]),
                h=xr.DataArray([0.0], dims=["h"]),
                href=xr.DataArray([2.0], dims=["href"]),
                species={"sulphate": 1.0},
            ),
            geo_config=atmo.GeoConfig(
                sza=xr.DataArray([40.0], dims=["sza"]),
                vza=xr.DataArray([0.0], dims=["vza"]),
                saa=xr.DataArray([0.0], dims=["saa"]),
                vaa=xr.DataArray([0.0], dims=["vaa"]),
            ),
            remove_rayleigh=False,
            cache=CacheStore(tmp_path),
        )
        policy = sampler._policy(sampler._space(), {})
        vec = {var.name for var in sampler._contract.vec}
        assert "RhoToaSymSampler" in str(policy.store)
        assert set(policy.chunks) <= vec
        assert policy.load is False

    def test_the_chunks_hold_a_batch_of_states(self, tmp_path):
        from adjeff.modules.samplers import RhoToaSymSampler
        from adjeff.utils import CacheStore

        sampler = RhoToaSymSampler(
            atmo_config=atmo.AtmoConfig(
                aot=xr.DataArray([0.0, 0.1, 0.2, 0.3], dims=["aot"]),
                rh=xr.DataArray([50.0, 90.0], dims=["rh"]),
                h=xr.DataArray([0.0, 3.0], dims=["h"]),
                href=xr.DataArray([2.0], dims=["href"]),
                species={"sulphate": 1.0},
            ),
            geo_config=atmo.GeoConfig(
                sza=xr.DataArray([40.0], dims=["sza"]),
                vza=xr.DataArray([0.0], dims=["vza"]),
                saa=xr.DataArray([0.0], dims=["saa"]),
                vaa=xr.DataArray([0.0], dims=["vaa"]),
            ),
            remove_rayleigh=False,
            batch_size=4,
            cache=CacheStore(tmp_path),
        )
        chunks = sampler._policy(sampler._space(), {}).chunks
        assert np.prod(list(chunks.values())) == 4

    def test_a_spectral_sweep_is_left_in_memory(self, tmp_path):
        from adjeff.modules.samplers import SphAlbSampler
        from adjeff.utils import CacheStore

        sampler = SphAlbSampler(
            atmo_config=atmo.AtmoConfig(
                aot=xr.DataArray([0.1], dims=["aot"]),
                rh=xr.DataArray([50.0], dims=["rh"]),
                h=xr.DataArray([0.0], dims=["h"]),
                href=xr.DataArray([2.0], dims=["href"]),
                species={"sulphate": 1.0},
            ),
            spectral_config=atmo.SpectralConfig.from_bands([S2Band.B02]),
            remove_rayleigh=False,
            cache=CacheStore(tmp_path),
        )
        assert sampler._policy(sampler._space(), {}).store is UNSET

    def test_two_species_do_not_share_a_store(self, tmp_path):
        from adjeff.modules.samplers import RhoToaSymSampler
        from adjeff.utils import CacheStore

        def name(species: str) -> str:
            sampler = RhoToaSymSampler(
                atmo_config=atmo.AtmoConfig(
                    aot=xr.DataArray([0.1, 0.2], dims=["aot"]),
                    rh=xr.DataArray([50.0], dims=["rh"]),
                    h=xr.DataArray([0.0], dims=["h"]),
                    href=xr.DataArray([2.0], dims=["href"]),
                    species={species: 1.0},
                ),
                geo_config=atmo.GeoConfig(
                    sza=xr.DataArray([40.0], dims=["sza"]),
                    vza=xr.DataArray([0.0], dims=["vza"]),
                    saa=xr.DataArray([0.0], dims=["saa"]),
                    vaa=xr.DataArray([0.0], dims=["vaa"]),
                ),
                remove_rayleigh=False,
                cache=CacheStore(tmp_path),
            )
            return str(sampler._policy(sampler._space(), {}).store)

        assert name("ammonium") != name("blackcar")

    def test_two_scenes_do_not_share_a_store(self, tmp_path):
        from adjeff.core import S2Band, gaussian_image_dict
        from adjeff.modules.samplers import RhoToaSymSampler
        from adjeff.utils import CacheStore

        sampler = RhoToaSymSampler(
            atmo_config=atmo.AtmoConfig(
                aot=xr.DataArray([0.1, 0.2], dims=["aot"]),
                rh=xr.DataArray([50.0], dims=["rh"]),
                h=xr.DataArray([0.0], dims=["h"]),
                href=xr.DataArray([2.0], dims=["href"]),
                species={"sulphate": 1.0},
            ),
            geo_config=atmo.GeoConfig(
                sza=xr.DataArray([40.0], dims=["sza"]),
                vza=xr.DataArray([0.0], dims=["vza"]),
                saa=xr.DataArray([0.0], dims=["saa"]),
                vaa=xr.DataArray([0.0], dims=["vaa"]),
            ),
            remove_rayleigh=False,
            cache=CacheStore(tmp_path),
        )
        band = S2Band.B02
        scenes = [
            gaussian_image_dict(sigma=s, res_km=0.5, bands=[band], n=9)[band]
            for s in (1.0, 5.0)
        ]
        space = sampler._space()
        stores = [
            str(sampler._policy(space, {"rho_s": ds, "band": band}).store)
            for ds in scenes
        ]
        assert stores[0] != stores[1]

    @pytest.mark.integration
    def test_a_mapped_sweep_keeps_the_scene_coordinates(self):
        """Every pixel gets a value, not the few whose index matches a label."""
        import numpy as np

        from adjeff.api import run_radiatives_from_scene
        from adjeff.core import gaussian_image_dict

        band, size = S2Band.B02, 5
        scene = gaussian_image_dict(sigma=1.0, res_km=0.5, bands=[band], n=size)
        ds = scene[band]
        rng = np.random.default_rng(0)
        ds["aot"] = (("y", "x"), rng.uniform(0.05, 0.4, (size, size)))
        ds["h"] = (("y", "x"), np.zeros((size, size)))
        for name, value in (
            ("rh", 50.0),
            ("href", 2.0),
            ("sza", 17.0),
            ("saa", 117.0),
            ("vza", 3.2),
            ("vaa", 207.0),
        ):
            ds[name] = xr.DataArray([value], dims=[name])
        scene[band] = ds

        pytest.importorskip("smartg")
        out = run_radiatives_from_scene(scene, n_bins=2)
        assert bool(np.isfinite(out[band]["tdir_down"]).all())

    def test_a_stray_scalar_coordinate_does_not_overwrite_a_real_axis(self):
        """An array carrying a stray same-named scalar coord keeps the real one."""
        from adjeff.modules.samplers import TdirDownSampler

        spectral = atmo.SpectralConfig.from_bands([S2Band.B8A])
        geo = atmo.GeoConfig(
            sza=xr.DataArray([17.0], dims=["sza"]),
            vza=xr.DataArray([3.0], dims=["vza"]),
            saa=xr.DataArray([117.0], dims=["saa"]),
            vaa=xr.DataArray([207.0], dims=["vaa"]),
        )
        # aot carries a stray scalar 'wl' coordinate, as if inherited from
        # a Dataset that already held per-wl radiative terms.
        aot = xr.DataArray([0.2], dims=["aot"]).assign_coords(wl=865.0)
        sampler = TdirDownSampler(
            atmo_config=atmo.AtmoConfig(
                aot=aot,
                rh=xr.DataArray([50.0], dims=["rh"]),
                h=xr.DataArray([0.1], dims=["h"]),
                href=xr.DataArray([2.0], dims=["href"]),
                species={"sulphate": 1.0},
            ),
            geo_config=geo,
            spectral_config=spectral,
            remove_rayleigh=False,
        )
        sampler._space()
        wl = sampler._dropped["wl"]
        assert wl.dims == ("wl",)
        assert float(wl.values[0]) == 865.0
