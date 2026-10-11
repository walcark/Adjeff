"""Configuration objects, from scalars or from a scene.

Classes
-------
    FullConfig
        See :mod:`adjeff.api`.
    make_full_config
        See :mod:`adjeff.api`.
    load_config
        See :mod:`adjeff.api`.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import (
    ImageDict,  # noqa: E402
    SensorBand,
)
from adjeff.exceptions import MissingVariableError

_Scalar = float | list[float] | xr.DataArray


def _da(val: _Scalar, dim: str) -> xr.DataArray:
    """Coerce a scalar/list/DataArray to a named-dim DataArray."""
    if isinstance(val, xr.DataArray):
        # A reduction such as the `.mean()` of `load_config(aggregate=True)`
        # returns a 0-d array.  The samplers allocate their output from the
        # named dimension, so it has to carry one.
        return val if val.ndim else val.expand_dims(dim)
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    return xr.DataArray(arr, dims=[dim])


def _make_atmo_config(
    aot: _Scalar = 0.1,
    rh: _Scalar = 50.0,
    h: _Scalar = 0.0,
    href: _Scalar = 2.0,
    species: dict[str, float] | None = None,
) -> AtmoConfig:
    """Return an :class:`~adjeff.atmosphere.AtmoConfig`, with defaults."""
    if species is None:
        species = {"sulphate": 1.0}
    return AtmoConfig(
        aot=_da(aot, "aot"),
        rh=_da(rh, "rh"),
        h=_da(h, "h"),
        href=_da(href, "href"),
        species=species,
    )


def _make_geo_config(
    sza: _Scalar = 30.0,
    vza: _Scalar = 0.0,
    saa: _Scalar = 120.0,
    vaa: _Scalar = 120.0,
    sat_height: float = 786.0,
) -> GeoConfig:
    """Return a :class:`~adjeff.atmosphere.GeoConfig`, with defaults."""
    return GeoConfig(
        sza=_da(sza, "sza"),
        vza=_da(vza, "vza"),
        saa=_da(saa, "saa"),
        vaa=_da(vaa, "vaa"),
        sat_height=sat_height,
    )


class FullConfig(TypedDict):
    """The three configs, to unpack as ``**cfg`` into the samplers."""

    atmo_config: AtmoConfig
    geo_config: GeoConfig
    spectral_config: SpectralConfig


def make_full_config(
    bands: list[SensorBand],
    aot: _Scalar = 0.1,
    rh: _Scalar = 50.0,
    h: _Scalar = 0.0,
    href: _Scalar = 2.0,
    species: dict[str, float] | None = None,
    sza: _Scalar = 30.0,
    vza: _Scalar = 0.0,
    saa: _Scalar = 120.0,
    vaa: _Scalar = 120.0,
    sat_height: float = 786.0,
) -> FullConfig:
    """Return the atmosphere, geometry and spectral configs of *bands*.

    Parameters are those of the module docstring; *sat_height* is the
    satellite altitude [km].
    """
    return FullConfig(
        atmo_config=_make_atmo_config(aot=aot, rh=rh, h=h, href=href, species=species),
        geo_config=_make_geo_config(
            sza=sza, vza=vza, saa=saa, vaa=vaa, sat_height=sat_height
        ),
        spectral_config=SpectralConfig.from_bands(bands),
    )


_SPECIES_ATTR = "adjeff:species"

_DEFAULT_SPECIES: dict[str, float] = {"sulphate": 1.0}

_REQUIRED_VARS = ["aot", "h", "rh", "href", "vza", "vaa", "sza", "saa"]


def _to_scalar(v: xr.DataArray | float) -> float:
    """Extract a Python float from a scalar, 0-d, or 1-element DataArray."""
    if isinstance(v, xr.DataArray):
        return float(v.values.flat[0])
    return float(v)


def _res_from_scene(scene: ImageDict, band: SensorBand) -> float:
    """Return the pixel size [km] of *scene[band]*, from its ``y`` spacing."""
    y: xr.DataArray = scene[band].coords["y"]
    return float(abs(float(y[1]) - float(y[0])))


def load_config(
    scene: ImageDict,
    band: SensorBand,
    *,
    aggregate: bool = False,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
) -> FullConfig:
    """Return the configs of the atmosphere and geometry held by ``scene[band]``.

    Parameters
    ----------
    aggregate : bool, optional
        Reduce every field to its mean; exclusive with *n_bins*.
    species : dict[str, float] or None, optional
        Species; else those stored by :func:`load_scene`, else sulphate.

    The other parameters are those of the module docstring.

    Raises
    ------
    MissingVariableError
        If ``scene[band]`` lacks one of the fields.
    ConfigurationError
        If both *aggregate* and *n_bins* are given.
    """
    from adjeff.exceptions import ConfigurationError

    if aggregate and n_bins is not None:
        raise ConfigurationError("`aggregate` and `n_bins` are mutually exclusive.")

    ds = scene[band]
    missing = [v for v in _REQUIRED_VARS if v not in ds]
    if missing:
        raise MissingVariableError(
            f"Variables {missing!r} are missing from band {band!r}."
        )

    _species: dict[str, float] = (
        species or ds.attrs.get(_SPECIES_ATTR) or _DEFAULT_SPECIES
    )

    def _field(name: str) -> xr.DataArray:
        da: xr.DataArray = ds[name]
        if aggregate:
            return da.mean()
        if n_bins is not None and name in ("aot", "h"):
            return da.adjeff.digitize(n_bins=n_bins)  # type: ignore[no-any-return]
        return da

    return make_full_config(
        bands=scene.bands,
        aot=_field("aot"),
        h=_field("h"),
        rh=_field("rh"),
        href=_field("href"),
        vza=_field("vza"),
        vaa=_field("vaa"),
        sza=_field("sza"),
        saa=_field("saa"),
        species=_species,
    )
