"""Synthetic scenes: Gaussian, disk or random reflectance fields.

Functions
---------
    gaussian_image_dict
        Isotropic Gaussian centred on the origin.
    disk_image_dict
        Uniform disc centred on the origin.
    random_image_dict
        Uniform random values in ``[0, 1)``.
"""

from collections.abc import Callable

import numpy as np
import xarray as xr

from adjeff.exceptions import ConfigurationError
from adjeff.utils.xrutils import square_grid

from .._logging import get_logger
from .bands import S2Band, SensorBand
from .image_dict import ImageDict

logger = get_logger(__name__)


def _resolve_n(
    bands: list[SensorBand],
    res_km: float | dict[SensorBand, float],
    n: int | dict[SensorBand, int] | None,
    extent_km: float | dict[SensorBand, float] | None,
) -> dict[SensorBand, int]:
    """Return the pixel count of each band, from *n* or ``extent_km / res_km``.

    Raises
    ------
    ConfigurationError
        Unless exactly one of *n* and *extent_km* is given.
    """
    if n is None and extent_km is None:
        raise ConfigurationError("Provide exactly one of `n` or `extent_km`.")
    if n is not None and extent_km is not None:
        raise ConfigurationError("`n` and `extent_km` are mutually exclusive.")

    if extent_km is not None:
        _res = res_km if isinstance(res_km, dict) else {b: res_km for b in bands}
        if isinstance(extent_km, dict):
            return {band: round(extent_km[band] / _res[band]) for band in bands}
        return {band: round(extent_km / _res[band]) for band in bands}

    assert n is not None
    if isinstance(n, dict):
        return {band: n[band] for band in bands}
    return {band: n for band in bands}


def _band_grids(
    bands: list[SensorBand],
    res_km: float | dict[SensorBand, float],
    n: int | dict[SensorBand, int] | None,
    extent_km: float | dict[SensorBand, float] | None,
) -> dict[SensorBand, xr.Coordinates]:
    """Return the square grid each band is sampled on, centred on (0, 0).

    Takes the same *res_km*, *n* and *extent_km* the public generators
    document; *n* and *extent_km* stay mutually exclusive.
    """
    per_band = res_km if isinstance(res_km, dict) else {b: res_km for b in bands}
    counts = _resolve_n(bands, res_km, n, extent_km)
    return {band: square_grid(counts[band], per_band[band]) for band in bands}


def _field_attrs(
    model: str, params: dict[str, float], analytical: bool
) -> dict[str, object]:
    """Return the provenance a generated field carries.

    An analytical field records the *model* that drew it and the
    *params* it was drawn with, so that a sampler can redraw it at
    another resolution.  Anything else is opaque and says so.
    """
    if not analytical:
        return {"adjeff:kind": "arbitrary"}
    return {
        "adjeff:kind": "analytical",
        "adjeff:model": model,
        "adjeff:params": dict(params),
    }


def _analytical_image_dict(
    data_fn: Callable[[xr.Coordinates], np.ndarray],
    model: str,
    params: dict[str, float],
    *,
    bands: list[SensorBand],
    res_km: float | dict[SensorBand, float],
    var: str,
    n: int | dict[SensorBand, int] | None,
    extent_km: float | dict[SensorBand, float] | None,
    analytical: bool,
) -> ImageDict:
    """Evaluate *data_fn* on every band's grid and stamp its provenance.

    *data_fn* takes one band's coordinates and returns the field on
    them; *model* and *params* describe it well enough to redraw it.
    The remaining arguments are the ones the public generators document.
    """
    logger.debug("scene.generate", model=model, bands=len(bands))

    band_datasets: dict[SensorBand, xr.Dataset] = {}
    for band, coords in _band_grids(bands, res_km, n, extent_km).items():
        values = xr.DataArray(
            np.asarray(data_fn(coords), dtype=np.float32),
            dims=["y", "x"],
            coords=coords,
            attrs=_field_attrs(model, params, analytical),
        )
        band_datasets[band] = xr.Dataset({var: values})
        logger.debug(
            "scene.generate_band",
            band=str(band),
            var=var,
            model=model,
            n=coords["x"].size,
            **params,
        )
    return ImageDict(band_datasets)


def _gaussian_data(
    coords: xr.Coordinates,
    sigma: float,
    rho_min: float,
    rho_max: float,
) -> np.ndarray:
    return rho_min + (rho_max - rho_min) * np.exp(
        -(coords["x"] ** 2 + coords["y"] ** 2) / (2 * sigma**2)
    )


def _disk_data(
    coords: xr.Coordinates,
    radius: float,
    rho_min: float,
    rho_max: float,
) -> np.ndarray:
    r2 = coords["x"] ** 2 + coords["y"] ** 2
    return np.where(r2 <= radius**2, rho_max, rho_min)


def gaussian_image_dict(
    sigma: float,
    res_km: float | dict[SensorBand, float],
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    bands: list[SensorBand] = [S2Band.B02],
    var: str = "rho_s",
    extent_km: float | dict[SensorBand, float] | None = None,
    n: int | dict[SensorBand, int] | None = None,
    analytical: bool = True,
) -> ImageDict:
    """Return a scene holding an isotropic Gaussian centred on the origin.

    ``rho = rho_min + (rho_max - rho_min) * exp(-(x² + y²) / (2 sigma²))``

    Parameters
    ----------
    sigma : float
        Standard deviation [km].
    res_km : float or dict[SensorBand, float]
        Pixel size [km], for all bands or per band.
    rho_min, rho_max : float, optional
        Background and peak reflectance, 0 and 1 by default.
    bands : list[SensorBand], optional
        Bands to generate, ``[S2Band.B02]`` by default.
    var : str, optional
        Variable name, ``"rho_s"`` by default.
    extent_km, n : float or int, or per-band dict, optional
        Image side [km] or pixels per side; exactly one is required.
    analytical : bool, optional
        Record the model and parameters, so that samplers can redraw the
        field exactly.  True by default.
    """
    return _analytical_image_dict(
        lambda coords: _gaussian_data(coords, sigma, rho_min, rho_max),
        "gauss",
        {"sigma": sigma, "rho_min": rho_min, "rho_max": rho_max},
        bands=bands,
        res_km=res_km,
        var=var,
        n=n,
        extent_km=extent_km,
        analytical=analytical,
    )


def disk_image_dict(
    radius: float,
    res_km: float | dict[SensorBand, float],
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    bands: list[SensorBand] = [S2Band.B02],
    var: str = "rho_s",
    extent_km: float | dict[SensorBand, float] | None = None,
    n: int | dict[SensorBand, int] | None = None,
    analytical: bool = True,
) -> ImageDict:
    """Return a scene holding a uniform disk centred on the origin.

    ``rho = rho_max`` within *radius* [km], ``rho_min`` outside.  The
    other parameters are those of :func:`gaussian_image_dict`.
    """
    return _analytical_image_dict(
        lambda coords: _disk_data(coords, radius, rho_min, rho_max),
        "disk",
        {"radius": radius, "rho_min": rho_min, "rho_max": rho_max},
        bands=bands,
        res_km=res_km,
        var=var,
        n=n,
        extent_km=extent_km,
        analytical=analytical,
    )


def random_image_dict(
    bands: list[SensorBand],
    variables: list[str],
    res_km: float | dict[SensorBand, float],
    seed: int | None = None,
    extent_km: float | dict[SensorBand, float] | None = None,
    n: int | dict[SensorBand, int] | None = None,
) -> ImageDict:
    """Return a scene of uniform random float32 *variables* in ``[0, 1)``.

    Without a *seed*, every run hashes differently and misses the cache.
    The other parameters are those of :func:`gaussian_image_dict`.
    """
    rng = np.random.default_rng(seed)
    logger.debug(
        "scene.generate_random",
        bands=len(bands),
        variables=variables,
        seed=seed,
    )
    band_datasets: dict[SensorBand, xr.Dataset] = {}
    for band, coords in _band_grids(bands, res_km, n, extent_km).items():
        side = int(coords["x"].size)
        band_datasets[band] = xr.Dataset(
            {
                v: xr.DataArray(
                    rng.random((side, side), dtype=np.float32),
                    dims=["y", "x"],
                    coords=coords,
                    attrs={"adjeff:kind": "arbitrary"},
                )
                for v in variables
            }
        )
    return ImageDict(band_datasets)
