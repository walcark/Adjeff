"""Functions to generate instances of ImageDict.

The three generators below differ only in the field they evaluate and in
the provenance they stamp on it.  Everything around that, resolving a
pixel count per band, laying a square grid, wrapping the values into a
Dataset, is shared here rather than copied once per generator.
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
    """Resolve per-band pixel counts from either ``n`` or ``extent_km``.

    Parameters
    ----------
    bands:
        Bands for which a pixel count is needed.
    res_km : float | dict
        Resolution per band - either a scalar applied to all bands or a
        per-band mapping.
    n : int | dict
        Number of pixels along one dimension — either a scalar applied to
        all bands or a per-band mapping.
    extent_km : float | dict
        Physical extent of the image [km] — either a scalar applied to all
        bands or a per-band mapping. ``n`` is derived per band as
        ``round(extent_km / res_km)``.

    Returns
    -------
    dict[SensorBand, int]
        Mapping from each band to its pixel count.

    Raises
    ------
    ValueError
        If both or neither of ``n`` and ``extent_km`` are provided.

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
    """Create an ImageDict with a Gaussian spatial pattern.

    The spatial grid resolution is defined from the band resolution (e.g.
    10 m, 20 m for Sentinel-2 bands). The generated field follows a 2D
    isotropic Gaussian centered at (0, 0):

    rho(x, y) = rho_min + (rho_max - rho_min) * exp(-(x^2 + y^2) / sigma^2)

    Parameters
    ----------
    sigma : float
        Standard deviation [km].
    res_km : float | dict[SensorBand, float]
        Pixel resolution [km]. Scalar or per-band mapping.
    rho_min : float, optional
        Minimum reflectance value, by default 0.0.
    rho_max : float, optional
        Maximum reflectance value, by default 1.0.
    bands : list of SensorBand, optional
        List of spectral bands to generate, by default [S2Band.B02].
    var : str, optional
        Name of the variable stored in the Dataset, by default "rho_s".
    extent_km : float | dict[SensorBand, float] | None
        Physical extent of the image [km]. Scalar or per-band mapping.
        Mutually exclusive with ``n``.
    n : int | dict[SensorBand, int] | None
        Number of pixels along one dimension. Scalar or per-band mapping.
        Mutually exclusive with ``extent_km``.
    analytical : bool
        Whether to register this field as analytical or not, default to True.

    Returns
    -------
    ImageDict
        Dictionary mapping each band to its corresponding Dataset.
        The Gaussian is centered at (0, 0) and radially symmetric.

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
    """Create an ImageDict with a disk-shaped spatial pattern.

    The spatial grid resolution is defined from the band resolution (e.g.
    10 m, 20 m for Sentinel-2 bands). The generated field is a binary disk
    centered at (0, 0):

    rho(x, y) = rho_max  if sqrt(x^2 + y^2) <= radius
                rho_min  otherwise

    Parameters
    ----------
    radius : float
        Radius of the disk [km].
    res_km : float | dict[SensorBand, float]
        Pixel resolution [km]. Scalar or per-band mapping.
    rho_min : float, optional
        Background reflectance, by default 0.0.
    rho_max : float, optional
        Reflectance value inside the disk, by default 1.0.
    bands : list of SensorBand, optional
        List of spectral bands to generate, by default [S2Band.B02].
    var : str, optional
        Name of the variable stored in the Dataset, by default "rho_s".
    extent_km : float | dict[SensorBand, float] | None
        Physical extent of the image [km]. Scalar or per-band mapping.
        Mutually exclusive with ``n``.
    n : int | dict[SensorBand, int] | None
        Number of pixels along one dimension. Scalar or per-band mapping.
        Mutually exclusive with ``extent_km``.
    analytical : bool
        Whether to register this field as analytical or not, default to True.

    Returns
    -------
    ImageDict
        Dictionary mapping each band to its corresponding Dataset.
        The disk is centered at (0, 0) and has a sharp boundary.

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
    """Create an ImageDict filled with uniform random float32 data.

    Each band gets a Dataset whose DataArrays have dims ``["y", "x"]``
    and shape ``(H, W)``.  All *variables* are created for every band.

    Parameters
    ----------
    bands : list of SensorBand
        List of spectral bands to generate.
    variables : list[str]
        Names of the variables stored in each Dataset.
    res_km : float | dict[SensorBand, float]
        Pixel resolution [km]. Scalar or per-band mapping.
    seed : int | None
        Optional RNG seed for reproducible data.  Required for cache
        hits across separate runs — without a fixed seed the input hash
        changes every time, guaranteeing a cache miss.
    extent_km : float | dict[SensorBand, float] | None
        Physical extent of the image [km]. Scalar or per-band mapping.
        Mutually exclusive with ``n``.
    n : int | dict[SensorBand, int] | None
        Number of pixels along one dimension. Scalar or per-band mapping.
        Mutually exclusive with ``extent_km``.

    Returns
    -------
    ImageDict
        Dictionary mapping each band to its corresponding Dataset,
        filled with uniform random float32 values in ``[0, 1)``.

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
