"""Methods to generation instances of ImageDict."""

import numpy as np
import structlog
import xarray as xr

from adjeff.utils import square_grid

from .bands import S2Band, SensorBand
from .image_dict import ImageDict

logger = structlog.getLogger(__name__)


def _resolve_n(
    bands: list[SensorBand],
    n: int | dict[SensorBand, int] | None,
    extent_km: float | dict[SensorBand, float] | None,
) -> dict[SensorBand, int]:
    """Resolve per-band pixel counts from either ``n`` or ``extent_km``.

    Parameters
    ----------
    bands:
        Bands for which a pixel count is needed.
    n:
        Number of pixels along one dimension — either a scalar applied to
        all bands or a per-band mapping.
    extent_km:
        Physical extent of the image [km] — either a scalar applied to all
        bands or a per-band mapping. ``n`` is derived per band as
        ``round(extent_km / band.res_km)``.

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
        raise ValueError("Provide exactly one of `n` or `extent_km`.")
    if n is not None and extent_km is not None:
        raise ValueError("`n` and `extent_km` are mutually exclusive.")

    if extent_km is not None:
        if isinstance(extent_km, dict):
            return {
                band: round(extent_km[band] / band.res_km) for band in bands
            }
        return {band: round(extent_km / band.res_km) for band in bands}

    assert n is not None
    if isinstance(n, dict):
        return {band: n[band] for band in bands}
    return {band: n for band in bands}


def gaussian_image_dict(
    sigma: float,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    bands: list[SensorBand] = [S2Band.B02],
    var: str = "rho_s",
    extent_km: float | dict[SensorBand, float] | None = None,
    n: int | dict[SensorBand, int] | None = None,
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

    Returns
    -------
    ImageDict
        Dictionary mapping each band to its corresponding Dataset.

    Notes
    -----
    The Gaussian is centered at (0, 0) and radially symmetric.

    """
    logger.debug("Creating Gaussian ImageDict.", bands=bands)

    band_n = _resolve_n(bands, n, extent_km)
    band_datasets: dict[SensorBand, xr.Dataset] = {}

    for band in bands:
        coords: xr.Coordinates = square_grid(band_n[band], band.res_km)

        data: np.ndarray = rho_min + (rho_max - rho_min) * np.exp(
            -(coords["x"] ** 2 + coords["y"] ** 2) / sigma**2
        )

        attrs = {
            "adjeff:kind": "analytical",
            "adjeff.model": "gaussian",
            "adjeff.params": {
                "sigma": sigma,
                "rho_min": rho_min,
                "rho_max": rho_max,
            },
        }

        data_vars = {
            var: xr.DataArray(
                np.asarray(data, dtype=np.float32),
                dims=["y", "x"],
                coords=coords,
                attrs=attrs,
            )
        }

        band_datasets[band] = xr.Dataset(data_vars)

        logger.debug(
            "Created Gaussian Image.",
            band=band,
            var=var,
            sigma=sigma,
            rho_min=rho_min,
            rho_max=rho_max,
            n=band_n[band],
        )

    return ImageDict(band_datasets)


def disk_image_dict(
    radius: float,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    bands: list[SensorBand] = [S2Band.B02],
    var: str = "rho_s",
    extent_km: float | dict[SensorBand, float] | None = None,
    n: int | dict[SensorBand, int] | None = None,
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

    Returns
    -------
    ImageDict
        Dictionary mapping each band to its corresponding Dataset.

    Notes
    -----
    The disk is centered at (0, 0) and has a sharp boundary.

    """
    logger.debug("Creating Disk ImageDict.", bands=bands)

    band_n = _resolve_n(bands, n, extent_km)
    band_datasets: dict[SensorBand, xr.Dataset] = {}

    for band in bands:
        coords: xr.Coordinates = square_grid(band_n[band], band.res_km)

        r2 = coords["x"] ** 2 + coords["y"] ** 2
        mask = r2 <= radius**2

        data: np.ndarray = np.where(mask, rho_max, rho_min)

        attrs = {
            "adjeff:kind": "analytical",
            "adjeff.model": "disk",
            "adjeff.params": {
                "radius": radius,
                "rho_min": rho_min,
                "rho_max": rho_max,
            },
        }

        data_vars = {
            var: xr.DataArray(
                np.asarray(data, dtype=np.float32),
                dims=["y", "x"],
                coords=coords,
                attrs=attrs,
            )
        }

        band_datasets[band] = xr.Dataset(data_vars)

        logger.debug(
            "Created Disk Image.",
            band=band,
            var=var,
            radius=radius,
            rho_min=rho_min,
            rho_max=rho_max,
            n=band_n[band],
        )

    return ImageDict(band_datasets)


def random_image_dict(
    bands: list[SensorBand],
    variables: list[str],
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

    """
    band_n = _resolve_n(bands, n, extent_km)
    rng = np.random.default_rng(seed)
    logger.debug(
        "Creating random ImageDict",
        bands=bands,
        variables=variables,
        seed=seed,
    )
    band_datasets: dict[SensorBand, xr.Dataset] = {}
    for band in bands:
        bn = band_n[band]
        coords: xr.Coordinates = square_grid(bn, band.res_km)

        data_vars = {
            v: xr.DataArray(
                rng.random((bn, bn), dtype=np.float32),
                dims=["y", "x"],
                coords=coords,
                attrs={"adjeff:kind": "arbitrary"},
            )
            for v in variables
        }
        band_datasets[band] = xr.Dataset(data_vars)
    return ImageDict(band_datasets)
