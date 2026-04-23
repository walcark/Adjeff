"""Pure Smart-G Monte Carlo kernel functions for all radiative samplers.

Each function in this module is a self-contained physics kernel: it
receives only plain values and DataArrays, calls Smart-G, and returns
a DataArray.  No module state is accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import geoclide as gc  # type: ignore[import-untyped]
import numpy as np
import xarray as xr
from smartg.visualizegeo import Entity, Plane, Transformation
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import GeneralizedGaussianPSF, PSFGrid, SensorBand
from adjeff.utils import fft_convolve_2D

if TYPE_CHECKING:
    from smartg.smartg import Sensor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# rho_atm
# ---------------------------------------------------------------------------


def rho_atm(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    vza: xr.DataArray,
    sza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
    saa: np.ndarray,
    vaa: np.ndarray,
    sat_height: float,
) -> xr.DataArray:
    """Compute the atmospheric reflectance (path radiance) with Smart-G.

    Parameters
    ----------
    wl : xr.DataArray
        Wavelengths [nm], 1-D.
    aot : xr.DataArray
        Aerosol optical thickness, 1-D.
    rh : xr.DataArray
        Relative humidity [%], 1-D.
    h : xr.DataArray
        Ground elevation [km], 1-D.
    href : xr.DataArray
        Reference height of the aerosol vertical profile [km], 1-D.
    vza : xr.DataArray
        Viewing zenith angles [°], 1-D.
    sza : xr.DataArray
        Solar zenith angles [°], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    n_ph : int
        Number of photons per Smart-G call.
    saa : np.ndarray
        Solar azimuth angle(s) [°].
    vaa : np.ndarray
        Viewing azimuth angle(s) [°].
    sat_height : float
        Satellite altitude [km].

    Returns
    -------
    xr.DataArray
        Atmospheric reflectance with dims ``(vza, sza, wl, ...)``.
    """
    from smartg.smartg import Smartg

    logger.info("Computing rho_atm ...", wl=wl, vza=vza, sza=sza)

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)
    sat_sensor = utils.make_sensors(
        180.0 - vza, float(vaa.flat[0]), posz=sat_height
    )
    sun_le = {
        "th_deg": np.atleast_1d(sza.values),
        "phi_deg": float(saa.flat[0]),
    }

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wl=atm.axes["wavelength"],
        atm=atm,
        sensor=sat_sensor,
        le=sun_le,
        NBPHOTONS=n_ph * atm_size * len(sat_sensor),
        NF=int(1e3),
    )["I_up (TOA)"].to_xarray()
    smartg.clear_context()

    res = utils.adapt_smartg_output(
        res,
        squeeze=["Azimuth angles"],
        rename={"sensor index": "vza", "Zenith angles": "sza"},
        coords={"vza": vza.values, "sza": sza.values},
        expand={
            "vza": vza.values,
            "sza": sza.values,
            "wavelength": atm.axes["wavelength"],
        },
    )

    res = res.transpose("vza", "sza", "wavelength")
    res = batch.unstack(
        xr.DataArray(
            res.values,
            dims=["vza", "sza", "index"],
            coords={
                "vza": vza.values,
                "sza": sza.values,
                "index": batch.index_coord,
            },
        )
    )
    logger.info("rho_atm successfully calculated.")
    return res


# ---------------------------------------------------------------------------
# tdir_down
# ---------------------------------------------------------------------------


def tdir_down(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    sza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int = int(1e2),
) -> xr.DataArray:
    """Compute the direct downward transmittance analytically.

    Parameters
    ----------
    wl : xr.DataArray
        Wavelengths [nm], 1-D.
    aot : xr.DataArray
        Aerosol optical thickness, 1-D.
    rh : xr.DataArray
        Relative humidity [%], 1-D.
    h : xr.DataArray
        Ground elevation [km], 1-D.
    href : xr.DataArray
        Reference height of the aerosol vertical profile [km], 1-D.
    sza : xr.DataArray
        Solar zenith angles [°], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh optical depth is set to zero.
    n_ph : int, optional
        Number of photons for the optical depth retrieval, by default 100.

    Returns
    -------
    xr.DataArray
        Direct downward transmittance with dims ``(sza, wl, ...)``.
    """
    logger.info("Computing tdir_down ...", wl=wl, sza=sza)

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    od = utils.compute_optical_depth(atm)
    od = batch.unstack(
        xr.DataArray(
            od.values,
            dims=["index"],
            coords={"index": batch.index_coord},
        )
    )
    logger.info("tdir_down successfully calculated.")
    return xr.DataArray(np.exp(-od / np.cos(np.deg2rad(sza))))


# ---------------------------------------------------------------------------
# tdir_up
# ---------------------------------------------------------------------------


def tdir_up(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    vza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int = int(1e2),
) -> xr.DataArray:
    """Compute the direct upward transmittance analytically.

    Parameters
    ----------
    wl : xr.DataArray
        Wavelengths [nm], 1-D.
    aot : xr.DataArray
        Aerosol optical thickness, 1-D.
    rh : xr.DataArray
        Relative humidity [%], 1-D.
    h : xr.DataArray
        Ground elevation [km], 1-D.
    href : xr.DataArray
        Reference height of the aerosol vertical profile [km], 1-D.
    vza : xr.DataArray
        Viewing zenith angles [°], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh optical depth is set to zero.
    n_ph : int, optional
        Number of photons for the optical depth retrieval, by default 100.

    Returns
    -------
    xr.DataArray
        Direct upward transmittance with dims ``(vza, wl, ...)``.
    """
    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    od = utils.compute_optical_depth(atm)
    od = batch.unstack(
        xr.DataArray(od, dims=["index"], coords={"index": batch.index_coord}),
    )
    logger.info("tdir_up successfully calculated.")
    return xr.DataArray(xr.apply_ufunc(np.exp, -od / np.cos(np.deg2rad(vza))))


# ---------------------------------------------------------------------------
# tdif_down
# ---------------------------------------------------------------------------


def tdif_down(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    sza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
    saa: np.ndarray,
    sat_height: float,
) -> xr.DataArray:
    """Compute the downward diffuse transmittance with Smart-G.

    Parameters
    ----------
    wl : xr.DataArray
        Wavelengths [nm], 1-D.
    aot : xr.DataArray
        Aerosol optical thickness, 1-D.
    rh : xr.DataArray
        Relative humidity [%], 1-D.
    h : xr.DataArray
        Ground elevation [km], 1-D.
    href : xr.DataArray
        Reference height of the aerosol vertical profile [km], 1-D.
    sza : xr.DataArray
        Solar zenith angles [°], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    n_ph : int
        Number of photons per Smart-G call.
    saa : np.ndarray
        Solar azimuth angle(s) [°].
    sat_height : float
        Satellite altitude [km].

    Returns
    -------
    xr.DataArray
        Downward diffuse transmittance with dims ``(sza, wl, ...)``.
    """
    from smartg.smartg import Smartg

    logger.info("Computing tdif_down ...", wl=wl, sza=sza)

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)
    sun_sensor = utils.make_sensors(
        180.0 - sza, float(saa.flat[0]), posz=sat_height
    )

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wl=atm.axes["wavelength"],
        atm=atm,
        sensor=sun_sensor,
        OUTPUT_LAYERS=3,
        flux="planar",
        NBPHOTONS=n_ph * atm_size * len(sun_sensor),
        NF=int(1e3),
    )["flux_down (0+)"].to_xarray()
    smartg.clear_context()

    res = utils.adapt_smartg_output(
        res,
        rename={"sensor index": "sza"},
        coords={"sza": sza.values},
        expand={"sza": sza.values, "wavelength": atm.axes["wavelength"]},
    )

    res = res.transpose("sza", "wavelength")
    res = batch.unstack(
        xr.DataArray(
            res.values,
            dims=["sza", "index"],
            coords={"sza": sza.values, "index": batch.index_coord},
        )
    )
    logger.info("tdif_down successfully calculated.")
    return res


# ---------------------------------------------------------------------------
# tdif_up
# ---------------------------------------------------------------------------


def tdif_up(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    vza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
    saa: np.ndarray,
) -> xr.DataArray:
    """Compute the upward diffuse transmittance with Smart-G.

    Parameters
    ----------
    wl : xr.DataArray
        Wavelengths [nm], 1-D.
    aot : xr.DataArray
        Aerosol optical thickness, 1-D.
    rh : xr.DataArray
        Relative humidity [%], 1-D.
    h : xr.DataArray
        Ground elevation [km], 1-D.
    href : xr.DataArray
        Reference height of the aerosol vertical profile [km], 1-D.
    vza : xr.DataArray
        Viewing zenith angles [°], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    n_ph : int
        Number of photons per Smart-G call.
    saa : np.ndarray
        Solar azimuth angle(s) [°].

    Returns
    -------
    xr.DataArray
        Upward diffuse transmittance with dims ``(vza, wl, ...)``.
    """
    from smartg.smartg import Sensor, Smartg

    logger.info("Computing tdif_up ...", wl=wl, vza=vza)

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)
    th_deg = np.atleast_1d(np.squeeze(vza.values))
    sat_le = {"th_deg": th_deg, "phi_deg": float(saa.flat[0])}

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wl=atm.axes["wavelength"],
        atm=atm,
        sensor=Sensor(POSZ=0.0, LOC="ATMOS", TYPE=1, FOV=90),
        le=sat_le,
        NBPHOTONS=n_ph * atm_size,
        NF=int(1e3),
    )["I_up (TOA)"].to_xarray()
    smartg.clear_context()
    res = utils.adapt_smartg_output(
        res,
        squeeze=["Azimuth angles"],
        rename={"Zenith angles": "vza"},
        coords={"vza": vza.values},
        expand={"vza": vza.values, "wavelength": atm.axes["wavelength"]},
    )

    res = res.transpose("vza", "wavelength")
    res = batch.unstack(
        xr.DataArray(
            res.values,
            dims=["vza", "index"],
            coords={"vza": vza.values, "index": batch.index_coord},
        )
    )
    logger.info("tdif_up successfully calculated.")
    return res


# ---------------------------------------------------------------------------
# sph_alb
# ---------------------------------------------------------------------------


def sph_alb(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
) -> xr.DataArray:
    """Compute the spherical albedo of the atmosphere with Smart-G.

    Parameters
    ----------
    wl : xr.DataArray
        Wavelengths [nm], 1-D.
    aot : xr.DataArray
        Aerosol optical thickness, 1-D.
    rh : xr.DataArray
        Relative humidity [%], 1-D.
    h : xr.DataArray
        Ground elevation [km], 1-D.
    href : xr.DataArray
        Reference height of the aerosol vertical profile [km], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    n_ph : int
        Number of photons per Smart-G call.

    Returns
    -------
    xr.DataArray
        Spherical albedo with dims ``(wl, ...)``.
    """
    from smartg.smartg import Sensor, Smartg

    logger.info("Computing sph_alb ...", wl=wl)

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)
    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wl=atm.axes["wavelength"],
        atm=atm,
        sensor=Sensor(POSZ=0.0, LOC="ATMOS", TYPE=1, FOV=90),
        OUTPUT_LAYERS=3,
        flux="planar",
        NBPHOTONS=n_ph * atm_size,
        NF=int(1e3),
    )["flux_down (0+)"].to_xarray()
    smartg.clear_context()

    res = utils.adapt_smartg_output(
        res, expand={"wavelength": atm.axes["wavelength"]}
    )
    res = batch.unstack(
        xr.DataArray(
            res.values,
            dims=["index"],
            coords={"index": batch.index_coord},
        )
    )
    logger.info("sph_alb successfully calculated.")
    return res


# ---------------------------------------------------------------------------
# rho_toa  (full 2-D, no symmetry assumption)
# ---------------------------------------------------------------------------


def rho_toa(
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    sza: float,
    vza: float,
    href: xr.DataArray,
    vaa: float,
    saa: float,
    rho_s: xr.Dataset,
    band: SensorBand,
    species: dict[str, float],
    sat_height: float,
    afgl_type: str,
    remove_rayleigh: bool,
    nx: int,
    ny: int,
    topleft_pix: tuple[int, int],
    n_ph: int,
    n_alb: int,
    rho_background: float | Literal["mean", "min", "zero"] = "mean",
) -> xr.DataArray:
    """Compute TOA reflectance from an arbitrary 2D surface reflectance map.

    Sensors are placed on an ``nx × ny`` sub-grid (row-major: y-outer,
    x-inner) starting at ``topleft_pix``.  After Smart-G the flat
    ``"sensor index"`` axis is reshaped to ``(y, x)`` and the result is
    reindexed to the full image grid with ``NaN`` for unsampled pixels.
    """
    from smartg.smartg import Smartg

    sun_le = {"th_deg": sza, "phi_deg": saa}

    if rho_s["rho_s"].adjeff.kind() != "arbitrary":
        raise ValueError(
            "RhoToaSampler requires an arbitrary rho_s surface "
            "(use gaussian_image_dict(..., analytical=False) or equivalent). "
            f"Got kind='{rho_s['rho_s'].adjeff.kind()}'."
        )

    factory = atmo.SurfaceFactory(rho_background=rho_background)
    surf = factory.surface(rho_s)
    env = factory.custom_environment(rho_s, n_alb)

    x_full = rho_s["rho_s"].coords["x"].values
    y_full = rho_s["rho_s"].coords["y"].values
    if topleft_pix[0] + nx > len(x_full):
        raise ValueError(
            f"topleft_pix[0] + nx must be <= {len(x_full)}, "
            f"got {topleft_pix[0] + nx}"
        )
    if topleft_pix[1] + ny > len(y_full):
        raise ValueError(
            f"topleft_pix[1] + ny must be <= {len(y_full)}, "
            f"got {topleft_pix[1] + ny}"
        )

    x_sample = x_full[topleft_pix[0] : topleft_pix[0] + nx]
    y_sample = y_full[topleft_pix[1] : topleft_pix[1] + ny]

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=xr.DataArray([band.wl_nm], dims=["wl"]),
        aot=aot,
        rh=rh,
        href=href,
        h=h,
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)
    sensors = _grid_sensors(x_sample, y_sample, vza, vaa, sat_height)
    n_sensors = nx * ny

    smartg = Smartg(autoinit=False)
    result: xr.DataArray = smartg.run(
        wl=atm.axes["wavelength"],
        atm=atm,
        surf=surf,
        env=env,
        sensor=sensors,
        le=sun_le,
        NBPHOTONS=n_ph * atm_size * n_sensors,
        NF=int(1e4),
    )["I_up (TOA)"].to_xarray()
    smartg.clear_context()

    result = utils.adapt_smartg_output(
        result,
        squeeze=["Azimuth angles", "Zenith angles"],
        rename={"sensor index": "sensor"},
        coords={"sensor": np.arange(n_sensors)},
        expand={
            "sensor": np.arange(n_sensors),
            "wavelength": atm.axes["wavelength"],
        },
    )

    result = utils.adapt_smartg_output(
        result,
        rename={"wavelength": "index"},
        coords={"index": batch.index_coord},
    )
    result = batch.unstack(result)

    # Reshape flat sensor dim (nx*ny) → (y, x)
    # Sensors were built row-major (y-outer, x-inner), so C-order reshape
    # maps sensor index i*nx + j to (y_sample[i], x_sample[j]).
    si = list(result.dims).index("sensor")
    new_shape = result.shape[:si] + (ny, nx) + result.shape[si + 1 :]
    new_dims = (
        list(result.dims[:si]) + ["y", "x"] + list(result.dims[si + 1 :])
    )
    extra_coords = {
        k: result.coords[k] for k in result.coords if k != "sensor"
    }
    result_2d = xr.DataArray(
        result.values.reshape(new_shape),
        dims=new_dims,
        coords={**extra_coords, "y": y_sample, "x": x_sample},
    )

    return result_2d.reindex(y=y_full, x=x_full).sel(wl=band.wl_nm)


def _grid_sensors(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    vza: float,
    vaa: float,
    sat_height: float,
) -> list[Sensor]:
    """Create a row-major 2D grid of Smart-G sensors.

    For a ground point at ``(gx, gy)``, the sensor is placed at altitude
    ``sat_height`` offset horizontally by ``sat_height * tan(vza)`` along
    the viewing azimuth direction, so that it looks straight down at
    ``(gx, gy)``.

    Sensors are ordered y-outer, x-inner (row-major), so sensor index
    ``i * len(x_vals) + j`` corresponds to ground point
    ``(x_vals[j], y_vals[i])``.

    Parameters
    ----------
    x_vals : np.ndarray
        x coordinates of ground sampling points [km].
    y_vals : np.ndarray
        y coordinates of ground sampling points [km].
    vza : float
        Viewing zenith angle [°].
    vaa : float
        Viewing azimuth angle [°].
    sat_height : float
        Satellite altitude [km].

    Returns
    -------
    list[Sensor]
        ``len(y_vals) * len(x_vals)`` Smart-G Sensor instances.
    """
    from smartg.smartg import Sensor

    dx = sat_height * np.tan(np.deg2rad(vza)) * np.cos(np.deg2rad(vaa))
    dy = sat_height * np.tan(np.deg2rad(vza)) * np.sin(np.deg2rad(vaa))

    return [
        Sensor(
            POSX=float(gx + dx),
            POSY=float(gy + dy),
            POSZ=sat_height,
            THDEG=180.0 - vza,
            PHDEG=(vaa + 180.0) % 360.0,
            LOC="ATMOS",
        )
        for gy in y_vals
        for gx in x_vals
    ]


# ---------------------------------------------------------------------------
# rho_toa_sym  (radial sampling under azimuthal symmetry assumption)
# ---------------------------------------------------------------------------


def rho_toa_sym(
    sza: float,
    vza: float,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    vaa: float,
    saa: float,
    rho_s: xr.Dataset,
    band: SensorBand,
    species: dict[str, float],
    sat_height: float,
    afgl_type: str,
    remove_rayleigh: bool,
    nr: int,
    n_ph: int,
) -> xr.DataArray:
    """Compute the TOA reflectance from the surface reflectance.

    This code assumes that the input field is symmetric.
    """
    from smartg.smartg import Smartg

    if rho_s["rho_s"].adjeff.kind() != "analytical":
        raise ValueError(
            "RhoToaSymSampler requires an analytical rho_s surface. "
            "Use RhoToaSampler for arbitrary fields. "
            f"Got kind='{rho_s['rho_s'].adjeff.kind()}'."
        )

    sun_le = {"th_deg": sza, "phi_deg": saa}
    factory = atmo.SurfaceFactory()
    surf = factory.surface(rho_s)
    env = factory.environment(rho_s)

    res: float = rho_s["rho_s"].adjeff.res
    n: int = rho_s["rho_s"].adjeff.n
    n = n - 1 if n % 2 == 0 else n
    approx_psf = GeneralizedGaussianPSF(
        band=band,
        grid=PSFGrid(res=res, n=n),
        sigma=0.00005,
        n=0.20,
    )
    rho_toa_approx = fft_convolve_2D(
        rho_s["rho_s"],
        approx_psf.to_dataarray(),
        padding="reflect",
        conv_type="same",
        device="cpu",
    )

    profile = rho_toa_approx.adjeff.radial()
    r_vals: xr.DataArray = profile.adjeff.radial("adaptive", n=nr, max_gap=0.1)

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=xr.DataArray([band.wl_nm], dims=["wl"]),
        aot=aot,
        rh=rh,
        href=href,
        h=h,
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)
    sensors = _radial_sensors(r_vals.coords["r"].data, vza, vaa, sat_height)

    smartg = Smartg(autoinit=False)
    result: xr.DataArray = smartg.run(
        wl=atm.axes["wavelength"],
        atm=atm,
        surf=surf,
        env=env,
        sensor=sensors,
        le=sun_le,
        NBPHOTONS=n_ph * atm_size * len(sensors),
        NF=int(1e4),
        RMIN=1,
    )["I_up (TOA)"].to_xarray()
    smartg.clear_context()

    result = utils.adapt_smartg_output(
        result,
        squeeze=["Azimuth angles", "Zenith angles"],
        rename={"sensor index": "r"},
        coords={"r": r_vals.coords["r"]},
        expand={
            "r": r_vals.coords["r"],
            "wavelength": atm.axes["wavelength"],
        },
    )

    result = utils.adapt_smartg_output(
        result,
        rename={"wavelength": "index"},
        coords={"index": batch.index_coord},
    )

    result = batch.unstack(result)

    # Add pre-computed rho_atm to avoid simulation noise
    result = result + rho_s["rho_atm"]

    # Reconstruct 2-D field from radial profile: `.compute()` materialises
    # dask chunks introduced by `+ rho_atm` above, because `to_field` uses
    # `apply_ufunc` without dask support.
    # TODO: add dask support to apply_ufunc with parallelize=True.
    return xr.DataArray(
        result.compute().adjeff.to_field(rho_s).sel(wl=band.wl_nm)
    )


def _radial_sensors(
    r_vals: np.ndarray,
    vza: float,
    vaa: float,
    sat_height: float,
) -> list[Sensor]:
    """Create position-specific sensors at radial distances from scene centre.

    Ground points are placed along the axis **perpendicular** to the viewing
    azimuth (vaa + 90°).  This axis is the symmetry plane of the atmospheric
    PSF: forward- and backward-scatter contributions are equal on both sides,
    so the sampled radial profile is representative of the azimuthal average
    even when VZA ≠ 0.  Sampling along vaa itself would bias the profile
    toward the elongated lobe of the PSF.

    For each ground point ``(gx, gy)`` on the perpendicular axis, the sensor
    is offset by ``sat_height * tan(vza)`` along the vaa direction so that
    it looks straight at ``(gx, gy)``.

    Parameters
    ----------
    r_vals : np.ndarray
        Radial distances of ground sampling points [km].
    vza : float
        Viewing zenith angle [°].
    vaa : float
        Viewing azimuth angle [°].
    sat_height : float
        Satellite altitude [km].

    Returns
    -------
    list[Sensor]
        One Smart-G Sensor per radial distance value.
    """
    from smartg.smartg import Sensor

    cos_vaa = np.cos(np.deg2rad(vaa))
    sin_vaa = np.sin(np.deg2rad(vaa))
    # Perpendicular to vaa: (cos(vaa+90°), sin(vaa+90°)) = (-sin_vaa, cos_vaa)
    cos_perp = -sin_vaa
    sin_perp = cos_vaa
    # Satellite offset to keep the viewing direction fixed at (vza, vaa)
    dx = sat_height * np.tan(np.deg2rad(vza)) * cos_vaa
    dy = sat_height * np.tan(np.deg2rad(vza)) * sin_vaa

    return [
        Sensor(
            POSX=float(r * cos_perp + dx),
            POSY=float(r * sin_perp + dy),
            POSZ=sat_height,
            THDEG=180.0 - vza,
            PHDEG=(vaa + 180.0) % 360.0,
            LOC="ATMOS",
        )
        for r in r_vals
    ]


# ---------------------------------------------------------------------------
# psf_atm
# ---------------------------------------------------------------------------


def psf_atm(
    vza: float,
    vaa: float,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    rho_s: xr.Dataset,
    band: SensorBand,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
) -> xr.DataArray:
    """Sample the atmospheric Point Spread Function."""
    from smartg.smartg import Smartg

    res: float = rho_s["rho_s"].adjeff.res
    n: int = rho_s["rho_s"].adjeff.n
    if n % 2 == 0:
        raise ValueError(
            f"Image grid size n must be odd (got {n}): a PSF kernel requires "
            "a well-defined centre pixel."
        )
    # SmartG computes cells per half-axis as floor(half_size / TC) then
    # doubles, so an odd n would yield n-1 cells. Use n+1 (even) for the
    # Entity and trim the extra edge row/col afterwards.
    half_size = res * (n + 1) / 2

    sampling_grid = Entity(
        name="receiver",
        TC=res,
        geo=Plane(
            p1=gc.Point(-half_size, -half_size, 0.0),
            p2=gc.Point(half_size, -half_size, 0.0),
            p3=gc.Point(-half_size, half_size, 0.0),
            p4=gc.Point(half_size, half_size, 0.0),
        ),
        transformation=Transformation(
            rotation=np.array([0.0, 0.0, 0.0]),
            translation=np.array([1e-5, 1e-5, 1e-5]),
        ),
    )

    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=xr.DataArray([band.wl_nm], dims=["wl"]),
        aot=aot,
        rh=rh,
        href=href,
        h=h,
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    atm_size = len(batch.index_coord)

    smartg = Smartg(obj3D=True, autoinit=False)
    result = smartg.run(
        wl=band.wl_nm,
        atm=atm,
        THVDEG=float(vza),
        PHVDEG=180.0 - float(vaa),
        myObjects=[sampling_grid],
        NBPHOTONS=n_ph * atm_size,
        NF=1e4,
    ).to_xarray()
    smartg.clear_context()

    result = utils.adapt_smartg_output(
        result["C_Receiver"].isel(Categories=0),
        rename={"X_Cell_Index": "x", "Y_Cell_Index": "y"},
        squeeze=["Categories"],
    )

    result = result.isel(x=slice(None, n), y=slice(None, n))

    # Assign proper spatial km coordinates from the input grid, and strip
    # the DataArray name so bundle.apply can combine results correctly.
    result = (
        result.assign_coords(
            x=rho_s["rho_s"].coords["x"].values,
            y=rho_s["rho_s"].coords["y"].values,
        )
        / result.sum()
    )
    return result.rename(None)
