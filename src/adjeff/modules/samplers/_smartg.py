"""Pure Smart-G Monte Carlo kernel functions for all radiative samplers.

Each function in this module is a self-contained physics kernel: it
receives only plain values and DataArrays, calls Smart-G, and returns
a DataArray.  No module state is accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geoclide as gc  # type: ignore[import-untyped]
import numpy as np
import xarray as xr
from smartg.objects3d import Entity, Plane, Transformation

import adjeff.atmosphere as atmo
from adjeff.core import GeneralizedGaussianPSF, PSFGrid, SensorBand
from adjeff.exceptions import ConfigurationError
from adjeff.utils import fft_convolve_2D
from adjeff.utils.smartgutils import (
    adapt_smartg_output,
    collect_batched,
    compute_optical_depth,
    make_sensors,
    pair_angles_with_points,
)
from adjeff.utils.xrutils import ParamBatch

from ..._logging import get_logger

if TYPE_CHECKING:
    from smartg.sensor import Sensor

logger = get_logger(__name__)


#: Kept as a module-level alias: the helper now lives in
#: :mod:`adjeff.utils.smartgutils`, next to the output adapter it belongs with.
_pair_angles_with_points = pair_angles_with_points


def _make_atmosphere(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
) -> tuple[Any, ParamBatch, int]:
    """Build a batched Smart-G atmosphere from atmospheric DataArrays.

    Returns the atmosphere profile table, the :class:`~adjeff.ParamBatch`
    used to build it, and the number of atmospheric profiles
    (``atm_size``).
    """
    batch = ParamBatch.from_dataarrays(wl=wl, aot=aot, rh=rh, href=href, h=h)
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )
    return atm, batch, len(batch.index_coord)


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
    saa: float,
    vaa: float,
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
    saa : float
        Solar azimuth angle(s) [°].
    vaa : float
        Viewing azimuth angle(s) [°].
    sat_height : float
        Satellite altitude [km].

    Returns
    -------
    xr.DataArray
        Atmospheric reflectance with dims ``(vza, sza, wl, ...)``.
    """
    from smartg.smartg import LocalEstimate, Smartg

    atm, batch, atm_size = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    sat_sensor = make_sensors(180.0 - vza, (vaa + 180.0) % 360.0, posz=sat_height)
    sun_le = LocalEstimate(th_deg=np.atleast_1d(sza.values), phi_deg=saa)

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wavelength=atm["wavelength"],
        atmosphere=atm,
        sensor=sat_sensor,
        le=sun_le,
        n_photons=n_ph * atm_size * len(sat_sensor),
        n_icdf=int(1e3),
    )["I_up (TOA)"]
    smartg.clear_context()

    return collect_batched(
        res,
        batch,
        angles={
            "sensor index": ("vza", vza.values),
            "Zenith angles": ("sza", sza.values),
        },
        drop=["Azimuth angles"],
    )


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
    atm, batch, _ = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    od = compute_optical_depth(atm)
    od = batch.unstack(
        xr.DataArray(
            od.values,
            dims=["index"],
            coords={"index": batch.index_coord},
        )
    )
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
    atm, batch, _ = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    od = compute_optical_depth(atm)
    od = batch.unstack(
        xr.DataArray(od, dims=["index"], coords={"index": batch.index_coord}),
    )
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
    saa: float,
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
    saa : float
        Solar azimuth angle(s) [°].
    sat_height : float
        Satellite altitude [km].

    Returns
    -------
    xr.DataArray
        Downward diffuse transmittance with dims ``(sza, wl, ...)``.
    """
    from smartg.smartg import Smartg

    atm, batch, atm_size = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    sun_sensor = make_sensors(180.0 - sza, saa, posz=sat_height)

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wavelength=atm["wavelength"],
        atmosphere=atm,
        sensor=sun_sensor,
        output_layers=3,
        flux="planar",
        n_photons=n_ph * atm_size * len(sun_sensor),
        n_icdf=int(1e3),
    )["flux_down (0+)"]
    smartg.clear_context()

    return collect_batched(res, batch, angles={"sensor index": ("sza", sza.values)})


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
    saa: float,
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
    saa : float
        Solar azimuth angle(s) [°].

    Returns
    -------
    xr.DataArray
        Upward diffuse transmittance with dims ``(vza, wl, ...)``.
    """
    from smartg.sensor import Sensor
    from smartg.smartg import LocalEstimate, Smartg

    atm, batch, atm_size = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    th_deg = np.atleast_1d(np.squeeze(vza.values))
    sat_le = LocalEstimate(th_deg=th_deg, phi_deg=saa)

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wavelength=atm["wavelength"],
        atmosphere=atm,
        sensor=Sensor(
            pos_z=0.0,
            loc="ATMOS",
            sensor_type=1,
            fov=90,
            # Explicit: Smart-G 2.0 changed the default from 0
            # (zenith) to 180 (nadir), which silently turned this
            # upward flux collector downward.
            th_deg=0.0,
        ),
        le=sat_le,
        n_photons=n_ph * atm_size,
        n_icdf=int(1e3),
    )["I_up (TOA)"]
    smartg.clear_context()
    return collect_batched(
        res,
        batch,
        angles={"Zenith angles": ("vza", vza.values)},
        drop=["Azimuth angles"],
    )


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
    from smartg.sensor import Sensor
    from smartg.smartg import Smartg

    atm, batch, atm_size = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wavelength=atm["wavelength"],
        atmosphere=atm,
        sensor=Sensor(
            pos_z=0.0,
            loc="ATMOS",
            sensor_type=1,
            fov=90,
            # Explicit: Smart-G 2.0 changed the default from 0
            # (zenith) to 180 (nadir), which silently turned this
            # upward flux collector downward.
            th_deg=0.0,
        ),
        output_layers=3,
        flux="planar",
        n_photons=n_ph * atm_size,
        n_icdf=int(1e3),
    )["flux_down (0+)"]
    smartg.clear_context()

    return collect_batched(res, batch)


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
    from smartg.smartg import LocalEstimate, Smartg

    sun_le = LocalEstimate(th_deg=sza, phi_deg=saa)

    if rho_s["rho_s"].adjeff.kind() != "arbitrary":
        raise ConfigurationError(
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
        raise ConfigurationError(
            f"topleft_pix[0] + nx must be <= {len(x_full)}, got {topleft_pix[0] + nx}"
        )
    if topleft_pix[1] + ny > len(y_full):
        raise ConfigurationError(
            f"topleft_pix[1] + ny must be <= {len(y_full)}, got {topleft_pix[1] + ny}"
        )

    x_sample = x_full[topleft_pix[0] : topleft_pix[0] + nx]
    y_sample = y_full[topleft_pix[1] : topleft_pix[1] + ny]

    atm, batch, atm_size = _make_atmosphere(
        xr.DataArray([band.wl_nm], dims=["wl"]),
        aot,
        rh,
        h,
        href,
        species,
        afgl_type,
        remove_rayleigh,
    )
    sensors = _grid_sensors(x_sample, y_sample, vza, vaa, sat_height)
    n_sensors = nx * ny

    smartg = Smartg(autoinit=False)
    result: xr.DataArray = smartg.run(
        wavelength=atm["wavelength"],
        atmosphere=atm,
        surface=surf,
        environment=env,
        sensor=sensors,
        le=sun_le,
        n_photons=n_ph * atm_size * n_sensors,
        n_icdf=int(1e4),
    )["I_up (TOA)"]
    smartg.clear_context()

    result = adapt_smartg_output(
        result,
        squeeze=["Azimuth angles", "Zenith angles"],
        rename={"sensor index": "sensor"},
        coords={"sensor": np.arange(n_sensors)},
        expand={
            "sensor": np.arange(n_sensors),
            "wavelength": atm["wavelength"],
        },
    )

    result = adapt_smartg_output(
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
    new_dims = list(result.dims[:si]) + ["y", "x"] + list(result.dims[si + 1 :])
    extra_coords = {k: result.coords[k] for k in result.coords if k != "sensor"}
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
    from smartg.sensor import Sensor

    dx = sat_height * np.tan(np.deg2rad(vza)) * np.cos(np.deg2rad(vaa))
    dy = sat_height * np.tan(np.deg2rad(vza)) * np.sin(np.deg2rad(vaa))

    return [
        Sensor(
            pos_x=float(gx + dx),
            pos_y=float(gy + dy),
            pos_z=sat_height,
            th_deg=180.0 - vza,
            ph_deg=(vaa + 180.0) % 360.0,
            loc="ATMOS",
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
    from smartg.smartg import LocalEstimate, Smartg

    if rho_s["rho_s"].adjeff.kind() != "analytical":
        raise ConfigurationError(
            "RhoToaSymSampler requires an analytical rho_s surface. "
            "Use RhoToaSampler for arbitrary fields. "
            f"Got kind='{rho_s['rho_s'].adjeff.kind()}'."
        )

    sun_le = LocalEstimate(th_deg=sza, phi_deg=saa)
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

    atm, batch, atm_size = _make_atmosphere(
        xr.DataArray([band.wl_nm], dims=["wl"]),
        aot,
        rh,
        h,
        href,
        species,
        afgl_type,
        remove_rayleigh,
    )
    sensors = _radial_sensors(r_vals.coords["r"].data, vza, vaa, sat_height)

    smartg = Smartg(autoinit=False)
    result: xr.DataArray = smartg.run(
        wavelength=atm["wavelength"],
        atmosphere=atm,
        surface=surf,
        environment=env,
        sensor=sensors,
        le=sun_le,
        n_photons=n_ph * atm_size * len(sensors),
        n_icdf=int(1e4),
        r_min=1,
    )["I_up (TOA)"]
    smartg.clear_context()

    result = adapt_smartg_output(
        result,
        squeeze=["Azimuth angles", "Zenith angles"],
        rename={"sensor index": "r"},
        coords={"r": r_vals.coords["r"]},
        expand={
            "r": r_vals.coords["r"],
            "wavelength": atm["wavelength"],
        },
    )

    result = adapt_smartg_output(
        result,
        rename={"wavelength": "index"},
        coords={"index": batch.index_coord},
    )

    result = batch.unstack(result)

    # Add pre-computed rho_atm to avoid simulation noise. It carries the
    # sza/vza axes of the sweep that produced it, while this call is at one
    # geometry, so the matching entry is selected rather than broadcast in:
    # broadcasting would give the return two dims the contract never
    # declares, and the result would be placed against the wrong axes.
    rho_atm = rho_s["rho_atm"]
    for dim, value in (("sza", sza), ("vza", vza)):
        if dim in rho_atm.dims:
            rho_atm = rho_atm.sel({dim: value}, method="nearest", drop=True)
    result = result + rho_atm

    # Reconstruct 2-D field from radial profile: `.compute()` materialises
    # dask chunks introduced by `+ rho_atm` above, because `to_field` uses
    # `apply_ufunc` without dask support.
    # TODO: add dask support to apply_ufunc with parallelize=True.
    return xr.DataArray(result.compute().adjeff.to_field(rho_s).sel(wl=band.wl_nm))


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
    from smartg.sensor import Sensor

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
            pos_x=float(r * cos_perp + dx),
            pos_y=float(r * sin_perp + dy),
            pos_z=sat_height,
            th_deg=180.0 - vza,
            ph_deg=(vaa + 180.0) % 360.0,
            loc="ATMOS",
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
        raise ConfigurationError(
            f"Image grid size n must be odd (got {n}): a PSF kernel requires "
            "a well-defined centre pixel."
        )
    # SmartG computes cells per half-axis as floor(half_size / tc) then
    # doubles, so an odd n would yield n-1 cells. Use n+1 (even) for the
    # Entity and trim the extra edge row/col afterwards.
    half_size = res * (n + 1) / 2

    sampling_grid = Entity(
        name="receiver",
        tc=res,
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

    atm, batch, atm_size = _make_atmosphere(
        xr.DataArray([band.wl_nm], dims=["wl"]),
        aot,
        rh,
        h,
        href,
        species,
        afgl_type,
        remove_rayleigh,
    )

    smartg = Smartg(obj3d=True, autoinit=False)
    result = smartg.run(
        wavelength=band.wl_nm,
        atmosphere=atm,
        th_deg=float(vza),
        ph_deg=180.0 - float(vaa),
        my_objects=[sampling_grid],
        n_photons=n_ph * atm_size,
        n_icdf=1e4,
    )
    smartg.clear_context()

    result = adapt_smartg_output(
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


# ---------------------------------------------------------------------------
# Non-lambertian surface: tdif_up and sph_alb under an RTLS BRDF
# ---------------------------------------------------------------------------


def _rtls_surface(k0: float, k1p: float, k2p: float) -> Any:
    """Return a Smart-G Ross-Thick Li-Sparse surface.

    The kernel weights travel as plain floats rather than as a built
    surface: a Smart-G object in a sampler's signature would end up in
    the cache key, where nothing guarantees it hashes the same twice.

    Parameters
    ----------
    k0 : float
        Spectral albedo of the isotropic kernel.
    k1p : float
        Weight of the geometric kernel, relative to the isotropic one.
    k2p : float
        Weight of the volumetric kernel, relative to the isotropic one.
    """
    import warnings

    from smartg.albedo import AlbedoCst
    from smartg.surface import RTLSSurface

    # The k0/k1p/k2p keywords Smart-G 1.1 advertises raise
    # "'tuple' object does not support item assignment": they write into
    # the tuple default they were given.  The deprecated `kp` triple is
    # the only path that runs, so its warning is not the caller's to see.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return RTLSSurface(kp=(AlbedoCst(k0), AlbedoCst(k1p), AlbedoCst(k2p)))


def _viewing_sensors(
    vza: xr.DataArray, azimuth: float, sat_height: float
) -> list["Sensor"]:
    """Return one Smart-G source per viewing zenith angle.

    A satellite at azimuth ``vaa`` sits along ``vaa`` from the ground
    point it observes, so in forward mode its photons travel along
    ``vaa + 180``.  Declaring ``ph_deg= vaa`` instead mirrors the
    geometry through the principal plane, which reverses the trend of
    the path reflectance with the relative azimuth: 25 percent at
    ``raa = 0`` and nothing at ``raa = 90``, where both are the same
    scattering angle.  See
    ``test_the_path_reflectance_follows_the_scattering_angle``.

    This matches :func:`_grid_sensors`, and **not**
    :func:`~adjeff.utils.smartgutils.make_sensors` as ``rho_atm`` calls
    it.

    Parameters
    ----------
    vza : xr.DataArray
        Viewing zenith angles [deg].
    azimuth : float
        Viewing azimuth ``vaa`` [deg], before the 180 degree reversal.
    sat_height : float
        Satellite altitude [km].
    """
    from smartg.sensor import Sensor

    return [
        Sensor(
            pos_z=sat_height,
            th_deg=float(180.0 - th),
            ph_deg=float((azimuth + 180.0) % 360.0),
            loc="ATMOS",
        )
        for th in np.atleast_1d(np.squeeze(vza.values))
    ]


def tdif_up_brdf(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    sza: xr.DataArray,
    vza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
    saa: float,
    raa: float,
    sat_height: float,
    k0: float,
    k1p: float,
    k2p: float,
) -> xr.DataArray:
    """Sample the two-way surface-reflected radiance over an RTLS surface.

    Photons leave the satellite along ``vza``, reflect **once** on the
    surface and are collected toward the sun by local estimate.  By
    reciprocity this is the sun-to-surface-to-satellite path, so the
    return is::

        raw = rho_eff * T(vza) * T_up(sza)

    with ``rho_eff`` the RTLS reflectance for the pair.  Turning it into
    a transmittance needs the downward quantities, which live in the
    scene; :class:`~adjeff.modules.samplers.TdifUpBrdfSampler` does that
    division.  ``RMIN = RMAX = 1`` keeps exactly one surface
    interaction, which removes the surface-atmosphere coupling instead
    of correcting for it afterwards, and keeps the result linear in the
    kernel weights.

    Unlike :func:`tdif_up`, this depends on **both** angles: a BRDF
    breaks the reciprocity that let the Lambertian case collapse them
    into one.

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
        Solar zenith angles [deg], 1-D.
    vza : xr.DataArray
        Viewing zenith angles [deg], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    n_ph : int
        Number of photons per Smart-G call and per sensor.
    saa : float
        Solar azimuth angle [deg], setting the absolute frame.
    raa : float
        Relative azimuth ``vaa - saa`` [deg].  It is what the BRDF
        actually depends on, so it is named rather than derived.
    sat_height : float
        Satellite altitude [km].
    k0, k1p, k2p : float
        RTLS kernel weights, see :func:`_rtls_surface`.

    Returns
    -------
    xr.DataArray
        Raw two-way radiance, with ``sza`` and ``vza`` paired against
        the points of a batched call.
    """
    from smartg.smartg import LocalEstimate, Smartg

    atm, batch, atm_size = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    sensors = _viewing_sensors(vza, saa + raa, sat_height)
    sun_le = LocalEstimate(th_deg=np.atleast_1d(np.squeeze(sza.values)), phi_deg=saa)

    smartg = Smartg(autoinit=False)
    try:
        res: xr.DataArray = smartg.run(
            wavelength=atm["wavelength"],
            atmosphere=atm,
            surface=_rtls_surface(k0, k1p, k2p),
            sensor=sensors,
            le=sun_le,
            n_photons=n_ph * atm_size * len(sensors),
            n_icdf=int(1e3),
            r_min=1,
            r_max=1,
        )["I_up (TOA)"]
    finally:
        smartg.clear_context()

    return collect_batched(
        res,
        batch,
        angles={
            "sensor index": ("vza", np.atleast_1d(np.squeeze(vza.values))),
            "Zenith angles": ("sza", np.atleast_1d(np.squeeze(sza.values))),
        },
        drop=["Azimuth angles"],
    )


def sph_alb_brdf(
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
    saa: float,
    sat_height: float,
    k0: float,
    k1p: float,
    k2p: float,
) -> xr.DataArray:
    """Sample the flux returned to the surface by one RTLS reflection.

    Photons leave the sun direction as a planar flux, reflect **once**
    on the surface, and the flux coming back down at ground level is
    read.  Normalised by the downward transmittance it gives the
    coupling term the 5S formula writes as ``sph_alb`` for a Lambertian
    surface; :class:`~adjeff.modules.samplers.SphAlbBrdfSampler` does
    that division.

    Only ``sza`` is swept: the quantity is a hemispheric integral, so
    there is no viewing direction to carry.

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
        Solar zenith angles [deg], 1-D.
    species : dict[str, float]
        OPAC aerosol species and fractional contributions.
    afgl_type : str
        AFGL standard atmosphere profile identifier.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    n_ph : int
        Number of photons per Smart-G call and per sensor.
    saa : float
        Solar azimuth angle [deg].
    sat_height : float
        Altitude the photons are launched from [km].
    k0, k1p, k2p : float
        RTLS kernel weights, see :func:`_rtls_surface`.

    Returns
    -------
    xr.DataArray
        Raw downward flux at ground, with ``sza`` paired against the
        points of a batched call.
    """
    from smartg.sensor import Sensor
    from smartg.smartg import Smartg

    atm, batch, atm_size = _make_atmosphere(
        wl, aot, rh, h, href, species, afgl_type, remove_rayleigh
    )
    # A planar-flux source (sensor_type=1), so the output is a flux and no
    # local estimate is involved: an `le` here would have no effect.
    sensors = [
        Sensor(
            pos_z=sat_height,
            th_deg=float(180.0 - th),
            ph_deg=float(saa),
            loc="ATMOS",
            sensor_type=1,
        )
        for th in np.atleast_1d(np.squeeze(sza.values))
    ]

    smartg = Smartg(autoinit=False)
    try:
        res: xr.DataArray = smartg.run(
            wavelength=atm["wavelength"],
            atmosphere=atm,
            surface=_rtls_surface(k0, k1p, k2p),
            sensor=sensors,
            n_photons=n_ph * atm_size * len(sensors),
            output_layers=3,
            flux="planar",
            n_icdf=int(1e3),
            r_min=1,
            r_max=1,
        )["flux_down (0+)"]
    finally:
        smartg.clear_context()

    return collect_batched(
        res,
        batch,
        angles={"sensor index": ("sza", np.atleast_1d(np.squeeze(sza.values)))},
    )
