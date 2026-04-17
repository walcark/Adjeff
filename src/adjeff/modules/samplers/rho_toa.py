"""Module that computes rho_toa with Smart-G (full 2D, no symmetry assumption).

Unlike :mod:`rho_toa_sym`, this module accepts an arbitrary surface
reflectance map.  The full 2D albedo field is passed to Smart-G via an
``Albedo_map`` environment; sensors are placed on an ``nx × ny`` sub-grid
starting at ``topleft_pix``.  Unsampled pixels are set to ``NaN`` (no
interpolation); a companion boolean variable ``rho_toa_valid`` marks which
pixels were actually computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict, SensorBand

from ..scene_module_sweep import SceneModuleSweep
from .rho_atm import SmartgSampler_Rho_atm

if TYPE_CHECKING:
    from smartg.smartg import Sensor

logger = get_logger(__name__)


class SmartgSampler_Rho_toa(SceneModuleSweep):
    """Compute rho_toa by 2D grid sampling without symmetry assumption.

    The full 2D surface reflectance map is encoded as an ``Albedo_map``
    environment passed to Smart-G.  ``nx × ny`` sensors are placed on the
    sub-grid starting at ``topleft_pix``; after the simulation the flat sensor
    axis is reshaped into ``(y, x)``.  Pixels outside the sampled region are
    set to ``NaN``; a companion ``rho_toa_valid`` boolean variable marks which
    pixels were computed.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric parameters — may be full arrays (swept via
        ``multi_profiles``).
    geo_config : GeoConfig
        Geometry — ``vza`` and ``sza`` must be scalar per call.
    remove_rayleigh : bool
        Whether to suppress Rayleigh scattering.
    afgl_type : str
        AFGL atmosphere profile identifier.
    nx : int
        Number of sensor columns (x dimension).
    ny : int
        Number of sensor rows (y dimension).
    n_ph : int
        Number of photons per sensor.
    n_alb : int
        Number of discrete albedo levels in the ``Albedo_map`` (default 1000).
    rho_background : float | "mean" | "min" | "zero"
        Reflectance of the ``LambSurface`` for photons leaving the
        ``Albedo_map`` region.  See :class:`~adjeff.atmosphere.SurfaceFactory`
        for details.  Default is ``"mean"``.
    """

    required_vars: ClassVar[list[str]] = ["rho_s"]
    output_vars: ClassVar[list[str]] = ["rho_toa"]
    scalar_dims: ClassVar[list[str]] = ["sza", "vza"]
    vector_dims: ClassVar[list[str]] = ["aot", "rh", "h", "href"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        nx: int = 50,
        ny: int = 50,
        topleft_pix: tuple[int, int] = (0, 0),
        n_ph: int = int(1e6),
        n_alb: int = 1000,
        rho_background: float | Literal["mean", "min", "zero"] = "mean",
        cache: utils.CacheStore | None = None,
    ) -> None:
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.remove_rayleigh = remove_rayleigh
        self.afgl_type = afgl_type
        self.topleft_pix = topleft_pix
        self.nx = nx
        self.ny = ny
        self.n_ph = n_ph
        self.n_alb = n_alb
        self.rho_background = rho_background
        super().__init__(cache=cache)

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.atmo_config, self.geo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Run the 2D rho_toa computation for every band in the scene."""
        bundle: utils.ConfigBundle = self._make_bundle()

        # First, compute rho_atm
        scene = SmartgSampler_Rho_atm(
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            spectral_config=atmo.SpectralConfig.from_bands(scene.bands),
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph=int(3e7),
            cache=self._cache,
        )(scene)

        for band in scene.bands:
            logger.info("Start rho_toa (2D) computation.", band=band)
            rho_toa_arr: xr.DataArray = bundle.apply(
                _rho_toa,
                saa=self.geo_config.saa.item(),
                vaa=self.geo_config.vaa.item(),
                rho_s=scene[band],
                band=band,
                species=self.atmo_config.species,
                sat_height=self.geo_config.sat_height,
                afgl_type=self.afgl_type,
                remove_rayleigh=self.remove_rayleigh,
                nx=self.nx,
                ny=self.ny,
                topleft_pix=self.topleft_pix,
                n_ph=self.n_ph,
                n_alb=self.n_alb,
                rho_background=self.rho_background,
            )
            logger.info(
                "Computed rho_toa (2D).", dims=rho_toa_arr.dims, band=band
            )
            scene[band]["rho_toa"] = rho_toa_arr

            x_full = scene[band]["rho_s"].coords["x"].values
            y_full = scene[band]["rho_s"].coords["y"].values
            x_s = x_full[self.topleft_pix[0] : self.topleft_pix[0] + self.nx]
            y_s = y_full[self.topleft_pix[1] : self.topleft_pix[1] + self.ny]
            valid = xr.DataArray(
                np.zeros((len(y_full), len(x_full)), dtype=bool),
                dims=["y", "x"],
                coords={"y": y_full, "x": x_full},
            )
            valid.loc[{"y": y_s, "x": x_s}] = True
            scene[band]["rho_toa_valid"] = valid

        return scene


def _rho_toa(
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
            "SmartgSampler_Rho_toa requires an arbitrary rho_s surface "
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

    # Atmosphere batch
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

    # Squeeze size-1 angle dims; ensure sensor and wavelength dims are present
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

    # Remap wavelength → batch index, then unstack to atmo coords
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

    # Reindex to the full image grid; unsampled pixels become NaN
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
