"""Module that computes rho_toa with Smart-G (full 2D, no symmetry assumption).

Unlike :mod:`rho_toa_sym`, this module accepts an arbitrary surface
reflectance map.  The full 2D albedo field is passed to Smart-G via an
``Albedo_map`` environment; sensors are placed on an ``nx × ny`` sub-grid
starting at ``topleft_pix``.  Unsampled pixels are set to ``NaN`` (no
interpolation); a companion boolean variable ``rho_toa_valid`` marks which
pixels were actually computed.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep
from ._smartg import rho_toa
from .rho_atm import SmartgSampler_Rho_atm

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
                rho_toa,
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
