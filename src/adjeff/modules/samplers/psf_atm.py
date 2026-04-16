"""Module that computes rho_toa with Smart-G (symmetric radial sampling)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import geoclide as gc  # type: ignore[import-untyped]
import numpy as np
import xarray as xr
from smartg.visualizegeo import Entity, Plane, Transformation
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict, SensorBand

from ..scene_module_sweep import SceneModuleSweep

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SmartgSampler_PSF_Atm(SceneModuleSweep):
    """Compute the atmospheric PSF of the 5S model.

    For each band, the module samples the PSF using a Smart-G Entity object.
    Photons are backward launched from the sensor, propagated through the
    atmosphere until they reach the Entity on the earth surface. When a photon
    hits the ground, its energy is registered by the Entity, and the photon
    path is terminated. This allows to avoid sampling coupling effects in the
    PSF model.

    vza and sza are scalar per call — sensor positions depend on vza so they
    cannot be vectorised within one run. Sweep over angles externally.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric parameters — may be full arrays (swept via multi_profiles).
    geo_config : GeoConfig
        Geometry — vza and sza must be single-element (scalar per call).
    spectral_config : SpectralConfig
        Bands to process.
    remove_rayleigh : bool
        Whether to suppress Rayleigh scattering.
    afgl_type : str
        AFGL atmosphere profile identifier.
    nr : int
        Number of radial sampling points.
    n_ph : int
        Number of photons per sensor.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["psf_atm"]
    scalar_dims: ClassVar[list[str]] = [
        "vza",
        "vaa",
        "aot",
        "rh",
        "h",
        "href",
    ]
    vector_dims: ClassVar[list[str]] = []

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        nr: int = 100,
        n_ph: int = int(1e6),
        cache: utils.CacheStore | None = None,
    ) -> None:
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.remove_rayleigh = remove_rayleigh
        self.afgl_type = afgl_type
        self.nr = nr
        self.n_ph = n_ph
        super().__init__(cache=cache)

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.atmo_config, self.geo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Run the radial rho_toa computation for every band in the scene."""
        bundle: utils.ConfigBundle = self._make_bundle()

        # Sample the Atmospheric PSF for each band
        new_scene = ImageDict({b: xr.Dataset() for b in scene.bands})
        for band in scene.bands:
            logger.info("Start rho_toa computation.", band=band)
            psf_atm_arr: xr.DataArray = bundle.apply(
                psf_atm,
                rho_s=scene[band],
                band=band,
                species=self.atmo_config.species,
                afgl_type=self.afgl_type,
                remove_rayleigh=self.remove_rayleigh,
                n_ph=self.n_ph,
            )
            logger.info(
                "Computed atmospheric PSF.",
                dims=psf_atm_arr.dims,
                band=band,
            )

            new_scene[band]["psf_atm"] = psf_atm_arr
        return new_scene


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

    # Create entity
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

    # Create an atmosphere for each combination of AtmoParams
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

    # Launch Smart-G for each atmosphere
    atm_size = len(batch.index_coord)

    # Computation with Smart-G
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
