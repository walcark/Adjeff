"""Module that computes spherical albedo with Smart-G."""

from typing import ClassVar

import xarray as xr
from smartg.smartg import Sensor, Smartg
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep

logger = get_logger(__name__)


class SmartgSampler_Sph_alb(SceneModuleSweep):
    """Sample spherical albedo of the atmosphere with Smart-G."""

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["sph_alb"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(2e7),
        cache: utils.CacheStore | None = None,
        chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(
            cache=cache, chunks=chunks, deduplicate_dims=deduplicate_dims
        )

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        for band in self.spectral_config.bands:
            if band not in scene.bands:
                scene[band] = xr.Dataset()

        bundle: utils.ConfigBundle = self._make_bundle()
        arr: xr.DataArray = bundle.apply(
            sph_alb,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
        )
        logger.info("Computed sph_alb.", dims=arr.dims)

        for band in self.spectral_config.bands:
            scene[band]["sph_alb"] = arr.sel(wl=band.wl_nm)

        return scene


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
    """Compute the spherical albedo of the atmosphere with Smart-G."""
    logger.info("Computing sph_alb ...", wl=wl)

    # Create an atmosphere for each combination of AtmoParams
    batch: utils.ParamBatch = utils.ParamBatch.from_dataarrays(
        wl=wl, aot=aot, rh=rh, href=href, h=h
    )
    atm = atmo.create_atmosphere(
        batch.as_dict(),
        species=species,
        afgl_type=afgl_type,
        remove_rayleigh=remove_rayleigh,
    )

    # Launch Smart-G for each atmosphere
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

    # Adapt output of Smart-G simulation
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
