"""Module that computes tdir_down analytically from optical depth."""

from typing import ClassVar

import numpy as np
import xarray as xr
from luts.luts import MLUT  # type: ignore[import-untyped]
from smartg.smartg import Smartg
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep

logger = get_logger(__name__)


class SmartgSampler_Tdir_down(SceneModuleSweep):
    """Sample direct downward transmittance analytically from optical depth.

    Computes ``tdir_down = exp(-OD / cos(sza))`` where the optical depth
    ``OD`` is retrieved from Smart-G.  This is an analytical computation;
    the Monte-Carlo photon count only affects the optical depth retrieval
    and can therefore be kept very small (default ``1e2``).

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Illumination geometry (``sza``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh optical depth is set to zero.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons for the optical depth retrieval,
        by default ``1e2``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    chunks : dict[str, int] or None, optional
        Chunk sizes for vector dimensions.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before sweeping.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdir_down"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href", "sza"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(1e2),
        cache: utils.CacheStore | None = None,
        chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(
            cache=cache, chunks=chunks, deduplicate_dims=deduplicate_dims
        )

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config, self.geo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        for band in self.spectral_config.bands:
            if band not in scene.bands:
                scene[band] = xr.Dataset()

        bundle: utils.ConfigBundle = self._make_bundle()
        arr: xr.DataArray = bundle.apply(
            tdir_down,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
        )
        logger.info("Computed tdir_down.", dims=arr.dims)

        for band in self.spectral_config.bands:
            scene[band]["tdir_down"] = arr.sel(wl=band.wl_nm)

        return scene


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

    # Compute optical depth with Smart-G and reconstruct full dimensions
    od = _optical_depth(atm, n_ph)
    od = batch.unstack(
        xr.DataArray(
            od.values, dims=["index"], coords={"index": batch.index_coord}
        )
    )
    logger.info("tdir_down successfully calculated.")
    return xr.DataArray(np.exp(-od / np.cos(np.deg2rad(sza))))


def _optical_depth(atm: MLUT, n_ph: int) -> xr.DataArray:
    wl = atm.axes["wavelength"]
    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wl=wl, atm=atm, NBPHOTONS=n_ph, NF=int(1e3)
    )["OD_atm"].to_xarray()
    smartg.clear_context()
    if len(wl) == 1 and "wavelength" not in res.dims:
        res = res.expand_dims(wavelength=wl)
    return res.sel(z_atm=0.0).drop_vars("z_atm")
