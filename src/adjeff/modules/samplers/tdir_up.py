"""Module that computes tdir_up with Smart-G."""

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


class SmartgSampler_Tdir_up(SceneModuleSweep):
    """Sample direct upward transmittance analytically from optical depth.

    Computes ``tdir_up = exp(-OD / cos(vza))`` where the optical depth
    ``OD`` is retrieved from Smart-G.  See also
    :class:`SmartgSampler_Tdir_down` for the downward counterpart.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Viewing geometry (``vza``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh optical depth is set to zero.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons for the optical depth retrieval,
        by default ``1e9``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    chunks : dict[str, int] or None, optional
        Chunk sizes for vector dimensions.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before sweeping.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdir_up"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href", "vza"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(1e9),
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
            cache=cache,
            chunks=chunks,
            deduplicate_dims=deduplicate_dims,
        )

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        """Return configuration objects."""
        return (self.spectral_config, self.atmo_config, self.geo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Compute the direct upward transmittance + sweep over parameters."""
        # Ensure that SpetralConfig wavelengths are in scene
        for band in self.spectral_config.bands:
            if band not in scene.bands:
                logger.warning(
                    "Computed band not in scene bands.",
                    band=band,
                    scene_bands=scene.bands,
                )
                scene[band] = xr.Dataset()
                logger.warning(
                    "Initialized an empty dataset for the band.",
                    band=band,
                )

        # Compute tdir_up for each parameters
        bundle: utils.ConfigBundle = self._make_bundle()
        tdir_up_arr: xr.DataArray = bundle.apply(
            tdir_up,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
        )
        logger.info("Computed tdir_up.", dims=tdir_up_arr.dims)

        # Add each wavelength to a scene band
        for band in self.spectral_config.bands:
            scene[band]["tdir_up"] = tdir_up_arr.sel(wl=band.wl_nm)

        return scene


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
    od = optical_depth(atm)
    od = batch.unstack(
        xr.DataArray(od, dims=["index"], coords={"index": batch.index_coord}),
    )
    logger.info("tdir_up successfully calculated.")
    return xr.DataArray(xr.apply_ufunc(np.exp, -od / np.cos(np.deg2rad(vza))))


def optical_depth(atm: MLUT) -> xr.DataArray:
    """Return the optical depth of the atmosphere."""
    wl = atm.axes["wavelength"]
    smartg = Smartg(autoinit=False)
    res: MLUT = smartg.run(atm=atm, wl=wl, NBPHOTONS=100, NF=1000)
    smartg.clear_context()
    return xr.DataArray(
        res["OD_atm"].to_xarray().sel(z_atm=0.0).drop_vars("z_atm")
    )
