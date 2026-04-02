"""Module that computes tdif_up with Smart-G."""

from typing import ClassVar

import numpy as np
import xarray as xr
from smartg.smartg import Sensor, Smartg
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep

logger = get_logger(__name__)


class SmartgSampler_Tdif_up(SceneModuleSweep):
    """Sample upward diffuse transmittance with Smart-G.

    Computes ``tdif_up`` — the diffuse component of the upward flux
    reaching the satellite — for every combination of viewing geometry
    and atmospheric state defined by the supplied configs.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Viewing geometry (``vza``, ``saa``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons per Smart-G call, by default ``3e7``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    chunks : dict[str, int] or None, optional
        Chunk sizes for vector dimensions.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before sweeping.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdif_up"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href", "vza"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(3e7),
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
            tdif_up,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
            saa=self.geo_config.saa.values,
        )
        logger.info("Computed tdif_up.", dims=arr.dims)

        for band in self.spectral_config.bands:
            scene[band]["tdif_up"] = arr.sel(wl=band.wl_nm)

        return scene


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
    logger.info("Computing tdif_up ...", wl=wl, vza=vza)

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
