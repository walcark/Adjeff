"""Module that computes rho_toa with Smart-G (symmetric radial sampling)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import GaussGeneralPSF, ImageDict, PSFGrid, SensorBand
from adjeff.utils import fft_convolve_2D

from ..scene_module_sweep import SceneModuleSweep
from .rho_atm import SmartgSampler_Rho_atm

if TYPE_CHECKING:
    from smartg.smartg import Sensor

logger = get_logger(__name__)


class SmartgSampler_Rho_toa_sym(SceneModuleSweep):
    """Compute rho_toa by radial sampling under the symmetric PSF assumption.

    For each band, the module:

    1. Reads ``rho_s`` and bins it into a radial profile (``nr`` points from
       centre to image edge) — these radii are the sampling locations.
    2. Batches all AtmoConfig combinations (aot, rh, h, href) with the band
       wavelength, building a multi-profile atmosphere via ``multi_profiles``.
    3. Creates one position-specific sensor per radial point.
    4. Runs a single Smart-G simulation and writes ``rho_toa`` with dims
       ``(r, aot, rh, h, href)`` into the scene.

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

        # First, compute rho_atm if required
        scene = SmartgSampler_Rho_atm(
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            spectral_config=atmo.SpectralConfig.from_bands(scene.bands),
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph=int(3e7),
            cache=self._cache,
        )(scene)

        # Second, compute rho_toa for each band
        for band in scene.bands:
            logger.info("Start rho_toa computation.", band=band)
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
                nr=self.nr,
                n_ph=self.n_ph,
            )
            logger.info("Computed rho_toa.", dims=rho_toa_arr.dims, band=band)

            scene[band]["rho_toa"] = rho_toa_arr

        return scene


def rho_toa(
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

    # Single (sza, saa) local estimate — scalar angles, no zip needed
    sun_le = {"th_deg": sza, "phi_deg": saa}
    surf = atmo.SurfaceFactory().surface(rho_s)
    env = atmo.SurfaceFactory().environment(rho_s)

    # Estimate approximate TOA radial profile
    res: float = rho_s["rho_s"].adjeff.res
    n: int = rho_s["rho_s"].adjeff.n
    n = n - 1 if n % 2 == 0 else n
    approx_psf = GaussGeneralPSF(
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

    # Inverse sampling of the profile CDF -> radial sampling points
    profile = rho_toa_approx.adjeff.radial()
    r_vals: xr.DataArray = profile.adjeff.radial_adaptive(n=nr, max_gap=0.1)

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

    # Adapt output of the Smart-G simulation: (i the Azimuth and Zenith angles
    # axes of size 1 are squeezed ; (ii) the sensor index is rename as r for
    # radial distance and (iii) ensure that the wavelength axis exists.
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

    # Transform the wavelength axis into index to unstack with batch
    result = utils.adapt_smartg_output(
        result,
        rename={"wavelength": "index"},
        coords={"index": batch.index_coord},
    )

    # Unstack index to restore original coordinates
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

    Each ground point at distance r from centre lies along the vaa direction.
    The satellite sensor is placed at distance ``sat_height * tan(vza)`` beyond
    the ground point along the same direction.

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

    x_offset = sat_height * np.tan(np.deg2rad(vza))
    cos_vaa = np.cos(np.deg2rad(vaa))
    sin_vaa = np.sin(np.deg2rad(vaa))

    return [
        Sensor(
            POSX=float((r + x_offset) * cos_vaa),
            POSY=float((r + x_offset) * sin_vaa),
            POSZ=sat_height,
            THDEG=180.0 - vza,
            PHDEG=(vaa + 180.0) % 360.0,
            LOC="ATMOS",
        )
        for r in r_vals
    ]
