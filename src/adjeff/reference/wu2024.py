"""The sampled PSF of Wu et al. (2024), reimplemented on Smart-G.

Reference
---------
Wu, Y. et al. (2024). Sensor-generic adjacency-effect correction for
remote sensing of coastal and inland waters.  *Remote Sensing of
Environment*, 315, 114433.

The method has no closed-form expression: the PSF is *sampled* by Monte
Carlo rather than fitted to a parametric shape.  Photons are launched
backward from the sensor and the energy they deposit on a surface
entity, as a function of its position, is the PSF.

What it deliberately leaves out is the earth-atmosphere coupling: a
photon that bounces off the surface, scatters in the atmosphere and
comes back down is not counted.  That assumption is what adjeff drops,
and comparing against this sampler is how the difference is measured.
The upstream implementation, T-Mart, exposes only a corrected image and
never the kernel, so the method is reimplemented here to make the
comparison possible at the PSF level.
"""

from __future__ import annotations

from typing import ClassVar

import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict
from adjeff.modules.samplers._smartg import psf_atm
from adjeff.modules.scene_module_sweep import SceneModuleSweep

logger = get_logger(__name__)


class WuPsfSampler(SceneModuleSweep):
    """Sample the atmospheric PSF the way Wu et al. (2024) do.

    A published method rather than one of adjeff's own: it lives in
    :mod:`adjeff.reference` because its purpose is to be compared
    against, not to be improved.

    Photons are launched backward from the sensor, propagated through
    the atmosphere until they hit a Smart-G ``Entity`` placed on the
    surface.  The energy deposited on the entity as a function of its
    position gives the atmospheric PSF directly, without coupling to
    surface adjacency effects.

    ``vza`` and ``sza`` are ``scalar_dims`` (one Smart-G call per
    angle combination); all atmospheric parameters are also swept as
    scalars.

    Produced variable: ``psf_atm``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Geometry — ``vza`` and ``sza`` must be single-element per call.
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
        """Run the atmospheric PSF sampling for every band in the scene."""
        bundle, _ = self._make_bundle()

        for band in scene.bands:
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

            scene[band]["psf_atm"] = psf_atm_arr
        return scene
