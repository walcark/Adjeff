"""Module that computes rho_toa with Smart-G (symmetric radial sampling)."""

from __future__ import annotations

from typing import Any, ClassVar

from structlog import get_logger

import adjeff.atmosphere as atmo
from adjeff.core import ImageDict
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from ..sweep_sampler import SweepSampler
from ._smartg import rho_toa_sym
from .rho_atm import RhoAtmSampler

logger = get_logger(__name__)


class RhoToaSymSampler(SweepSampler):
    """Compute rho_toa by radial sampling under the symmetric PSF assumption.

    Assumes the scene is radially symmetric around the image centre.
    For each band the module:

    1. Bins ``rho_s`` into ``nr`` radial samples from centre to edge.
    2. Places one Smart-G sensor per radial point.
    3. Runs a single simulation per ``(sza, vza)`` combination, sweeping
       all atmospheric states (``aot``, ``rh``, ``h``, ``href``) as a
       multi-profile atmosphere.
    4. Writes ``rho_toa`` with dims ``(r, aot, rh, h, href)`` into the
       scene.

    ``sza`` and ``vza`` are ``scalar_dims`` — sensor positions depend on
    ``vza`` and cannot be vectorised within one call.

    Produced variable: ``rho_toa``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Geometry — ``vza`` and ``sza`` must be single-element per call.
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
    # `sza` and `vza` stay `loop`, not `batch`: the sensor positions are
    # built from them, so a call carries one geometry.  The atmospheric
    # axes ride along as `vec`, which is how Smart-G wants them.
    contract: ClassVar[str] = (
        # The output dim order is the one ParamBatch produces inside
        # _smartg, which broadcasts wl, aot, rh, href, h in that order.
        "loop(sza, vza) vec(aot, rh, h, href) "
        "-> rho_toa(aot, rh, href, h, y, x)"
    )
    point_fn: ClassVar[Any] = staticmethod(rho_toa_sym)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        nr: int = 100,
        n_ph: int = int(1e6),
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
    ) -> None:
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.remove_rayleigh = remove_rayleigh
        self.afgl_type = afgl_type
        self.nr = nr
        self.n_ph = n_ph
        super().__init__(cache=cache, batch_size=batch_size, dedup=dedup)

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        return (self.atmo_config, self.geo_config)

    def _statics(self) -> dict[str, Any]:
        return {
            "saa": self.geo_config.saa.item(),
            "vaa": self.geo_config.vaa.item(),
            "species": self.atmo_config.species,
            "sat_height": self.geo_config.sat_height,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "nr": self.nr,
            "n_ph": self.n_ph,
        }

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Run the radial rho_toa computation for every band in the scene."""
        scene = RhoAtmSampler(
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            spectral_config=atmo.SpectralConfig.from_bands(scene.bands),
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph=int(3e7),
            cache=self._cache,
        )(scene)

        # One sweep per band: the physics reads the band's own scene, so
        # the two travel together rather than through the sweep space.
        for band in scene.bands:
            arr = self._sweep(rho_s=scene[band], band=band)
            scene[band]["rho_toa"] = self._restore_coords(arr, scene[band])

        return scene
