"""Module that computes rho_toa with Smart-G (full 2D, no symmetry assumption).

Unlike :mod:`rho_toa_sym`, this module accepts an arbitrary surface
reflectance map.  The full 2D albedo field is passed to Smart-G via an
``Albedo_map`` environment; sensors are placed on an ``nx × ny`` sub-grid
starting at ``topleft_pix``.  Unsampled pixels are set to ``NaN`` (no
interpolation); a companion boolean variable ``rho_toa_valid`` marks which
pixels were actually computed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from structlog import get_logger

import adjeff.atmosphere as atmo
from adjeff.core import ImageDict
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from ..sweep_sampler import SweepSampler
from ._smartg import rho_toa
from .rho_atm import ensure_rho_atm

logger = get_logger(__name__)


class RhoToaSampler(SweepSampler):
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
    #: Reused when the scene carries it, computed otherwise.  Keyed so
    #: that two different path reflectances cannot share one entry.
    optional_vars: ClassVar[list[str]] = ["rho_atm"]
    # `sza` and `vza` stay `loop`: the sensor grid is built from them, so
    # a call carries one geometry.  The output dim order is the one
    # ParamBatch produces inside _smartg (wl, aot, rh, href, h).
    contract: ClassVar[str] = (
        "loop(sza, vza) vec(aot, rh, h, href) "
        "-> rho_toa(aot, rh, href, h, y, x)"
    )
    point_fn: ClassVar[Any] = staticmethod(rho_toa)

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
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
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
            "nx": self.nx,
            "ny": self.ny,
            "topleft_pix": self.topleft_pix,
            "n_ph": self.n_ph,
            "n_alb": self.n_alb,
            "rho_background": self.rho_background,
        }

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Run the 2D rho_toa computation for every band in the scene."""
        scene = ensure_rho_atm(
            scene,
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph=int(3e7),
            cache=self._cache,
        )

        # One sweep per band: the physics reads the band's own scene, so
        # the two travel together rather than through the sweep space.
        for band in scene.bands:
            arr = self._sweep(rho_s=scene[band], band=band)
            scene[band]["rho_toa"] = self._restore_coords(arr, scene[band])

        return scene
