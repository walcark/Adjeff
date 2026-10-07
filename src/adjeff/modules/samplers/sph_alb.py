"""Atmospheric spherical albedo.

Classes
-------
    SphAlbSampler
        Samples ``sph_alb``.
"""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
from adjeff.utils import CacheStore

from ._atmo_sampler import AtmoSampler
from ._smartg import sph_alb


class SphAlbSampler(AtmoSampler):
    """Sample the atmospheric spherical albedo with Smart-G Monte-Carlo.

    ``sph_alb`` is the fraction of the upwelling flux that is reflected
    back downward by the atmosphere.  It appears in the 5S formula as
    the multiple-reflection coupling term::

        rho_toa = rho_atm + t_up * t_down * rho_unif / (1 - sph_alb * rho_unif)

    Geometry-independent: no ``geo_config`` is required.  All parameters
    are swept as ``vector_dims``.

    Produced variable: ``sph_alb``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters are those of :class:`AtmoSampler`, less *geo_config*:
    the spherical albedo does not depend on a geometry.
    """

    _output_vars: ClassVar[list[str]] = ["sph_alb"]
    contract: ClassVar[str] = "batch(aot, rh, h, href) vec(wl) -> sph_alb(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(sph_alb)
    default_n_ph: ClassVar[int] = 20000000

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int | None = None,
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rename: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            atmo_config=atmo_config,
            geo_config=None,
            spectral_config=spectral_config,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            n_ph=n_ph,
            cache=cache,
            batch_size=batch_size,
            dedup=dedup,
            rename=rename,
        )
