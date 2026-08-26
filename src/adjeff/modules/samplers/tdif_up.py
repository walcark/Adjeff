"""Diffuse upward transmittance (``tdif_up``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

from ._atmo_sampler import AtmoSampler
from ._smartg import tdif_up


class TdifUpSampler(AtmoSampler):
    """Sample upward diffuse transmittance with Smart-G Monte-Carlo.

    ``tdif_up`` is the diffuse component of the surface-reflected flux
    reaching the satellite (scattered photons, as opposed to the direct
    beam ``tdir_up``).  It contributes to the total upward transmittance
    in the 5S formula::

        t_up = tdir_up + tdif_up

    Produced variable: ``tdif_up``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters are those of :class:`AtmoSampler`.
    """

    _output_vars: ClassVar[list[str]] = ["tdif_up"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, vza) vec(wl) -> tdif_up(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdif_up)
    default_n_ph: ClassVar[int] = 30000000
    geo_statics: ClassVar[tuple[str, ...]] = ("saa",)
