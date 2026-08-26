"""Diffuse downward transmittance (``tdif_down``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

from ._atmo_sampler import AtmoSampler
from ._smartg import tdif_down


class TdifDownSampler(AtmoSampler):
    """Sample downward diffuse transmittance with Smart-G Monte-Carlo.

    ``tdif_down`` is the diffuse fraction of the solar flux reaching the
    surface (scattered photons, as opposed to the direct beam
    ``tdir_down``).  It contributes to the total downward transmittance
    in the 5S formula::

        t_down = tdir_down + tdif_down

    Produced variable: ``tdif_down``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters are those of :class:`AtmoSampler`.
    """

    _output_vars: ClassVar[list[str]] = ["tdif_down"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, sza) vec(wl) -> tdif_down(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdif_down)
    default_n_ph: ClassVar[int] = 30000000
    geo_statics: ClassVar[tuple[str, ...]] = ("saa", "sat_height")
