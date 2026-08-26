"""Direct upward transmittance (``tdir_up``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

from ._atmo_sampler import AtmoSampler
from ._smartg import tdir_up


class TdirUpSampler(AtmoSampler):
    """Sample direct upward transmittance analytically from optical depth.

    ``tdir_up`` is the fraction of the surface-reflected flux that
    reaches the satellite without being scattered.  Computed
    analytically from the total optical depth ``OD`` retrieved by
    Smart-G::

        tdir_up = exp(-OD / cos(vza))

    This is the upward counterpart of :class:`TdirDownSampler`.

    Produced variable: ``tdir_up``.

    Notes
    -----
    Requires a CUDA-capable GPU (for the Smart-G optical depth
    retrieval), but very few photons suffice (default ``n_ph=1e9``).

    Parameters are those of :class:`AtmoSampler`.
    """

    _output_vars: ClassVar[list[str]] = ["tdir_up"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, vza) vec(wl) -> tdir_up(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdir_up)
    default_n_ph: ClassVar[int] = 1000000000
