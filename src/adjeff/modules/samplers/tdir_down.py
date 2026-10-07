"""Direct downward transmittance.

Classes
-------
    TdirDownSampler
        Samples ``tdir_down``.
"""

from typing import Any, Callable, ClassVar

from ._atmo_sampler import AtmoSampler
from ._smartg import tdir_down


class TdirDownSampler(AtmoSampler):
    """Sample direct downward transmittance analytically from optical depth.

    ``tdir_down`` is the fraction of the solar flux that reaches the
    surface without being scattered.  It is computed analytically from
    the total optical depth ``OD`` retrieved by Smart-G::

        tdir_down = exp(-OD / cos(sza))

    The Monte-Carlo photon count only affects the optical depth
    retrieval and can therefore be kept very small (default ``1e2``).
    This is the downward counterpart of :class:`TdirUpSampler`.

    Produced variable: ``tdir_down``.

    Notes
    -----
    Requires a CUDA-capable GPU (for the Smart-G optical depth
    retrieval), but very few photons suffice (default ``n_ph=1e2``).

    Parameters are those of :class:`AtmoSampler`.
    """

    _output_vars: ClassVar[list[str]] = ["tdir_down"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, sza) vec(wl) -> tdir_down(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdir_down)
    default_n_ph: ClassVar[int] = 100
