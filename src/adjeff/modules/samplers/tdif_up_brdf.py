"""Diffuse upward transmittance over an RTLS surface.

Classes
-------
    TdifUpBrdfSampler
        Samples ``tdif_up``, in place of TdifUpSampler.
"""

from typing import Any, Callable, ClassVar

import xarray as xr

from ._brdf_sampler import BrdfSampler
from ._smartg import tdif_up_brdf


class TdifUpBrdfSampler(BrdfSampler):
    """Sample ``tdif_up`` over an RTLS surface with Smart-G Monte-Carlo.

    A BRDF has to be lit from a direction, so photons leave the
    satellite along ``vza``, reflect once, and are collected toward the
    sun.  Dividing by the downward transmittance and removing the direct
    beam leaves::

        tdif_up = raw / (tdir_down + tdif_down) - tdir_up

    It writes the slot the Lambertian :class:`TdifUpSampler` writes, so
    nothing downstream changes.  Two costs follow from the BRDF: ``sza``
    and ``vza`` are both swept, reciprocity no longer collapsing them;
    and subtracting the direct beam costs a factor ``sqrt(T / tdif)`` in
    relative error, hence the higher ``default_n_ph``.

    Parameters are those of :class:`BrdfSampler`.
    """

    _output_vars: ClassVar[list[str]] = ["tdif_up"]
    contract: ClassVar[str] = (
        "batch(aot, rh, h, href, sza, vza) vec(wl) -> tdif_up_raw(wl)"
    )
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdif_up_brdf)
    default_n_ph: ClassVar[int] = 100000000
    geo_statics: ClassVar[tuple[str, ...]] = ("saa", "raa", "sat_height")

    def _normalise(self, raw: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
        """Divide out the downward path, then remove the direct beam."""
        return raw / self._t_sun(ds) - ds[self._slot("tdir_up")]
