"""Spherical albedo coupling term over a non-lambertian surface."""

from typing import Any, Callable, ClassVar

import xarray as xr

from ._brdf_sampler import BrdfSampler
from ._smartg import sph_alb_brdf


class SphAlbBrdfSampler(BrdfSampler):
    """Sample ``sph_alb`` over an RTLS surface with Smart-G Monte-Carlo.

    Photons leave the sun direction as a planar flux, reflect once on
    the surface, and the flux returning to the ground is read.  Divided
    by the downward transmittance it gives the coupling term the 5S
    formula writes as ``sph_alb``::

        sph_alb = raw / (tdir_down + tdif_down)

    Produced variable: ``sph_alb``, the same slot the Lambertian sampler
    writes.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Only ``sza`` is swept: the quantity is a hemispheric integral, so
    there is no viewing direction to carry, and the relative azimuth
    does not enter either.

    Parameters are those of :class:`BrdfSampler`.
    """

    _optional_vars: ClassVar[list[str]] = ["tdir_down", "tdif_down"]
    _output_vars: ClassVar[list[str]] = ["sph_alb"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, sza) vec(wl) -> sph_alb_raw(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(sph_alb_brdf)
    default_n_ph: ClassVar[int] = 30000000
    geo_statics: ClassVar[tuple[str, ...]] = ("saa", "sat_height")

    def _normalise(self, raw: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
        """Divide out the downward path the photons paid on the way in."""
        return raw / self._t_sun(ds)
