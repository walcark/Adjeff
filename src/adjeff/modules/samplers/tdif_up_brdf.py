"""Diffuse upward transmittance over a non-lambertian surface."""

from typing import Any, Callable, ClassVar

import xarray as xr

from ._brdf_sampler import BrdfSampler
from ._smartg import tdif_up_brdf


class TdifUpBrdfSampler(BrdfSampler):
    """Sample ``tdif_up`` over an RTLS surface with Smart-G Monte-Carlo.

    The Lambertian :class:`TdifUpSampler` emits isotropically from the
    ground, which only measures the atmosphere because a Lambertian
    surface re-emits the same field whatever the incidence.  A BRDF does
    not, so the surface has to be lit from a direction: photons leave
    the satellite along ``vza``, reflect once, and are collected toward
    the sun.  Dividing by the downward transmittance and removing the
    direct beam leaves::

        tdif_up = raw / (tdir_down + tdif_down) - tdir_up

    Produced variable: ``tdif_up``, the same slot the Lambertian sampler
    writes, so nothing downstream changes.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Two costs separate it from the Lambertian sampler, and both are
    physical rather than incidental.  It sweeps ``sza`` **and** ``vza``,
    since a BRDF breaks the reciprocity that collapsed them into one.
    And it measures the total transmittance and subtracts the direct
    part, where the Lambertian sampler measures the diffuse part
    directly: for a given photon count the relative error is worse by
    roughly ``sqrt(T / tdif)``, hence the higher ``default_n_ph``.

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
