"""Shared construction of the two RTLS samplers.

Both sample a raw quantity over an RTLS surface, then divide out the
downward transmittance to obtain a 5S term.

Classes
-------
    BrdfSampler
        SweepSampler over the atmosphere and the RTLS kernels.
"""

from __future__ import annotations

from typing import Any, ClassVar

import xarray as xr

import adjeff.atmosphere as atmo
from adjeff.core import ImageDict
from adjeff.utils import CacheStore

from ._atmo_sampler import AtmoSampler
from ._ensure import ensure_downward


class BrdfSampler(AtmoSampler):
    """AtmoSampler over an RTLS surface, normalised by the scene.

    Subclasses implement :meth:`_normalise`, turning the raw output of
    one band into its 5S term.  The kernel weights are floats rather
    than a Smart-G surface, so that they hash reliably in the cache key.

    Parameters
    ----------
    k0 : float, optional
        Albedo of the isotropic kernel, 1 by default.  ``k0 = 1`` with
        ``k1p = k2p = 0`` is a white Lambertian surface.
    k1p, k2p : float, optional
        Geometric and volumetric kernel weights, relative to *k0*.
    n_ph_tdif_down : int or None, optional
        Photons for ``tdif_down`` when the scene lacks it; *n_ph* for
        ``None``, since the ratio's error depends on both.

    The other parameters are those of :class:`AtmoSampler`.
    """

    #: Reused when the scene carries them, computed otherwise.  Keyed so
    #: that two different draws cannot share one entry.
    _optional_vars: ClassVar[list[str]] = ["tdir_down", "tdif_down", "tdir_up"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        k0: float = 1.0,
        k1p: float = 0.0,
        k2p: float = 0.0,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int | None = None,
        n_ph_tdif_down: int | None = None,
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rename: dict[str, str] | None = None,
    ) -> None:
        self.k0 = k0
        self.k1p = k1p
        self.k2p = k2p
        super().__init__(
            atmo_config=atmo_config,
            geo_config=geo_config,
            spectral_config=spectral_config,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            n_ph=n_ph,
            cache=cache,
            batch_size=batch_size,
            dedup=dedup,
            rename=rename,
        )
        self.n_ph_tdif_down = self.n_ph if n_ph_tdif_down is None else n_ph_tdif_down

    def _statics(self) -> dict[str, Any]:
        statics = super()._statics()
        statics.update(k0=self.k0, k1p=self.k1p, k2p=self.k2p)
        return statics

    def _geo_static(self, name: str) -> Any:
        """Serve ``raa`` from the two azimuths, everything else as usual.

        The BRDF responds to the relative azimuth, not to either azimuth
        alone.  Deriving it here rather than in the physics keeps the
        point function reading in the terms the surface model uses.
        """
        if name != "raa":
            return super()._geo_static(name)
        saa = super()._geo_static("saa")
        vaa = super()._geo_static("vaa")
        return float((vaa - saa) % 360.0)

    def _normalise(self, raw: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
        """Turn the raw sweep output for one band into the 5S term."""
        raise NotImplementedError

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Sweep, then divide the raw draw by the scene's own transmittances."""
        assert self.geo_config is not None  # a BRDF always has a geometry
        scene = ensure_downward(
            scene,
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            spectral_config=self.spectral_config,
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph_tdif_down=self.n_ph_tdif_down,
            cache=self._cache,
            rename={
                role: self._slot(role)
                for role in self._optional_vars
                if self._slot(role) != role
            },
        )
        raw = self._sweep()
        name = self._output_vars[0]
        for band in self.spectral_config.bands:
            ds = scene[band]
            # Coordinates first: the algebra below aligns by label, and
            # xsweep numbers a dim it did not sweep from zero, which no
            # longer matches the scene's own labels.
            arr = self._restore_coords(raw.sel(wl=band.wl_nm), ds)
            ds[self._slot(name)] = self._normalise(arr, ds)
        return scene

    def _t_sun(self, ds: xr.Dataset) -> xr.DataArray:
        """Return the total downward transmittance the scene holds."""
        return ds[self._slot("tdir_down")] + ds[self._slot("tdif_down")]
