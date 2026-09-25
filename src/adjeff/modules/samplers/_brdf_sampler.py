"""What the two non-lambertian samplers share.

Both draw a raw Monte-Carlo quantity over an RTLS surface and turn it
into a 5S term by dividing out the downward transmittance.  They differ
only in what they sweep and in the last line of algebra, so everything
else is declared once here.
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
    """A Smart-G sampler over an RTLS surface, normalised by the scene.

    Declaring one
    -------------
    Beyond what :class:`AtmoSampler` asks for, a subclass implements
    :meth:`_normalise`, which receives the raw sweep output for one band
    and that band's dataset and returns the 5S term.

    The kernel weights travel as three floats rather than as a built
    Smart-G surface.  A surface object in ``__init__`` would land in the
    cache key through :meth:`SceneModule._config_dict`, where nothing
    guarantees it hashes the same across two processes: two different
    BRDFs could then share one entry.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters.
    geo_config : GeoConfig
        Observation geometry.  ``saa`` sets the absolute frame and
        ``vaa`` the relative azimuth the BRDF responds to.
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    k0 : float, optional
        Spectral albedo of the isotropic RTLS kernel.  ``k1p = k2p = 0``
        with ``k0 = 1`` is a Lambertian surface of albedo one, which is
        how the result is checked against the Lambertian samplers.
    k1p : float, optional
        Weight of the geometric kernel, relative to the isotropic one.
    k2p : float, optional
        Weight of the volumetric kernel, relative to the isotropic one.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier.
    n_ph : int or None, optional
        Photons per Smart-G call and per sensor.  ``None`` uses
        ``default_n_ph``.
    n_ph_tdif_down : int or None, optional
        Photons for ``tdif_down`` when the scene does not already carry
        it.  ``None`` matches *n_ph*: the result is a ratio, so its
        relative error is the quadrature sum of the two, and precision
        spent on only one side is wasted.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    rename : dict[str, str] or None, optional
        Slot names to read and write instead of the declared roles.
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
