"""Shared construction of the six radiative samplers.

They differ only in the term produced, the photon count, and which
geometry values are constants rather than sweep axes.

Classes
-------
    AtmoSampler
        SweepSampler over the atmospheric configuration.
"""

from typing import Any, ClassVar

import xarray as xr

import adjeff.atmosphere as atmo
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from ..sweep_sampler import SweepSampler


class AtmoSampler(SweepSampler):
    """A Smart-G sampler over an atmosphere and an observation geometry.

    Declaring one
    -------------
    Beyond ``contract`` and ``point_fn`` that :class:`SweepSampler`
    already asks for:

    ``_output_vars`` — the single variable produced.

    ``default_n_ph`` — photons per Smart-G call when the caller says
    nothing.  It differs by two orders of magnitude between quantities:
    a direct transmittance is an extinction along one line, a path
    reflectance is a scattering integral.

    ``geo_statics`` — the geometry arguments the point function takes as
    constants, in the order it declares them.  ``sza`` and ``vza`` are
    not among them: those are swept, and appear in ``contract``.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig or None
        Observation geometry, or ``None`` for a quantity that does not
        depend on one.
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier.
    n_ph : int or None, optional
        Photons per Smart-G call.  ``None`` uses ``default_n_ph``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    rename : dict[str, str] or None, optional
        Slot names to read and write instead of the declared roles.
    """

    _required_vars: ClassVar[list[str]] = []
    default_n_ph: ClassVar[int]
    geo_statics: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig | None,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int | None = None,
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rename: dict[str, str] | None = None,
    ) -> None:
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.spectral_config = spectral_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = self.default_n_ph if n_ph is None else n_ph
        super().__init__(cache=cache, batch_size=batch_size, dedup=dedup, rename=rename)

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        configs: tuple[ConfigProtocol, ...] = (
            self.spectral_config,
            self.atmo_config,
        )
        if self.geo_config is not None:
            configs = (*configs, self.geo_config)
        return configs

    def _statics(self) -> dict[str, Any]:
        statics: dict[str, Any] = {
            "species": self.atmo_config.species,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "n_ph": self.n_ph,
        }
        for name in self.geo_statics:
            statics[name] = self._geo_static(name)
        return statics

    def _geo_static(self, name: str) -> Any:
        """Return one geometry value Smart-G takes as a constant.

        An angle is stored as a DataArray so that it can be swept; a
        sampler that keeps it constant takes its first value, which is
        the only one a single Smart-G call can honour.
        """
        if self.geo_config is None:
            raise ValueError(
                f"{type(self).__name__} needs {name!r} but was built "
                "without a geo_config."
            )
        value = getattr(self.geo_config, name)
        if isinstance(value, xr.DataArray):
            return float(value.values.flat[0])
        return value
