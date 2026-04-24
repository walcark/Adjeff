"""Base classes for atmospheric correction scene operations."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib  # type: ignore[import-untyped]
import structlog
import torch
import torch.nn as nn
import xarray as xr

from adjeff.utils import CacheStore
from adjeff.utils._config import _Config

if TYPE_CHECKING:
    from adjeff.core import ImageDict
    from adjeff.core._psf import PSFModule
    from adjeff.core.bands import SensorBand

logger = structlog.get_logger(__name__)


class SceneModule:
    """Base class for scene transforms on ImageDict.

    Subclasses must define two class attributes in order to work properly.

    1) required_vars: variables that must be in the input ImageDict. For
    instance, if required_vars = ["rho_s"], the following ImageDict:

    >>> ImageDict(
    >>>    S2Band.B02: ['rho_s', 'rho_toa']
    >>>    S2Band.B03: ['rho_s', 'rho_toa']
    >>>    S2Band.B04: ['rho_s', 'rho_toa']
    >>> )

    can be used as input, but not this one:

    >>> ImageDict(
    >>>    S2Band.B02: ['rho_unif', 'rho_toa']
    >>>    S2Band.B03: ['rho_unif', 'rho_toa']
    >>>    S2Band.B04: ['rho_unif', 'rho_toa']
    >>> )

    2) output_vars: variables that will be written by the SceneModule in
    the return ImageDict. For instance, if output_vars = ["rho_toa"], the
    input ImageDict:

    >>> ImageDict(
    >>>    S2Band.B02: ['rho_s']
    >>>    S2Band.B03: ['rho_s']
    >>>    S2Band.B04: ['rho_s']
    >>> )

    will produce an output of the following shape:

    >>> ImageDict(
    >>>    S2Band.B02: ['rho_unif', 'rho_toa']
    >>>    S2Band.B03: ['rho_unif', 'rho_toa']
    >>>    S2Band.B04: ['rho_unif', 'rho_toa']
    >>> )

    Each module produces the output with the forward method, called via
    ``__call__``.  This forward method calls self._compute() that must be
    defined in every subclass.

    Parameters
    ----------
    cache : CacheStore | None
        The cache used to store output variables for future usages.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = []

    def __init__(self, cache: CacheStore | None = None) -> None:
        super().__init__()
        self._cache = cache if cache is not None else CacheStore()
        self._log = logger.bind(module=type(self).__name__)

    def __call__(self, scene: "ImageDict") -> "ImageDict":
        """Apply the module to *scene*."""
        return self.forward(scene)

    def forward(self, scene: "ImageDict") -> "ImageDict":
        """Perform the module operations on the input ImageDict.

        The order of operations is the following:
        1) inputs are validated to ensure that all required parameters are
        present,
        2) cache is checked for inputs and the module configuration,
        3) self._compute() is called if the cache is not hit,
        4) results are stamped and eventually cached.
        """
        scene = scene.shallow_copy()
        scene.require_vars(self.required_vars)

        key = self._cache_key(scene)
        log = self._log.bind(key=key[:8])

        cached = self._cache.load_vars(key, scene.bands, self.output_vars)
        if cached is not None:
            for band, var_map in cached.items():
                ds = scene[band]
                for var_name, da in var_map.items():
                    ds[var_name] = da
            log.info("done", bands=[str(b) for b in scene.bands], cached=True)
            return scene

        scene = self._compute(scene)
        self._stamp_provenance(scene, key)
        self._cache.save_vars(key, scene, self.output_vars)
        log.info("done", bands=[str(b) for b in scene.bands], cached=False)
        return scene

    @abstractmethod
    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the core transform."""

    def _cache_key(self, scene: "ImageDict") -> str:
        """Hash the content of the SceneModule instance."""
        return str(
            joblib.hash(
                {
                    "module": type(self).__name__,
                    "config": self._config_dict(),
                    "inputs": self._input_hashes(scene),
                }
            )
        )

    # Parameters excluded from auto-detection: they're infrastructure, not
    # computation config (don't affect the output value for given inputs).
    # Note: deduplicate_dims is intentionally NOT excluded — it changes the
    # shape of the output (spatial broadcast vs parameter sweep dimensions),
    # so two runs with different deduplicate_dims must produce distinct cache
    # entries.
    _INFRA_PARAMS: ClassVar[frozenset[str]] = frozenset(
        ("self", "cache", "chunks")
    )

    def _config_dict(self) -> dict[str, object]:
        """Return frozen configuration for cache keying.

        Auto-detects public ``__init__`` parameters stored as same-named
        instance attributes, excluding infrastructure params (``cache``,
        ``chunks``, ``deduplicate_dims``).

        Subclasses with privately-stored params (e.g. ``_psf_dict``) must
        override this method.
        """
        sig = inspect.signature(type(self).__init__)
        raw = {
            name: getattr(self, name)
            for name in sig.parameters
            if name not in self._INFRA_PARAMS and hasattr(self, name)
        }
        return {
            k: v._stable_hash_repr if isinstance(v, _Config) else v
            for k, v in raw.items()
        }

    def _input_hashes(self, scene: "ImageDict") -> dict[str, str]:
        """Return a hash per (band, variable) pair in the input scene."""
        hashes: dict[str, str] = {}
        for band in scene.bands:
            ds = scene[band]
            for var in self.required_vars:
                da: xr.DataArray = ds[var]
                provenance_key: str | None = da.attrs.get(
                    "_adjeff_provenance", {}
                ).get("key")
                hash_val: str = str(
                    provenance_key
                    if provenance_key is not None
                    else joblib.hash(da.values)
                )
                hashes[f"{band}.{var}"] = hash_val
        return hashes

    def _stamp_provenance(self, scene: "ImageDict", key: str) -> None:
        """Store the provenance as attribute for each ImageDict's DataArray."""
        provenance = {"module": type(self).__name__, "key": key}
        for band in scene.bands:
            ds = scene[band]
            for var in self.output_vars:
                if var in ds:
                    ds[var].attrs["_adjeff_provenance"] = provenance


class TrainableSceneModule(nn.Module, SceneModule):
    """Abstract SceneModule with a differentiable per-band forward pass.

    Inherits from both :class:`torch.nn.Module` (for parameter registration
    and gradient flow) and :class:`SceneModule` (for the xarray pipeline
    contract).  When called as ``model(scene)``, the ``nn.Module.__call__``
    machinery is used (hooks fire, then ``forward`` is dispatched).

    Subclasses must implement :meth:`forward_band` and expose their
    per-band PSF modules via :attr:`psf_modules`.

    These two additions form the contract consumed by
    :class:`~adjeff.optim._Optimizer`.
    """

    def __init__(self, cache: CacheStore | None = None) -> None:
        nn.Module.__init__(self)
        SceneModule.__init__(self, cache=cache)

    def forward(self, scene: "ImageDict") -> "ImageDict":
        """Delegate to :meth:`SceneModule.forward` (resolves MRO ambiguity)."""
        return SceneModule.forward(self, scene)

    @property
    @abstractmethod
    def psf_modules(self) -> "dict[str, PSFModule]":
        """Mapping of band IDs to PSF modules."""

    @abstractmethod
    def forward_band(
        self,
        band: "SensorBand",
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable per-band forward pass for the training loop."""
