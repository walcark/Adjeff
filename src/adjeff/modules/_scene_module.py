"""Base nn.Module for all atmospheric correction scene operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import joblib  # type: ignore[import-untyped]
import structlog
import torch.nn as nn
import xarray as xr

from adjeff.utils import CacheStore

if TYPE_CHECKING:
    from adjeff.core import ImageDict

logger = structlog.get_logger(__name__)


class SceneModule(nn.Module):
    """Base nn.Module for scene transforms on ImageDict.

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

    Each module produce the output with the forward method, as they inherit
    from nn.Module. This forward method calls self._compute() that must be
    defined in every subclasses.

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

    def forward(self, scene: "ImageDict") -> "ImageDict":
        """Perform the module operations of the input ImageDict.

        The order of operations is the following:
        1) inputs are validated to ensure that all required parameters are
        present,
        2) cache is checked for inputs and the module configuration,
        3) self._compute() is called if the cache is not hit,
        4) results are stamped and eventually cached.
        """
        module_name = type(self).__name__
        scene.require_vars(self.required_vars)

        key = self._cache_key(scene)
        log = logger.bind(module=module_name, key=key[:8])

        cached = self._cache.load_vars(key, scene.bands, self.output_vars)
        if cached is not None:
            log.info(
                "cache hit — skipping compute",
                bands=scene.bands,
                vars=self.output_vars,
            )
            for band, var_map in cached.items():
                ds = scene[band]
                for var_name, da in var_map.items():
                    ds[var_name] = da
            return scene

        log.info("Starting compute", bands=scene.bands)
        scene = self._compute(scene)
        log.info("Computation finished", vars=self.output_vars)
        self._stamp_provenance(scene, key)
        self._cache.save_vars(key, scene, self.output_vars)
        return scene

    @abstractmethod
    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the core transform."""

    def _cache_key(self, scene: "ImageDict") -> str:
        """Hash the content of the SceneModule instance.

        The hash is computed from the module name, the configuration e.g. input
        arguments of the __init__ method, and the dictionnary of input hashes
        from the input ImageDict taken by forward().
        """
        return str(
            joblib.hash(
                {
                    "module": type(self).__name__,
                    "config": self._config_dict(),
                    "inputs": self._input_hashes(scene),
                }
            )
        )

    def _config_dict(self) -> dict[str, object]:
        """Return frozen configuration for cache keying.

        The configuration is composed of all the input parameters used in the
        __init__ method.
        """
        return {}

    def _input_hashes(self, scene: "ImageDict") -> dict[str, str]:
        """Return the dictionnary mapping in band and variable to their hash.

        In an input ImageDict, all bands and all keys of the datasets are
        mapped to a unique DataArray. Two cases are treated:

        1) if the DataArray has been produced by another SceneModule, it has
        a provenance_key that can be used as a hash.
        2) if not, the content of the DataArray is hashed to produce the key.

        This enables to cache results from non-deterministic modules, as the
        results hash key is not produced directly from its inside values but
        by the provenance key of the input ImageDict.
        """
        hashes: dict[str, str] = {}
        for band in scene.bands:
            ds = scene[band]
            for var in self.required_vars:
                da: xr.DataArray = ds[var]
                provenance_key: str | None = da.attrs.get(
                    "_atcor_provenance", {}
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
                    ds[var].attrs["_atcor_provenance"] = provenance
