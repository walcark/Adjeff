"""Generic pipeline for chaining SceneModule instances."""

from __future__ import annotations

import itertools

import xarray as xr

from adjeff.core import ImageDict
from adjeff.core.bands import SensorBand

from .scene_module import SceneModule


class Pipeline:
    """Ordered sequence of SceneModule instances applied to an ImageDict.

    Validates at construction that inter-module dependencies are satisfied
    (variables produced by prior modules are available when required).
    Variables not produced by any module are assumed to come from the input
    scene and are not checked at construction time.

    Parameters
    ----------
    modules : list[SceneModule]
        Modules to chain in order.
    stream_dims : dict[str, int] or None
        Dimensions to stream over for memory management.  When provided,
        the pipeline iterates over chunks of these dimensions (slicing the
        input scene) and concatenates results.  Effective when the
        dimension already exists in the input scene's DataArrays.

        Example: ``{"aot": 3}`` processes 3 AOT values at a time through
        the whole chain — useful when xarray modules like
        :class:`~adjeff.modules.classic.Toa2Unif` would otherwise receive
        arrays too large to fit in memory.
    """

    def __init__(
        self,
        modules: list[SceneModule],
        stream_dims: dict[str, int] | None = None,
    ) -> None:
        self._modules = list(modules)
        self._stream_dims = stream_dims or {}
        self._validate_chain()

    def _validate_chain(self) -> None:
        """Check inter-module variable dependencies.

        Only raises when a variable required by a module could have been
        produced by a prior module in this pipeline but was not.  Variables
        produced by no module in the pipeline are assumed to come from the
        input scene.
        """
        all_produced = {v for m in self._modules for v in m.output_vars}
        produced: set[str] = set()
        for mod in self._modules:
            pipeline_missing = (
                set(mod.required_vars) & all_produced
            ) - produced
            if pipeline_missing:
                raise ValueError(
                    f"{type(mod).__name__} requires "
                    f"{sorted(pipeline_missing)!r}, "
                    "not produced by any prior module."
                )
            produced.update(mod.output_vars)

    @property
    def required_vars(self) -> list[str]:
        """Variables that must be present in the input scene.

        These are variables required by at least one module that are not
        produced by any prior module in this pipeline.
        """
        produced: set[str] = set()
        needed: list[str] = []
        for mod in self._modules:
            for var in mod.required_vars:
                if var not in produced and var not in needed:
                    needed.append(var)
            produced.update(mod.output_vars)
        return needed

    @property
    def output_vars(self) -> list[str]:
        """All variables produced by the pipeline, in declaration order."""
        result: list[str] = []
        for mod in self._modules:
            result.extend(mod.output_vars)
        return result

    def __call__(self, scene: ImageDict) -> ImageDict:
        """Apply all modules in order and return the enriched scene.

        When ``stream_dims`` is configured and the corresponding dimensions
        are present in the scene's DataArrays, the pipeline iterates over
        chunks of those dimensions and concatenates results.

        Parameters
        ----------
        scene : ImageDict
            Input scene. Each module receives the output of the previous.

        Returns
        -------
        ImageDict
            Scene enriched with all pipeline output variables.
        """
        if self._stream_dims:
            return self._call_streaming(scene)
        return self._call_full(scene)

    def _call_full(self, scene: ImageDict) -> ImageDict:
        for mod in self._modules:
            scene = mod(scene)
        return scene

    def _call_streaming(self, scene: ImageDict) -> ImageDict:
        """Run the pipeline on chunks of stream_dims present in the scene."""
        # Collect which stream dims actually exist in scene DataArrays
        present: dict[str, int] = {}
        for band in scene.bands:
            for da in scene[band].data_vars.values():
                for dim, chunk_size in self._stream_dims.items():
                    if dim in da.dims and dim not in present:
                        present[dim] = da.sizes[dim]

        if not present:
            return self._call_full(scene)

        # Build slice specs for each present dim
        chunk_specs: list[tuple[str, list[slice]]] = [
            (
                dim,
                [
                    slice(i, min(i + self._stream_dims[dim], size))
                    for i in range(0, size, self._stream_dims[dim])
                ],
            )
            for dim, size in present.items()
        ]

        chunk_results: list[ImageDict] = []
        for combo in itertools.product(*[slices for _, slices in chunk_specs]):
            selector = {dim: slc for (dim, _), slc in zip(chunk_specs, combo)}
            sub_scene = ImageDict(
                {
                    band: scene[band].isel(
                        {
                            d: s
                            for d, s in selector.items()
                            if d in scene[band].dims
                        }
                    )
                    for band in scene.bands
                }
            )
            chunk_results.append(self._call_full(sub_scene))

        return self._concat_chunks(chunk_results, list(present))

    @staticmethod
    def _concat_chunks(chunks: list[ImageDict], dims: list[str]) -> ImageDict:
        """Concatenate pipeline outputs along the streamed dimensions."""
        bands = chunks[0].bands
        result: dict[SensorBand, xr.Dataset] = {}
        for band in bands:
            datasets = [c[band] for c in chunks]
            vars_out: dict[str, xr.DataArray] = {}
            for var_name in datasets[0].data_vars:
                var = str(var_name)
                arrays = [ds[var] for ds in datasets]
                stream_dim = next(
                    (d for d in dims if d in arrays[0].dims), None
                )
                if stream_dim is not None:
                    vars_out[var] = xr.concat(arrays, dim=stream_dim)
                else:
                    vars_out[var] = arrays[0]
            result[band] = xr.Dataset(vars_out)
        return ImageDict(result)
