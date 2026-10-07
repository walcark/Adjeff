"""Chain of scene modules.

Classes
-------
    Pipeline
        Runs modules in order, checking at construction that each
        input is produced upstream; optionally streams over dimensions.
"""

from __future__ import annotations

import itertools

import xarray as xr

from adjeff.core import ImageDict
from adjeff.core.bands import SensorBand
from adjeff.exceptions import ConfigurationError

from .._logging import get_logger, run_context, timed
from .scene_module import SceneModule

logger = get_logger(__name__)


class Pipeline:
    """Scene modules applied in order.

    At construction, every variable a module reads that another module
    writes must be written upstream.  Other inputs are expected in the
    scene.

    Parameters
    ----------
    modules : list[SceneModule]
        Modules, in order.
    stream_dims : dict[str, int] or None, optional
        Run the chain on chunks of these dimensions and concatenate the
        results, to bound memory, e.g. ``{"aot": 3}``.
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
        """Raise if a module reads a pipeline output before it is written.

        Raises
        ------
        ConfigurationError
            Naming the module and the variables.
        """
        all_produced = {v for m in self._modules for v in m.output_vars}
        produced: set[str] = set()
        for mod in self._modules:
            pipeline_missing = (set(mod.required_vars) & all_produced) - produced
            if pipeline_missing:
                raise ConfigurationError(
                    f"{type(mod).__name__} requires "
                    f"{sorted(pipeline_missing)!r}, "
                    "not produced by any prior module."
                )
            produced.update(mod.output_vars)

    @property
    def required_vars(self) -> list[str]:
        """Variables that must be present in the input scene."""
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
        """Apply every module in order, streaming over ``stream_dims``."""
        if self._stream_dims:
            return self._call_streaming(scene)
        return self._call_full(scene)

    def _call_full(self, scene: ImageDict) -> ImageDict:
        """Apply all modules sequentially without chunking."""
        names = [type(m).__name__ for m in self._modules]
        with timed(logger, "pipeline", modules=len(names), chain=" ".join(names)):
            for position, mod in enumerate(self._modules, start=1):
                with run_context(stage=f"{position}/{len(names)}"):
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
                        {d: s for d, s in selector.items() if d in scene[band].dims}
                    )
                    for band in scene.bands
                }
            )
            chunk_results.append(self._call_full(sub_scene))

        return self._concat_chunks(
            chunk_results,
            [dim for dim, _ in chunk_specs],
            [len(slices) for _, slices in chunk_specs],
        )

    @staticmethod
    def _concat_chunks(
        chunks: list[ImageDict],
        dims: list[str],
        counts: list[int],
    ) -> ImageDict:
        """Fold the row-major grid of chunk results back into one scene.

        Folded one dimension at a time, innermost first, which is what a
        grid over several dimensions needs.
        """
        for dim, count in zip(reversed(dims), reversed(counts)):
            chunks = [
                Pipeline._concat_along(chunks[i : i + count], dim)
                for i in range(0, len(chunks), count)
            ]
        return chunks[0]

    @staticmethod
    def _concat_along(chunks: list[ImageDict], dim: str) -> ImageDict:
        """Concatenate *chunks* along *dim*, one band Dataset at a time.

        Variables that do not carry *dim* are taken from the first chunk.
        Dataset attributes are carried over: they hold the aerosol
        species written by :func:`~adjeff.api.load_scene`, which
        :func:`~adjeff.api.load_config` reads back later.
        """
        result: dict[SensorBand, xr.Dataset] = {}
        for band in chunks[0].bands:
            datasets = [c[band] for c in chunks]
            vars_out: dict[str, xr.DataArray] = {}
            for var_name in datasets[0].data_vars:
                var = str(var_name)
                arrays = [ds[var] for ds in datasets]
                if dim in arrays[0].dims:
                    vars_out[var] = xr.concat(arrays, dim=dim)
                else:
                    vars_out[var] = arrays[0]
            result[band] = xr.Dataset(vars_out, attrs=dict(datasets[0].attrs))
        return ImageDict(result)
