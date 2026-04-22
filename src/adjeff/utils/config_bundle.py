"""ConfigBundle: parameter aggregation, deduplication, and iteration engine.

A ConfigBundle allows to store multiple configuration classes e.g. pydantic
BaseModel with multiple varying parameters stored as xr.DataArray. It allows
to iterate over values within the DataArrays by specifying the attributes
that should vary value by value (scalars) or by chunks (vectors).

The ConfigBundle provides a wrapper ``apply()`` around a method that allows
to iterates over each useful attributes used in the method. To serve that
purpose, three different classes are used:

1) _FlatDim: a class storing the original / stacked dimensions of the scalars
attributes. When iterating over scalar attributes, they are broadcasted and
then stacked in a single dimension. This class allows to easily reverse the
stacking process.

2) _IndexMap: a class storing the result of a deduplication process in order
to easily reverse it. Deduplication is used when some attributes share one
or multiple dimensions that should be ``deduplicated`` e.g. for which only
unique tuple of attributes values should be computed.

3) ConfigBundle: the main class containing the apply method to easily sweep
over a set of attributes values.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import structlog
import xarray as xr

from ._config import ConfigProtocol

logger = structlog.get_logger(__name__)


@dataclass
class _FlatDim:
    """Stores everything needed to reverse one stacking operation.

    ``flat_name`` is the name of the stacked dimension created by
    ``xarray.stack``, formed by joining the original dimension names
    with double underscores (for example ``"x__y"``).

    ``orig_dims`` is the ordered list of dimension names that were
    stacked together.  ``sizes`` maps each of those names to its size
    before stacking.  ``coords`` maps each name to its coordinate array
    before stacking.
    """

    flat_name: str
    orig_dims: list[str]
    sizes: dict[str, int]
    coords: dict[str, Any]

    def unflatten(self, result: xr.DataArray) -> xr.DataArray:
        """Reverse the stacking operation on a DataArray.

        Parameters
        ----------
        result : xr.DataArray
            Array that may carry ``flat_name`` as one of its axes.

        Returns
        -------
        xr.DataArray
            Array with the flat dimension replaced by the original
            component dimensions.  If ``flat_name`` is absent or only
            one component dimension was stacked, *result* is returned
            unchanged.
        """
        if self.flat_name not in result.dims or len(self.orig_dims) == 1:
            return result
        ax = list(result.dims).index(self.flat_name)
        new_shape = (
            result.shape[:ax]
            + tuple(self.sizes[d] for d in self.orig_dims)
            + result.shape[ax + 1 :]
        )
        new_dims = (
            list(result.dims[:ax])
            + self.orig_dims
            + list(result.dims[ax + 1 :])
        )
        # Carry over all coords except flat dim, then restore originals.
        new_coords = {
            d: result.coords[d]
            for d in result.dims
            if d != self.flat_name and d in result.coords
        }
        new_coords.update(self.coords)
        return xr.DataArray(
            result.values.reshape(new_shape),
            dims=new_dims,
            coords=new_coords,
        )


@dataclass
class _IndexMap:
    """Stores the deduplication result and knows how to reverse it.

    Parameters
    ----------
    inverse_map : xr.DataArray
        Shaped like the original ``eff_dims`` space.  Each element
        holds the unique-row index for that coordinate combination.
        Use ``result.isel(index=inverse_map)`` to expand a
        deduplicated result back to the original spatial grid.
    unique_first : np.ndarray
        Position in the stacked ``eff_dims`` space of the first
        occurrence of each unique coordinate combination.
    eff_dims : list[str]
        Ordered list of dimension names that were deduplicated.
    """

    inverse_map: xr.DataArray
    unique_first: np.ndarray
    eff_dims: list[str]

    @classmethod
    def build(
        cls,
        involved: dict[str, xr.DataArray],
        eff_dims: list[str],
    ) -> tuple[_IndexMap, dict[str, xr.DataArray]]:
        """Find unique coordinate combinations and build the index map.

        All *involved* arrays are broadcast together so that extra
        dimensions beyond ``eff_dims`` (e.g. a time axis ``t`` in
        ``aot(x, y, t)`` when ``eff_dims=["x", "y"]``) are included
        in the uniqueness computation.  Two grid points that agree on
        every ``eff_dim`` coordinate but differ along an extra
        dimension produce separate unique entries.

        All broadcast arrays are reindexed from the uniqueness matrix
        and returned in the second element of the tuple.

        Parameters
        ----------
        involved : dict[str, xr.DataArray]
            Arrays whose duplicate rows should be collapsed.
        eff_dims : list[str]
            Dimensions to consider for uniqueness.

        Returns
        -------
        tuple[_IndexMap, dict[str, xr.DataArray]]
            The constructed index map and the reindexed pure arrays,
            now shaped ``(n_unique,)`` along an ``"index"`` dimension.
        """
        bcasted = (
            dict(zip(involved, xr.broadcast(*involved.values())))
            if len(involved) > 1
            else dict(involved)
        )
        ref = next(iter(bcasted.values()))
        broadcast_dims = list(ref.dims)
        # Keep eff_dims first, then any extra dims from wider shapes.
        extra = [str(d) for d in broadcast_dims if d not in eff_dims]
        used_dims = [d for d in eff_dims if d in broadcast_dims] + extra

        # Stack all arrays into a matrix and find the unique rows.
        stacked = xr.Dataset(bcasted).stack(index=used_dims)
        mat = np.stack([stacked[k].values for k in bcasted], axis=1)
        unique_rows, unique_first, inv_idx = np.unique(
            mat, axis=0, return_index=True, return_inverse=True
        )

        # inv_idx maps every stacked point to its unique row index.
        # Reshaped to the original grid it becomes the inverse map.
        inverse_map = xr.DataArray(
            inv_idx.reshape([ref.sizes[d] for d in used_dims]),
            dims=used_dims,
            coords={d: ref.coords[d] for d in used_dims if d in ref.coords},
        )
        reindexed = {
            name: xr.DataArray(unique_rows[:, i], dims=["index"])
            for i, name in enumerate(bcasted)
        }
        return cls(
            inverse_map=inverse_map,
            unique_first=unique_first,
            eff_dims=used_dims,
        ), reindexed

    def reindex_extra(self, da: xr.DataArray) -> xr.DataArray:
        """Reindex an array that carries eff_dims plus extra dimensions.

        The ``eff_dims`` are stacked into a temporary axis.  The rows
        corresponding to unique coordinate combinations are selected
        via ``unique_first``.  The result has an ``"index"`` dimension
        in place of the ``eff_dims`` while extra dimensions are
        preserved in their original order.

        Parameters
        ----------
        da : xr.DataArray
            Array carrying at least one ``eff_dim`` and any number of
            additional dimensions.

        Returns
        -------
        xr.DataArray
            Array with shape ``(n_unique, *extra_shape)``.
        """
        my_eff = [d for d in self.eff_dims if d in da.dims]
        extra = [d for d in da.dims if d not in self.eff_dims]
        stacked = da.stack(_eff=my_eff)
        if stacked.dims[0] != "_eff":
            stacked = stacked.transpose("_eff", *extra)
        selected = stacked.isel(_eff=self.unique_first)
        coords = {
            d: selected.coords[d].values for d in extra if d in selected.coords
        }
        coords["index"] = np.arange(len(self.unique_first))
        return xr.DataArray(
            selected.values, dims=["index"] + extra, coords=coords
        )

    def expand(self, result: xr.DataArray) -> xr.DataArray:
        """Expand the index dimension back to the original eff_dims grid."""
        return result.isel(index=self.inverse_map)


def _normalize(name: str, da: xr.DataArray) -> xr.DataArray:
    """Ensure a scalar DataArray is ready for Cartesian iteration.

    A 0-D array is promoted to 1-D with a single element. A 1-D array
    whose only dimension matches its name but has no coordinate labels
    gets those labels assigned from its values. All other arrays are
    returned unchanged.
    """
    if da.ndim == 0:
        v = float(da)
        return xr.DataArray([v], dims=[name], coords={name: [v]})
    if da.ndim == 1 and da.dims[0] == name and name not in da.coords:
        return da.assign_coords({name: da.values})
    return da


def _aggregate(
    configs: list[ConfigProtocol],
    scalars: list[str],
    vectors: list[str],
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Collect DataArrays and non-DataArray fields from all configs.

    For each attribute name the first config that provides it wins.
    Attributes not listed in *scalars* or *vectors* are ignored.
    Non-DataArray fields from ``model_fields`` are stored in the
    ``other`` dict on a first-seen basis.

    Parameters
    ----------
    configs : list[ConfigProtocol]
        Source configuration objects, searched in order.
    scalars : list[str]
        Attribute names expected to be found as DataArrays.
    vectors : list[str]
        Attribute names expected to be found as DataArrays.

    Returns
    -------
    tuple[dict[str, xr.DataArray], dict[str, Any]]
        DataArray attributes keyed by name, and non-DataArray fields
        keyed by name.

    Raises
    ------
    ValueError
        If any name from *scalars* or *vectors* is absent from all
        configs.
    """
    wanted = set(scalars) | set(vectors)
    das: dict[str, xr.DataArray] = {}
    other: dict[str, Any] = {}
    for cfg in configs:
        for name, da in cfg._arrays.items():
            if name in wanted and name not in das:
                das[name] = da
        for name, val in cfg._non_arrays.items():
            if name not in other:
                other[name] = val
    missing = wanted - set(das.keys())
    if missing:
        raise ValueError(
            f"Attribute(s) {sorted(missing)!r} listed in scalars/vectors "
            "were not found in any of the provided configs."
        )
    return das, other


def _deduplicate(
    das: dict[str, xr.DataArray],
    scalars: list[str],
    vectors: list[str],
    deduplicate_dims: list[str],
) -> tuple[dict[str, xr.DataArray], list[str], list[str], _IndexMap | None]:
    """Replace repeated coordinate combinations with a single index dim.

    First the effective dims are identified as those from
    *deduplicate_dims* that actually appear in at least one array.
    Then the involved arrays (those carrying at least one effective dim)
    are passed to :meth:`_IndexMap.build`, which computes unique
    coordinate combinations and returns reindexed pure arrays.  Arrays
    with extra dimensions beyond the effective dims are reindexed
    separately via :meth:`_IndexMap.reindex_extra`.

    **Demotion rule**: if at least one scalar carries the effective
    dims, every vector that also carries those dims is promoted to
    scalar so they are iterated point-by-point rather than passed whole.

    Parameters
    ----------
    das : dict[str, xr.DataArray]
        DataArrays keyed by attribute name.
    scalars : list[str]
        Names of scalar attributes.
    vectors : list[str]
        Names of vector attributes.
    deduplicate_dims : list[str]
        Dimensions to deduplicate jointly.

    Returns
    -------
    tuple[dict[str, xr.DataArray], list[str], list[str], _IndexMap | None]
        Updated arrays, scalar names, vector names, and the index map
        (``None`` when no arrays are affected).
    """
    eff_dims = [
        d for d in deduplicate_dims if any(d in da.dims for da in das.values())
    ]
    involved = {
        n: da for n, da in das.items() if any(d in da.dims for d in eff_dims)
    }
    if not involved:
        return das, scalars, vectors, None

    index_map, reindexed_pure = _IndexMap.build(involved, eff_dims)

    new_das = dict(das)
    new_das.update(reindexed_pure)
    for name, da in involved.items():
        if name not in reindexed_pure:
            new_das[name] = index_map.reindex_extra(da)

    # Promote vectors to scalars when a scalar already carries eff_dims.
    new_scalars, new_vectors, demoted = list(scalars), list(vectors), []
    if any(sn in involved for sn in scalars):
        for name in involved:
            if name in vectors:
                new_vectors.remove(name)
                new_scalars.append(name)
                demoted.append(name)

    logger.debug(
        "Deduplicated",
        eff_dims=eff_dims,
        n_original=index_map.inverse_map.size,
        n_unique=len(index_map.unique_first),
        reduction_factor=round(
            index_map.inverse_map.size / len(index_map.unique_first), 2
        ),
        demoted_to_scalar=demoted,
    )
    return new_das, new_scalars, new_vectors, index_map


def _flatten_scalars(
    das: dict[str, xr.DataArray],
    scalars: list[str],
) -> tuple[dict[str, xr.DataArray], list[_FlatDim]]:
    """Flatten scalar DataArrays to 1-D for Cartesian iteration.

    Vectors are left untouched and passed to the function in their
    original shape.

    Scalars that share the ``"index"`` dimension (introduced by
    deduplication) are broadcast together before stacking so they all
    land on the same compound dimension.  This is required for coherent
    Cartesian iteration over the deduplicated index.  All other
    multi-dimensional scalars are flattened individually.  The original
    shape information is stored in :class:`_FlatDim` objects so it can
    be restored by :meth:`ConfigBundle.reconstruct`.

    Parameters
    ----------
    das : dict[str, xr.DataArray]
        DataArrays keyed by attribute name.
    scalars : list[str]
        Names of scalar attributes.

    Returns
    -------
    tuple[dict[str, xr.DataArray], list[_FlatDim]]
        Updated arrays (scalars now 1-D) and a list of flat-dim
        descriptors for later unflattening.
    """
    new_das = dict(das)
    flat_dims: list[_FlatDim] = []

    dedup_scalars = {
        n: das[n] for n in scalars if n in das and "index" in das[n].dims
    }
    rest_scalars = {
        n: das[n] for n in scalars if n in das and "index" not in das[n].dims
    }

    if dedup_scalars and any(da.ndim > 1 for da in dedup_scalars.values()):
        bcasted = dict(
            zip(dedup_scalars, xr.broadcast(*dedup_scalars.values()))
        )
        ref = next(iter(bcasted.values()))
        dims = [str(d) for d in ref.dims]
        fd = _FlatDim(
            flat_name="__".join(dims),
            orig_dims=dims,
            sizes={d: ref.sizes[d] for d in dims},
            coords={d: ref.coords[d].values for d in dims if d in ref.coords},
        )
        flat_dims.append(fd)
        for name, da in bcasted.items():
            new_das[name] = da.stack({fd.flat_name: dims})

    for name, da in rest_scalars.items():
        da = _normalize(name, da)
        new_das[name] = da
        if da.ndim > 1:
            dims = [str(d) for d in da.dims]
            fd = _FlatDim(
                flat_name="__".join(dims),
                orig_dims=dims,
                sizes={d: da.sizes[d] for d in dims},
                coords={
                    d: da.coords[d].values for d in dims if d in da.coords
                },
            )
            flat_dims.append(fd)
            new_das[name] = da.stack({fd.flat_name: dims})

    return new_das, flat_dims


def _validate_chunks(
    chunks: dict[str, int],
    deduplicate_dims: list[str],
    index_map: _IndexMap | None,
    das: dict[str, xr.DataArray],
    vectors: list[str],
) -> None:
    """Check chunk keys for consistency with the deduplication outcome.

    A key naming a raw ``deduplicate_dim`` is invalid because that
    dimension no longer exists after deduplication: it was merged into
    ``"index"``.  Use ``"index"`` as the chunk key instead.
    Conversely, ``"index"`` is invalid when no deduplication was
    performed because the index dimension does not exist at all.  A key
    that does not match any dimension of any vector is also rejected so
    that misspelled keys are caught immediately.

    Parameters
    ----------
    chunks : dict[str, int]
        Requested chunk sizes, keyed by dimension name.
    deduplicate_dims : list[str]
        Dimensions that were (or will be) absorbed into ``"index"``.
    index_map : _IndexMap | None
        ``None`` when deduplication was not performed.
    das : dict[str, xr.DataArray]
        DataArrays keyed by attribute name.
    vectors : list[str]
        Names of vector attributes.

    Raises
    ------
    ValueError
        If any chunk key is inconsistent with the resolved dimensions.
    """
    absorbed = set(deduplicate_dims)
    vector_dims = {str(d) for n in vectors if n in das for d in das[n].dims}
    for key in chunks:
        if key in absorbed:
            raise ValueError(
                f"Chunk key '{key}' was absorbed into 'index' by "
                "deduplication. Use 'index' as the chunk key instead."
            )
        if key == "index" and index_map is None:
            raise ValueError(
                "Chunk key 'index' requires deduplication, but no "
                "deduplicate_dims were provided."
            )
        if key not in vector_dims:
            raise ValueError(
                f"Chunk key '{key}' does not match any dimension of the "
                f"declared vectors (found dims: {sorted(vector_dims)})."
            )


class ConfigBundle:
    """Aggregate config objects into a sweep-ready set and drive iteration.

    Construction runs four steps in sequence.

    **Aggregation** collects the DataArray attributes listed in
    *scalars* and *vectors* from all configs, taking the first
    occurrence of each name. Non-DataArray fields are stored in the
    :attr:`other` attribute.

    **Deduplication** (optional, enabled by *deduplicate_dims*)
    identifies repeated coordinate combinations across the named
    dimensions and replaces them with a single index dimension.  This
    avoids redundant computation when many grid points share the same
    parameter values.  Vectors that carry the same dims as at least one
    scalar are promoted to scalar at this stage.

    **Flattening** reduces every scalar DataArray to 1-D for Cartesian
    iteration.  Vectors remain in their original shape.

    **Chunk validation** checks that chunk keys are coherent with the
    resolved dimension names after deduplication.

    Parameters
    ----------
    configs : list[ConfigProtocol]
        Source configuration objects.
    scalars : list[str]
        Attribute names iterated point-by-point as a Cartesian product.
    vectors : list[str]
        Attribute names passed as whole DataArrays (or chunked slices)
        to the function.
    deduplicate_dims : list[str] | None, optional
        Dimensions to deduplicate jointly.
    chunks : dict[str, int] | None, optional
        Maps dimension names to chunk sizes for vector iteration.
        Multiple keys produce nested chunk loops.
    """

    def __init__(
        self,
        configs: list[ConfigProtocol],
        scalars: list[str],
        vectors: list[str],
        deduplicate_dims: list[str] | None = None,
        chunks: dict[str, int] | None = None,
    ) -> None:
        self._scalars: list[str] = list(scalars)
        self._vectors: list[str] = list(vectors)
        self._chunks: dict[str, int] = chunks or {}

        self._das, self.other = _aggregate(
            configs, self._scalars, self._vectors
        )
        logger.debug(
            "Aggregated ConfigBundle parameters..",
            scalars=list(self._scalars),
            vectors=list(self._vectors),
            das={k: list(v.dims) for k, v in self._das.items()},
            other=list(self.other.keys()),
        )

        self._index_map: _IndexMap | None = None
        if deduplicate_dims:
            self._das, self._scalars, self._vectors, self._index_map = (
                _deduplicate(
                    self._das, self._scalars, self._vectors, deduplicate_dims
                )
            )

        self._das, self._flat_dims = _flatten_scalars(self._das, self._scalars)
        _validate_chunks(
            self._chunks,
            deduplicate_dims or [],
            self._index_map,
            self._das,
            self._vectors,
        )

        logger.debug(
            "Initialization finshed.",
            arrays={k: list(v.dims) for k, v in self._das.items()},
            flat_dims=[fd.flat_name for fd in self._flat_dims],
            deduplicated=self._index_map is not None,
        )

    @property
    def arrays(self) -> dict[str, xr.DataArray]:
        """DataArrays ready for the sweep, keyed by attribute name."""
        return dict(self._das)

    @property
    def inverse_map(self) -> xr.DataArray | None:
        """Maps each index value back to its original coordinate combination.

        ``None`` when deduplication was not performed.  Use
        ``result.isel(index=bundle.inverse_map)`` to expand a
        deduplicated result back to the original spatial grid.
        """
        return (
            self._index_map.inverse_map
            if self._index_map is not None
            else None
        )

    def reconstruct(self, result: xr.DataArray) -> xr.DataArray:
        """Restore the original dimension layout of a bundle result.

        If the result has any compound (stacked) dimensions created
        during scalar flattening, they are reshaped back to their
        original multi-dimensional form.  If deduplication was
        performed, the ``"index"`` dimension is expanded back to the
        original spatial coordinates using the stored inverse map.

        Parameters
        ----------
        result : xr.DataArray
            Output of the sweep function, potentially with compound
            dimensions or an ``"index"`` dimension.

        Returns
        -------
        xr.DataArray
            Array with dimensions restored to the original layout.
        """
        for fd in self._flat_dims:
            result = fd.unflatten(result)
        if self._index_map is not None:
            result = self._index_map.expand(result)
        logger.debug(
            "Reconstructed original dimensions.",
            dims=list(result.dims),
            shape=result.shape,
        )
        return result

    def apply(
        self,
        func: Callable[..., xr.DataArray],
        **kwargs: Any,
    ) -> xr.DataArray:
        """Apply a function over the full parameter space.

        The outer loop iterates over the Cartesian product of all scalar
        dimensions.  At each step one value is selected from each scalar
        and coordinate information is recorded so that
        ``combine_by_coords`` can reassemble the outputs correctly.

        The inner loop iterates over all combinations of vector chunk
        slices as specified by *chunks*.  At each step the function is
        called with all matching parameters passed as keyword arguments,
        matched by name against the bundle's scalar and vector arrays.
        The function may declare its parameters in any order; only the
        names need to match.  Parameters not present in the bundle are
        left to *kwargs* or to the function's own defaults.

        Parameters
        ----------
        func : Callable[..., xr.DataArray]
            Function to call at each sweep point.  Its parameters are
            matched by name to the bundle scalars and vectors; it must
            return a DataArray whose dimensions are compatible with the
            vector arrangement it received.
        **kwargs : Any
            Extra keyword arguments forwarded to every call of *func*.
            Must not overlap with any scalar or vector name.

        Returns
        -------
        xr.DataArray
            Combined output over the full parameter space, with
            dimensions restored to the original layout.

        Raises
        ------
        TypeError
            If *func* returns a Dataset instead of a DataArray.
        ValueError
            If *kwargs* contains a key that also exists in the bundle
            scalars or vectors, or if *func* has a required parameter
            that is neither in the bundle nor in *kwargs*.
        """
        scalar_names = [n for n in self._scalars if n in self._das]
        scalar_das = [self._das[n] for n in scalar_names]

        sig = inspect.signature(func)
        _VAR_KINDS = {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        # Names of concrete (non-variadic) parameters declared by func.
        concrete_params = {
            k: p for k, p in sig.parameters.items() if p.kind not in _VAR_KINDS
        }

        # Detect kwargs keys that would shadow bundle parameters.
        all_bundle_names = set(self._scalars) | set(self._vectors)
        collisions = set(kwargs) & all_bundle_names
        if collisions:
            raise ValueError(
                f"kwargs key(s) {sorted(collisions)!r} collide with "
                "bundle scalar/vector names. Rename them or remove "
                "them from kwargs."
            )

        # Detect required parameters that cannot be satisfied.
        satisfiable = all_bundle_names | set(kwargs)
        missing_required = {
            k
            for k, p in concrete_params.items()
            if p.default is inspect.Parameter.empty and k not in satisfiable
        }
        if missing_required:
            raise ValueError(
                f"func has required parameter(s) {sorted(missing_required)!r} "
                "that are not provided by the bundle or kwargs."
            )

        if scalar_das:
            bcasted = dict(zip(scalar_names, xr.broadcast(*scalar_das)))
            ref = next(iter(bcasted.values()))
            dim_order = [str(d) for d in ref.dims]
            coord_vals = [ref.coords[d].values for d in dim_order]
            idx_ranges = [range(len(c)) for c in coord_vals]
        else:
            bcasted, dim_order, coord_vals, idx_ranges = {}, [], [], []

        # Compound dims (e.g. "index__t") carry MultiIndex tuple coords.
        # Decompose them into their original component dims so that
        # expand_dims receives plain scalar values and combine_by_coords
        # produces clean output.
        flat_dim_lookup = {fd.flat_name: fd for fd in self._flat_dims}

        all_results: list[xr.DataArray] = []

        n_steps = (
            int(np.prod([len(c) for c in coord_vals])) if coord_vals else 1
        )
        step_num = 0

        for idx_combo in (
            itertools.product(*idx_ranges) if idx_ranges else [()]
        ):
            step_num += 1
            step_coords: dict[str, Any] = {}
            for i, (d, j) in enumerate(zip(dim_order, idx_combo)):
                v = coord_vals[i][j]
                if d in flat_dim_lookup:
                    # v is a MultiIndex tuple: unpack into component dims
                    for orig_dim, component in zip(
                        flat_dim_lookup[d].orig_dims, v
                    ):
                        step_coords[orig_dim] = component
                else:
                    step_coords[d] = v

            logger.debug(
                "Sweep step",
                step=f"{step_num}/{n_steps}",
                coords={k: float(v) for k, v in step_coords.items()},
            )

            scalar_vals = [
                bcasted[name].isel(
                    {
                        d: int(j)
                        for d, j in zip(dim_order, idx_combo)
                        if d in bcasted[name].dims
                    }
                )
                for name in scalar_names
            ]
            for chunk in self._iter_vector_chunks():
                all_named = {**dict(zip(scalar_names, scalar_vals)), **chunk}
                bound = {
                    k: all_named[k] for k in concrete_params if k in all_named
                }
                out = func(**bound, **kwargs)

                for d, v in step_coords.items():
                    if d not in out.dims:
                        out = out.expand_dims({d: [v]})
                all_results.append(out)

        combined = xr.combine_by_coords(all_results)
        if not isinstance(combined, xr.DataArray):
            raise TypeError(
                "combine_by_coords returned a Dataset instead of a "
                "DataArray.  Make sure the function always returns an "
                "unnamed DataArray."
            )
        return self.reconstruct(combined)

    def _iter_vector_chunks(self) -> Iterator[dict[str, xr.DataArray]]:
        """Yield one dict of vector DataArrays per chunk combination.

        When no chunks are configured or no vectors are present, a
        single dict with the full vector arrays is yielded.  Otherwise,
        for each dimension listed in *chunks*, all arrays carrying that
        dimension are sliced into windows of the specified size.  When
        multiple chunk dimensions are given, all combinations of their
        windows are iterated as a nested loop.

        Yields
        ------
        dict[str, xr.DataArray]
            Mapping from vector name to the (possibly sliced) array.
        """
        vectors = {n: self._das[n] for n in self._vectors if n in self._das}
        if not vectors or not self._chunks:
            yield vectors
            return

        chunk_specs: list[tuple[str, list[slice]]] = []
        for dim_name, chunk_size in self._chunks.items():
            dim_size = next(
                (
                    da.sizes[dim_name]
                    for da in vectors.values()
                    if dim_name in da.dims
                ),
                None,
            )
            if dim_size is None:
                continue
            slices = [
                slice(s, min(s + chunk_size, dim_size))
                for s in range(0, dim_size, chunk_size)
            ]
            chunk_specs.append((dim_name, slices))

        if not chunk_specs:
            yield vectors
            return

        dim_names = [cs[0] for cs in chunk_specs]
        for combo in itertools.product(*[cs[1] for cs in chunk_specs]):
            chunk = dict(vectors)
            for dim_name, slc in zip(dim_names, combo):
                for name in vectors:
                    if dim_name in chunk[name].dims:
                        chunk[name] = chunk[name].isel({dim_name: slc})
            yield chunk
