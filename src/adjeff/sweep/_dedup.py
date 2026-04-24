"""Spatial deduplication for parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class UniqueIndex:
    """Maps a multi-dimensional parameter grid to its unique row combinations.

    Build with :meth:`build`; call :meth:`expand` to map results computed
    on unique combinations back to the original grid.
    """

    inverse_map: xr.DataArray

    @classmethod
    def build(
        cls,
        arrays: dict[str, xr.DataArray],
        dims: list[str],
    ) -> tuple["UniqueIndex", dict[str, xr.DataArray]]:
        """Deduplicate *arrays* along *dims*.

        Return an index + reindexed arrays.

        Arrays carrying any dimension in *dims* are reduced to a 1-D
        ``"index"`` dimension of length ``n_unique``.  Arrays with no *dims*
        dimension pass through unchanged.

        If any array carries extra dimensions beyond *dims* (e.g.
        ``aot(x, y, t)`` with ``dims=["x", "y"]``), those extra dimensions
        are included in the uniqueness computation — two points that share
        the same ``(x, y)`` but differ in ``t`` are treated as distinct.

        Parameters
        ----------
        arrays
            Input arrays keyed by attribute name.
        dims
            Dimensions defining the deduplication space.

        Returns
        -------
        tuple[UniqueIndex, dict[str, xr.DataArray]]
            The index (for :meth:`expand`) and the reindexed arrays —
            involved ones on ``"index"`` dim, passthrough ones unchanged.

        Raises
        ------
        ValueError
            If no array carries any of the requested dimensions.
        """
        involved = {
            k: v for k, v in arrays.items() if any(d in v.dims for d in dims)
        }
        passthrough = {k: v for k, v in arrays.items() if k not in involved}

        if not involved:
            raise ValueError(f"No arrays carry any of dims {dims!r}.")

        # Strip target-dim coords before broadcasting to prevent xarray from
        # aligning arrays by label instead of by position when fields carry
        # different coordinate values on the same dimension.
        stripped = {
            k: v.drop_vars([d for d in dims if d in v.coords], errors="ignore")
            for k, v in involved.items()
        }
        bcasted = (
            dict(zip(stripped, xr.broadcast(*stripped.values())))
            if len(stripped) > 1
            else dict(stripped)
        )
        ref = next(iter(bcasted.values()))
        extra = [str(d) for d in ref.dims if d not in dims]
        eff_dims = [d for d in dims if d in ref.dims] + extra

        stacked = xr.Dataset(bcasted).stack(index=eff_dims)
        mat = np.stack([stacked[k].values for k in bcasted], axis=1)
        unique_rows, _, inv_idx = np.unique(
            mat, axis=0, return_index=True, return_inverse=True
        )

        inverse_map = xr.DataArray(
            inv_idx.reshape([ref.sizes[d] for d in eff_dims]),
            dims=eff_dims,
            coords={d: ref.coords[d] for d in eff_dims if d in ref.coords},
        )
        reindexed: dict[str, xr.DataArray] = {
            k: xr.DataArray(unique_rows[:, i], dims=["index"])
            for i, k in enumerate(bcasted)
        }
        return cls(inverse_map=inverse_map), {**reindexed, **passthrough}

    def expand(self, result: xr.DataArray) -> xr.DataArray:
        """Expand the ``"index"`` dimension back to the original grid."""
        return result.isel(index=self.inverse_map)
