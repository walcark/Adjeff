"""The ``.adjeff`` accessor of DataArrays.

Classes
-------
    AdjeffDataArrayAccessor
        adjeff metadata, grid, radial analysis and tensor conversion.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import xarray as xr

from .analysis import radial_profile, to_field, transect
from .exceptions import AdjeffAccessorError
from .utils.radial import (
    radial_distances,
)


@xr.register_dataarray_accessor("adjeff")  # type: ignore[no-untyped-call]
class AdjeffDataArrayAccessor:
    """``da.adjeff``: adjeff utilities on a DataArray.

    - Metadata: :meth:`kind`, :meth:`is_analytical`, :meth:`model`,
      :meth:`params`, :meth:`band_id`.
    - Grid: :attr:`res`, :attr:`n`, :attr:`dists`.
    - Radial analysis: :meth:`radial`, :meth:`transect`, :meth:`to_field`.
    - Dimensions: :meth:`tidy`, :meth:`untidy`.
    - Conversion: :meth:`to_tensor`, :meth:`digitize`.
    """

    def __init__(self, da: xr.DataArray) -> None:
        self._da = da

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def kind(self) -> str | None:
        """Return the ``adjeff:kind`` attribute, or None if absent."""
        return self._da.attrs.get("adjeff:kind")

    def is_analytical(self) -> bool:
        """Return True if this array is analytical."""
        return self.kind() == "analytical"

    def model(self) -> str | None:
        """Return the ``adjeff:model`` attribute, or None if absent."""
        return self._da.attrs.get("adjeff:model")

    def params(self) -> dict[str, object] | None:
        """Return the ``adjeff:params`` attribute, or None if absent."""
        return self._da.attrs.get("adjeff:params")

    def band_id(self) -> str | None:
        """Return the ``band`` attribute, a band id, or None if absent."""
        band_id = self._da.attrs.get("band")
        return str(band_id) if band_id is not None else None

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    def _x_coord_name(self) -> str:
        """Return ``"x"``, or ``"x_psf"`` for a kernel; raise if neither."""
        for name in ("x", "x_psf"):
            if name in self._da.coords:
                return name
        raise AdjeffAccessorError(
            "No spatial x-coordinate found. Expected 'x' or 'x_psf' in coordinates."
        )

    @property
    def res(self) -> float:
        """Pixel size in coordinate units, inferred from the x-coordinate."""
        x = self._da.coords[self._x_coord_name()].values
        return float(x[1] - x[0])

    @property
    def n(self) -> int:
        """Number of pixels on the x spatial dimension."""
        return len(self._da.coords[self._x_coord_name()])

    # ------------------------------------------------------------------
    # Radial analysis
    # ------------------------------------------------------------------

    def radial(
        self,
        stat: Literal["mean", "cdf", "std", "adaptive"] = "mean",
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
        *,
        normalize: bool = True,
        n: int | None = None,
        max_gap: float | None = None,
        symmetric: bool = False,
    ) -> xr.DataArray:
        """See :func:`adjeff.analysis.radial_profile`."""
        return radial_profile(
            self._da,
            stat,
            center,
            n_bins,
            normalize=normalize,
            n=n,
            max_gap=max_gap,
            symmetric=symmetric,
        )

    def transect(
        self,
        angle: float,
        center: tuple[float, float] | None = None,
        n_points: int | None = None,
    ) -> xr.DataArray:
        """See :func:`adjeff.analysis.transect`."""
        return transect(self._da, angle, center, n_points)

    def to_tensor(self) -> torch.Tensor:
        """Return the values as a float32 tensor of the same shape."""
        return torch.from_numpy(self._da.values.astype(np.float32))

    def tidy(self) -> xr.DataArray:
        """Turn every length-one dimension into a scalar coordinate.

        Recombine with :func:`xarray.concat`; call :meth:`untidy` before
        :func:`xarray.merge` or :func:`xarray.combine_by_coords`.

        Examples
        --------
        >>> rho_toa.dims
        ('sza', 'vza', 'aot', 'rh', 'href', 'h', 'y', 'x')
        >>> tidied = rho_toa.adjeff.tidy()
        >>> tidied.dims
        ('y', 'x')
        >>> float(tidied.aot)
        0.4
        """
        singleton = [str(dim) for dim in self._da.dims if self._da.sizes[dim] == 1]
        if not singleton:
            return self._da
        return self._da.squeeze(singleton, drop=False)

    def untidy(self, *names: str) -> xr.DataArray:
        """Turn the scalar coordinates *names*, all by default, back into dims.

        The inverse of :meth:`tidy`.
        """
        wanted = list(names) or [
            str(name)
            for name, coord in self._da.coords.items()
            if coord.ndim == 0 and name not in self._da.dims
        ]
        if not wanted:
            return self._da
        return self._da.expand_dims(wanted)

    @property
    def dists(self) -> torch.Tensor:
        """Distance of each pixel to the centre, as a float32 ``(ny, nx)`` tensor."""
        rr_np, _ = radial_distances(self._da, center=None)
        shape = (self._da.shape[-2], self._da.shape[-1])
        return torch.from_numpy(rr_np.reshape(shape))

    def digitize(self, n_bins: int) -> xr.DataArray:
        """Return the field snapped to the nearest of *n_bins* even levels.

        Levels span the field's minimum to maximum; dims, coords and
        attrs are kept.
        """
        data = np.asarray(self._da.data)

        mini, maxi = np.nanmin(data), np.nanmax(data)
        bins = np.linspace(mini, maxi, n_bins)
        edges = (bins[:-1] + bins[1:]) / 2
        idx = np.digitize(data, edges, right=False)

        return xr.DataArray(
            bins[idx],
            dims=self._da.dims,
            coords=self._da.coords,
            attrs=self._da.attrs,
        )

    def to_field(self, target_ds: xr.Dataset) -> xr.DataArray:
        """See :func:`adjeff.analysis.to_field`."""
        return to_field(self._da, target_ds)
