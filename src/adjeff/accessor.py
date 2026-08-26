"""Define the xarray DataArray accessor for adjeff.

Registers the ``adjeff`` accessor on ``xr.DataArray`` only. All metadata and
radial-analysis utilities operate directly on the array — no ``var`` argument,
no Dataset indirection.
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
    """Adjeff-specific utilities on :class:`xr.DataArray`.

    Available on every DataArray via ``da.adjeff.<method>()``.

    **Metadata** — read adjeff attributes stored on the DataArray:
    :meth:`kind`, :meth:`is_analytical`, :meth:`model`,
    :meth:`params`, :meth:`band`.

    **Spatial** — pixel size and count inferred from coordinates:
    :attr:`res`, :attr:`n`.

    **Radial analysis** — convert a 2-D image to a radial profile or
    reconstruct a field from a profile:
    :meth:`radial`, :meth:`transect`, :meth:`to_field`.

    **Tensor / quantisation** — convert to PyTorch or discretise:
    :meth:`to_tensor`, :attr:`dists`, :meth:`digitize`.
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
        """Return the band id this array was produced for, or None.

        The id rather than the :class:`~adjeff.core.SensorBand` itself:
        the attribute has to survive a zarr round-trip, and an enum is
        not JSON serialisable.
        """
        band_id = self._da.attrs.get("band")
        return str(band_id) if band_id is not None else None

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    def _x_coord_name(self) -> str:
        """Return the name of the x spatial coordinate.

        Tries ``"x"`` first, then ``"x_psf"`` as a fallback for PSF kernels.

        Raises
        ------
        AdjeffAccessorError
            If neither ``"x"`` nor ``"x_psf"`` is present in the coordinates.
        """
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
        """Radial profile of this DataArray.

        See :func:`adjeff.analysis.radial_profile`, which this spells.
        """
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
        """Sample values along a line through the centre at an azimuth.

        See :func:`adjeff.analysis.transect`, which this spells.
        """
        return transect(self._da, angle, center, n_points)

    def to_tensor(self) -> torch.Tensor:
        """Convert this DataArray to a float32 :class:`torch.Tensor`.

        Returns
        -------
        torch.Tensor
            Float32 tensor with the same shape as this DataArray.
        """
        return torch.from_numpy(self._da.values.astype(np.float32))

    def tidy(self) -> xr.DataArray:
        """Turn every length-one dimension into a scalar coordinate.

        A scalar in a configuration is coerced to an array of length one,
        so an output carries an ``aot``, ``rh``, ``h`` and ``href``
        dimension even when a single atmospheric state was simulated.
        This peels them off while keeping the values, so the array still
        says which state produced it.

        Selecting one value of a swept parameter needs no such thing:
        ``da.sel(aot=0.4)`` already removes the dimension and keeps the
        coordinate.  This is for the parameters that were never swept.

        Returns
        -------
        xr.DataArray
            The same data, with fewer dimensions and the same coordinates.

        Notes
        -----
        A tidied array is recombined with :func:`xarray.concat`, which
        promotes the scalar coordinate back to a dimension.  It is not
        recombined with :func:`xarray.merge` or
        :func:`xarray.combine_by_coords`, which need a dimension to align
        on: call :meth:`untidy` first.

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
        """Turn scalar coordinates back into dimensions of length one.

        The inverse of :meth:`tidy`, needed before an alignment that
        works on dimensions rather than on values.

        Parameters
        ----------
        *names : str
            Coordinates to expand.  Defaults to every scalar coordinate
            that is not already a dimension.

        Returns
        -------
        xr.DataArray
            The same data, with one dimension per named coordinate.
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
        """Per-pixel radial distances as a float32 :class:`torch.Tensor`.

        The distances are computed from the spatial coordinates relative to
        the array centre. The returned tensor has shape ``(ny, nx)``.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(ny, nx)``.
        """
        rr_np, _ = radial_distances(self._da, center=None)
        shape = (self._da.shape[-2], self._da.shape[-1])
        return torch.from_numpy(rr_np.reshape(shape))

    def digitize(self, n_bins: int) -> xr.DataArray:
        """Quantise the field into *n_bins* discrete levels.

        Levels are evenly spaced from the field minimum to the field
        maximum.  Each pixel is mapped to the nearest level, producing
        a DataArray with at most *n_bins* distinct values.

        Parameters
        ----------
        n_bins : int
            Number of discrete output levels.

        Returns
        -------
        xr.DataArray
            Quantised DataArray with the same shape, dims, coords, and
            attrs as the original.
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
        """Reconstruct a field from this radial profile.

        See :func:`adjeff.analysis.to_field`, which this spells.
        """
        return to_field(self._da, target_ds)
