"""Define the xarray DataArray accessor for adjeff.

Registers the ``adjeff`` accessor on ``xr.DataArray`` only. All metadata and
radial-analysis utilities operate directly on the array — no ``var`` argument,
no Dataset indirection.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import xarray as xr

from .core.bands import SensorBand
from .utils.radial import (
    _profile_to_field,
    _sample_radial_from_cdf,
    bin_radial,
    natural_npix,
    radial_distances,
)


@xr.register_dataarray_accessor("adjeff")  # type: ignore[no-untyped-call]
class AdjeffDataArrayAccessor:
    """Accessor providing adjeff-specific utilities on ``xr.DataArray``.

    Available on every DataArray via ``da.adjeff.<method>()``.
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

    def band(self) -> SensorBand | None:
        """Return the ``band`` attribute, or None if absent."""
        return self._da.attrs.get("band")

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    def _x_coord_name(self) -> str:
        """Return the name of the x spatial coordinate.

        Tries ``"x"`` first, then ``"x_psf"`` as a fallback for PSF kernels.

        Raises
        ------
        ValueError
            If neither ``"x"`` nor ``"x_psf"`` is present in the coordinates.
        """
        for name in ("x", "x_psf"):
            if name in self._da.coords:
                return name
        raise ValueError(
            "No spatial x-coordinate found. "
            "Expected 'x' or 'x_psf' in coordinates."
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
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
    ) -> xr.DataArray:
        """Azimuthal mean as a function of radius.

        Bins at the natural pixel-sized resolution (no empty bins). When
        *n_bins* exceeds the natural maximum the profile is upsampled via
        linear interpolation so the result is always gap-free.

        Parameters
        ----------
        center : tuple[float, float] or None, optional
            ``(cx, cy)`` origin in coordinate units. Defaults to the
            coordinate mean.
        n_bins : int or None, optional
            Number of radial bins. When ``None``, natural bin count is used.

        Returns
        -------
        xr.DataArray
            1-D DataArray with dim ``"r"`` containing the azimuthal mean.
        """
        rr_np, vv_np = radial_distances(self._da, center=center)
        npix = natural_npix(self._da)

        rr = torch.from_numpy(rr_np)
        vv = torch.from_numpy(vv_np)

        _, _, counts, sum_vals, r_centers = bin_radial(rr, vv, npix)

        val_mean = torch.full((npix,), float("nan"))
        mask = counts > 0
        val_mean[mask] = sum_vals[mask] / counts[mask]

        r_np = r_centers.numpy()
        v_np = val_mean.numpy()

        if n_bins is not None and n_bins > npix:
            valid = ~np.isnan(v_np)
            r_np_new = np.linspace(r_np[0], r_np[-1], n_bins)
            v_np = np.interp(r_np_new, r_np[valid], v_np[valid])
            r_np = r_np_new

        return xr.DataArray(v_np, dims=["r"], coords={"r": r_np})

    def radial_cdf(
        self,
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
        normalize: bool = True,
    ) -> xr.DataArray:
        """Radial CDF, weighted by annulus area.

        Parameters
        ----------
        center : tuple[float, float] or None, optional
            ``(cx, cy)`` origin. Defaults to the coordinate mean.
        n_bins : int or None, optional
            Number of radial bins forwarded to :meth:`radial`.
        normalize : bool, optional
            If ``True`` (default), the CDF is normalised to ``[0, 1]``.

        Returns
        -------
        xr.DataArray
            1-D DataArray with dim ``"r"`` containing the cumulative
            area-weighted profile.
        """
        radial = self.radial(center, n_bins)
        r = torch.from_numpy(radial.coords["r"].values.astype(np.float32))
        f = torch.clamp(
            torch.from_numpy(radial.values.astype(np.float32)), min=0.0
        )

        dr = r[1:] - r[:-1]
        edges = torch.empty(r.numel() + 1, dtype=r.dtype)
        edges[1:-1] = 0.5 * (r[:-1] + r[1:])
        edges[0] = r[0] - 0.5 * dr[0]
        edges[-1] = r[-1] + 0.5 * dr[-1]

        area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        cdf = torch.cumsum(f * area, dim=0)

        if normalize and cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        return xr.DataArray(
            cdf.numpy(), dims=radial.dims, coords=radial.coords
        )

    def radial_std(
        self,
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
    ) -> xr.DataArray:
        """Azimuthal standard deviation per radial bin.

        Parameters
        ----------
        center : tuple[float, float] or None, optional
            ``(cx, cy)`` origin. Defaults to the coordinate mean.
        n_bins : int or None, optional
            Number of radial bins. When ``None``, natural bin count is used.

        Returns
        -------
        xr.DataArray
            1-D DataArray with dim ``"r"`` containing the per-bin std.
        """
        rr_np, vv_np = radial_distances(self._da, center=center)
        npix = n_bins if n_bins is not None else natural_npix(self._da)

        rr = torch.from_numpy(rr_np)
        vv = torch.from_numpy(vv_np)

        _, inds, counts, sum_vals, r_centers = bin_radial(rr, vv, npix)
        sum_sq = torch.bincount(inds, weights=vv**2, minlength=npix)

        std = torch.full((npix,), float("nan"))
        mask = counts > 0
        mean = sum_vals[mask] / counts[mask]
        std[mask] = torch.sqrt(
            (sum_sq[mask] / counts[mask] - mean**2).clamp(min=0.0)
        )

        return xr.DataArray(
            std.numpy(), dims=["r"], coords={"r": r_centers.numpy()}
        )

    def radial_adaptive(
        self,
        n: int,
        max_gap: float | None = None,
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
    ) -> xr.DataArray:
        """Radial profile resampled at *n* adaptively spaced radii.

        Sample positions are chosen via inverse-CDF of the profile's absolute
        gradient.  *max_gap* enforces a minimum spatial coverage by inserting
        uniform points wherever two consecutive samples exceed that distance.

        If the DataArray already has dim ``"r"`` (i.e. is itself a radial
        profile), the inverse-CDF sampling is applied directly without
        recomputing the azimuthal mean.

        Parameters
        ----------
        n : int
            Number of gradient-driven radial samples.
        max_gap : float or None, optional
            Maximum allowed gap between consecutive samples (coordinate units).
        center : tuple[float, float] or None, optional
            ``(cx, cy)`` origin. Defaults to the coordinate mean.
        n_bins : int or None, optional
            Number of radial bins for the intermediate profile.

        Returns
        -------
        xr.DataArray
            1-D DataArray with dim ``"r"`` at the adaptive sample positions.
        """
        if "r" in self._da.dims:
            profile = self._da
        else:
            profile = self.radial(center=center, n_bins=n_bins)

        r_vals = _sample_radial_from_cdf(profile, n, max_gap=max_gap)
        values = np.interp(r_vals, profile.coords["r"].values, profile.values)
        return xr.DataArray(values, dims=["r"], coords={"r": r_vals})

    def to_tensor(self) -> torch.Tensor:
        """Convert this DataArray to a float32 :class:`torch.Tensor`.

        Returns
        -------
        torch.Tensor
            Float32 tensor with the same shape as this DataArray.
        """
        return torch.from_numpy(self._da.values.astype(np.float32))

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
        """Return values binned in ``n_bins`` values.

        The values range from the minimum to the maximum of the field. The
        new values are produced with np.linspace.

        Parameters
        ----------
        n_bins : int
            Number of different values in the output DataArray.

        Returns
        -------
        xr.DataArray
            The binned DataArray.
        """
        data = self._da.data

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
        """Reconstruct a field from radial profile via Pchip interpolation.

        Interpolates the DataArray (dim ``"r"``) at the radial distances of
        every pixel in *target_ds*, broadcasting over all extra dimensions
        (e.g. ``aot``, ``wavelength``).

        Parameters
        ----------
        target_ds : xr.Dataset
            Dataset whose ``"x"`` and ``"y"`` coordinates define the output
            grid.

        Returns
        -------
        xr.DataArray
            DataArray with dims ``(..., "y", "x")`` on the target grid.
        """
        x = target_ds.coords["x"].values
        y = target_ds.coords["y"].values
        xx, yy = np.meshgrid(x, y)
        r = self._da.coords["r"].values

        def _interp(values: np.ndarray) -> np.ndarray:
            return _profile_to_field(r, values, xx, yy)

        result = xr.apply_ufunc(
            _interp,
            self._da,
            input_core_dims=[["r"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
        )
        return xr.DataArray(result.assign_coords(y=y, x=x))
