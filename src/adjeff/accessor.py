"""Define xarray Dataset accessor for adjeff.

Registers the ``adjeff`` accessor on xr.Dataset so that both ArrayDict and
PSFDict datasets share the same utility methods.

Utility methods are the following:
1) to_tensor(var) : register a variable as tensor
"""

from __future__ import annotations

import math

import numpy as np
import torch
import xarray as xr

from .utils.radial import bin_radial, natural_npix, radial_distances


@xr.register_dataset_accessor("adjeff")  # type: ignore[no-untyped-call]
class AdjeffAccessor:
    """Accessor providing adjeff-specific utilities on xr.Dataset.

    Available on every Dataset via ``ds.adjeff.<method>()``.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def to_tensor(
        self,
        var: str,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Extract *var* as a float32 torch.Tensor (zero-copy when possible).

        Parameters
        ----------
        var : str
            Dataset variable to extract.
        device : str or torch.device or None
            Target device (e.g. ``"cuda"``). Defaults to CPU.
        """
        t = torch.from_numpy(np.array(self._ds[var].values, dtype=np.float32))
        return t if device is None else t.to(device)

    def from_tensor(self, t: torch.Tensor, like: str, target: str) -> None:
        """Write tensor back as DataArray *target*, reusing coords from *like*.

        Parameters
        ----------
        t : torch.Tensor
            Tensor to store.
        like : str
            Existing variable whose coordinates are reused.
        target : str
            Name of the new variable to write.
        """
        ref = self._ds[like]
        arr = t.detach().cpu().numpy().astype(np.float32)
        self._ds[target] = xr.DataArray(arr, coords=ref.coords, dims=ref.dims)

    def apply_mask(self, mask: xr.DataArray) -> None:
        """Apply a mask (NaN out masked pixels) to all variables."""
        for var in self._ds.data_vars:
            self._ds[var] = self._ds[var].where(~mask)

    def kind(self, var: str) -> str | None:
        """Return the kind of model for *var*, or None if absent."""
        return self._ds[var].attrs.get("adjeff:kind")

    def is_analytical(self, var: str) -> bool:
        """Return True if *var* is analytical."""
        return self.kind(var) == "analytical"

    def model(self, var: str) -> str | None:
        """Return the analytical model of *var*, or None."""
        return self._ds[var].attrs.get("adjeff:model")

    def params(self, var: str) -> dict[str, object] | None:
        """Return parameters of *var*, or None."""
        return self._ds[var].attrs.get("adjeff:params")

    def radial(
        self,
        var: str,
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
    ) -> xr.DataArray:
        """Azimuthal mean of *var* as a function of radius.

        Bins at the natural pixel-sized resolution (no empty bins). When
        *n_bins* exceeds the natural maximum, the profile is upsampled via
        linear interpolation so the result is always gap-free.
        """
        rr_np, vv_np = radial_distances(self._ds, var, center)
        npix = natural_npix(self._ds, var)

        rr = torch.from_numpy(rr_np)
        vv = torch.from_numpy(vv_np)

        _, _, counts, sum_vals, r_centers = bin_radial(rr, vv, npix)

        val_mean = torch.full((npix,), float("nan"))
        mask = counts > 0
        val_mean[mask] = sum_vals[mask] / counts[mask]

        r_np = r_centers.numpy()
        v_np = val_mean.numpy()

        # Upsample via linear interpolation when more bins are requested
        if n_bins is not None and n_bins > npix:
            valid = ~np.isnan(v_np)
            r_np_new = np.linspace(r_np[0], r_np[-1], n_bins)
            v_np = np.interp(r_np_new, r_np[valid], v_np[valid])
            r_np = r_np_new

        return xr.DataArray(v_np, dims=["r"], coords={"r": r_np})

    def radial_cdf(
        self,
        var: str,
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
        normalize: bool = True,
    ) -> xr.DataArray:
        """Radial CDF of *var*, weighted by annulus area."""
        radial = self.radial(var, center, n_bins)
        r = torch.from_numpy(radial.coords["r"].values.astype(np.float32))
        f = torch.from_numpy(radial.values.astype(np.float32))

        f = torch.clamp(f, min=0.0)

        # Bin edges from bin centers
        dr = r[1:] - r[:-1]
        edges = torch.empty(r.numel() + 1, dtype=r.dtype)
        edges[1:-1] = 0.5 * (r[:-1] + r[1:])
        edges[0] = r[0] - 0.5 * dr[0]
        edges[-1] = r[-1] + 0.5 * dr[-1]

        # Annulus area weighting: π(r_out² - r_in²)
        area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        cdf = torch.cumsum(f * area, dim=0)

        if normalize and cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        return xr.DataArray(
            cdf.numpy(), dims=radial.dims, coords=radial.coords
        )

    def radial_std(
        self,
        var: str,
        center: tuple[float, float] | None = None,
        n_bins: int | None = None,
    ) -> xr.DataArray:
        """Azimuthal std per radial bin — departure from circular symmetry."""
        rr_np, vv_np = radial_distances(self._ds, var, center)
        npix = n_bins if n_bins is not None else natural_npix(self._ds, var)

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
            std.numpy(),
            dims=["r"],
            coords={"r": r_centers.numpy()},
        )
