"""Frozen PSF kernels, stored as an ``xr.DataTree``.

One group per band, named after ``band.id``, holds ``kernel`` and one
variable per fitted parameter.  Groups need not share grids or sweep
dimensions, and the tree round-trips through zarr::

    <DataTree>
    ├── B02
    │       kernel  (y_psf, x_psf)
    │       sigma   ()
    └── B03
            kernel  (aot, y_psf, x_psf)
            sigma   (aot)

Functions
---------
    psf_tree
        Tree from per-band kernels and parameters.
    freeze
        Tree from the current kernels of live PSF modules.
    psf_kernel
        Kernel of one band, with its parameters in attrs when scalar.
    psf_params
        Fitted parameters of one band.
    tree_band_ids
        Band ids a tree holds.
    write_band
        Write one band group to zarr, one chunk per combo.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from ._psf import PSFModule
from .bands import SensorBand

__all__ = [
    "PSF_KERNEL",
    "freeze",
    "psf_kernel",
    "psf_params",
    "psf_tree",
    "tree_band_ids",
    "write_band",
]

#: Name of the kernel variable inside each band group.
PSF_KERNEL = "kernel"

# Dimensions that belong to the kernel itself rather than to the sweep it
# was optimised over.
_SPATIAL_DIMS = ("y_psf", "x_psf")


def psf_tree(
    kernels: dict[SensorBand, xr.DataArray],
    params: dict[SensorBand, dict[str, xr.DataArray]] | None = None,
) -> xr.DataTree:
    """Return the tree of per-band *kernels*, with their fitted *params*.

    Kernels keep any sweep dims beside ``(y_psf, x_psf)``; *params* is
    e.g. ``{B02: {"sigma": DataArray(aot)}}``.
    """
    groups: dict[str, xr.Dataset] = {}
    for band, kernel in kernels.items():
        variables: dict[str, xr.DataArray] = {PSF_KERNEL: kernel}
        if params and band in params:
            variables.update(params[band])
        groups[f"/{band.id}"] = xr.Dataset(variables)
    return xr.DataTree.from_dict(groups)


def freeze(modules: dict[SensorBand, PSFModule]) -> xr.DataTree:
    """Return the tree of the current kernels of live *modules*."""
    return psf_tree({band: psf.to_dataarray() for band, psf in modules.items()})


def psf_kernel(tree: xr.DataTree, band: SensorBand) -> xr.DataArray:
    """Return the kernel stored for *band*.

    Raises
    ------
    KeyError
        If the tree holds no group for *band*.
    """
    try:
        group = tree[band.id]
    except KeyError:
        known = ", ".join(sorted(tree_band_ids(tree))) or "none"
        raise KeyError(
            f"No PSF for band {band.id!r} in this tree. Holds: {known}."
        ) from None
    kernel: xr.DataArray = group.ds[PSF_KERNEL].copy()

    # Restore the fitted parameters only for a single combo, where each is
    # one value (possibly on length-one dims): plane normalisation needs them.
    fitted = {
        name: float(array.values.reshape(()))
        for name, array in group.ds.data_vars.items()
        if name != PSF_KERNEL and array.size == 1
    }
    if fitted and len(fitted) == len(group.ds.data_vars) - 1:
        kernel.attrs["adjeff:params"] = fitted
    return kernel


def psf_params(tree: xr.DataTree, band: SensorBand) -> dict[str, xr.DataArray]:
    """Return the fitted parameters of *band*, with their sweep dims; ``{}`` if none."""
    dataset = tree[band.id].ds
    return {
        str(name): dataset[name] for name in dataset.data_vars if name != PSF_KERNEL
    }


def tree_band_ids(tree: xr.DataTree) -> list[str]:
    """Return the band ids the tree holds, sorted."""
    return sorted(tree.children)


def write_band(dest: Path, dataset: xr.Dataset) -> None:
    """Write one band group to zarr at *dest*, one chunk per combo."""
    dims = {str(d) for da in dataset.data_vars.values() for d in da.dims}
    chunks = {d: 1 if d not in _SPATIAL_DIMS else -1 for d in dims}
    dataset.drop_encoding().chunk(chunks).to_zarr(dest, mode="w")
