"""Building Smart-G inputs and normalising its outputs.

Functions
---------
    make_sensors
        One sensor per zenith angle.
    compute_optical_depth
        Total optical depth of an atmosphere.
    adapt_smartg_output
        Squeeze, rename, label and expand a Smart-G output.
    pair_angles_with_points
        Keep, per point of a batched call, its own angle.
    collect_batched
        Batched Smart-G output, unstacked onto the swept dims.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .xrutils import ParamBatch

if TYPE_CHECKING:
    from smartg.sensor import Sensor


def make_sensors(
    angles: xr.DataArray,
    phi_scalar: float,
    posz: float,
    loc: str = "ATMOS",
) -> list["Sensor"]:
    """Return one Smart-G sensor per zenith angle [°]."""
    from smartg.sensor import Sensor

    thdeg = np.atleast_1d(angles.values)
    phi = np.full_like(thdeg, phi_scalar)
    return [
        Sensor(pos_z=posz, th_deg=float(th), ph_deg=float(ph), loc=loc)
        for th, ph in zip(thdeg, phi)
    ]


def compute_optical_depth(atm: xr.Dataset) -> xr.DataArray:
    """Return the total optical depth at the ground, by wavelength.

    A property of the atmosphere: the few photons of the run do not affect it.
    """
    from smartg.smartg import Smartg

    wl = atm["wavelength"]

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wavelength=wl,
        atmosphere=atm,
        n_photons=1000,
        n_icdf=1000,
    )["OD_atm"]
    smartg.clear_context()

    if len(wl) == 1 and "wavelength" not in res.dims:
        res = res.expand_dims(wavelength=wl)

    return res.sel(z_atm=0.0).drop_vars("z_atm")


def adapt_smartg_output(
    res: xr.DataArray,
    *,
    squeeze: list[str] | None = None,
    rename: dict[str, str] | None = None,
    coords: dict[str, np.ndarray | xr.DataArray] | None = None,
    expand: dict[str, np.ndarray | xr.DataArray] | None = None,
) -> xr.DataArray:
    """Normalise a Smart-G output, whose dims depend on its inputs.

    Applied in order, each only where it applies: *squeeze* drops
    length-one dims, *rename* renames dims, *coords* labels them, and
    *expand* adds the dims Smart-G left out.
    """
    for dim in squeeze or []:
        if dim in res.dims:
            # Smart-G labels these axes, but not always: a dim it built
            # without a coordinate has nothing to drop.
            res = res.squeeze(dim=dim).drop_vars(dim, errors="ignore")

    if rename:
        present = {src: tgt for src, tgt in rename.items() if src in res.dims}
        if present:
            res = res.rename(present)

    if coords:
        res = res.assign_coords({k: v for k, v in coords.items() if k in res.dims})

    for dim, values in (expand or {}).items():
        if dim not in res.dims:
            res = res.expand_dims({dim: values})

    return res


def pair_angles_with_points(res: xr.DataArray, *angles: str) -> xr.DataArray:
    """Keep, for each point of a batched call, the angle it asked for.

    Smart-G returns every angle for every point; only the diagonal is
    meaningful.  Outside a batch, *res* is returned unchanged.
    """
    if ParamBatch.GROUP_DIM not in res.dims:
        return res
    n = res.sizes[ParamBatch.GROUP_DIM]
    picks = {
        name: xr.DataArray(np.arange(n), dims=ParamBatch.GROUP_DIM)
        for name in angles
        if name in res.dims and res.sizes[name] == n
    }
    return res.isel(picks) if picks else res


def collect_batched(
    res: xr.DataArray,
    batch: ParamBatch,
    *,
    angles: Mapping[str, tuple[str, np.ndarray]] | None = None,
    drop: Sequence[str] = (),
) -> xr.DataArray:
    """Return a batched Smart-G output on the dims of the sweep contract.

    The ``"wavelength"`` axis of a batched call is the batch index: it
    is labelled as such and unstacked.

    Parameters
    ----------
    angles : Mapping[str, tuple[str, np.ndarray]] or None, optional
        Smart-G dim to ``(name, values)``, e.g.
        ``{"Zenith angles": ("vza", vza.values)}``; each is then paired
        against the points.
    drop : Sequence[str], optional
        Smart-G dims to squeeze, e.g. ``"Azimuth angles"``.
    """
    angles = angles or {}
    res = adapt_smartg_output(
        res,
        squeeze=list(drop),
        rename={
            **{src: name for src, (name, _) in angles.items()},
            "wavelength": "index",
        },
        coords={**dict(angles.values()), "index": batch.index_coord},
        expand=dict(angles.values()),
    )
    # A single atmospheric state leaves Smart-G no axis to return, so the
    # index has to be put back before it can be unstacked.
    if "index" not in res.dims:
        res = res.expand_dims(index=1).assign_coords(index=batch.index_coord)
    return pair_angles_with_points(
        batch.unstack(res), *(name for name, _ in angles.values())
    )
