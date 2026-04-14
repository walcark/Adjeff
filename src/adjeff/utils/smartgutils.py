"""Utilitary methods to handle Smart-G related data."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from luts.luts import MLUT  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from smartg.smartg import Sensor


def make_sensors(
    angles: xr.DataArray,
    phi_scalar: float,
    posz: float,
    loc: str = "ATMOS",
) -> list["Sensor"]:
    """Build a list of Smart-G Sensor objects, one per angle value.

    Smart-G's Sensor only accepts scalar ``THDEG``/``PHDEG``, so multiple
    angles require a list of Sensor instances.

    Parameters
    ----------
    angles : xr.DataArray
        Zenith angles [°] to iterate over (``THDEG`` values).
    phi_scalar : float
        Azimuth angle [°] shared by all sensors (``PHDEG``).
    posz : float
        Sensor altitude [km] (``POSZ``).
    loc : str, optional
        Smart-G location flag, by default ``"ATMOS"``.

    Returns
    -------
    list[Sensor]
        One Smart-G ``Sensor`` instance per element in *angles*.
    """
    from smartg.smartg import Sensor

    thdeg = np.atleast_1d(angles.values)
    phi = np.full_like(thdeg, phi_scalar)
    return [
        Sensor(POSZ=posz, THDEG=float(th), PHDEG=float(ph), LOC=loc)
        for th, ph in zip(thdeg, phi)
    ]


def compute_optical_depth(atm: MLUT) -> xr.DataArray:
    """Read the optical depth from the output LUT of a Smartg.run() result.

    No number of photons needs to be specified because the simulation result
    is not important. The optical depth is calculated by the Atmosphere object
    and not the simulation process.

    Parameters
    ----------
    atm : MLUT
        The Atmosphere object for which the optical depth is computed.
    """
    from smartg.smartg import Smartg

    wl = atm.axes["wavelength"]

    smartg = Smartg(autoinit=False)
    res: xr.DataArray = smartg.run(
        wl=wl,
        atm=atm,
        NBPHOTONS=1000,
        NF=1000,
    )["OD_atm"].to_xarray()
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
    """Normalize a Smart-G output DataArray.

    This is necessary because Smart-G output sometimes has a global shape
    that is hard to guess. For instance, some dimensions may or may not be
    in the output results dimensions depending on the size of the parameters
    passes to ``Smartg.run()``. Operations to perform on the outputs are:

    1) squeezing: drop a dimension of size 1 that should not be in the output
    2) renaming: rename a dimension name (ex: Zenith angles -> vza)
    3) assigning coordinates: in order to keep track of the parameters used
       for the computations.
    4) expanding: expand the array with a new dimension, when Smart-G does
       not built the dimension for a parameter used in computations.

    Parameters
    ----------
    squeeze : list[str] | none [default=None]
        Dims to squeeze and drop if present (e.g. ``"Azimuth angles"``).
    rename : rename[str, str] | None [default=None]
        SmartG dim name → target name. Only applied if the source dim exists.
    coords : dict[str, np.ndarray | xr.DataArray] | None [default=None]
        Coordinates to assign after renaming.
    expand : dict[str, np.ndarray | xr.DataArray] | None [default=None]
        Target dim → values. Expands the dim if absent.
    """
    for dim in squeeze or []:
        if dim in res.dims:
            res = res.squeeze(dim=dim).drop_vars(dim)

    if rename:
        present = {src: tgt for src, tgt in rename.items() if src in res.dims}
        if present:
            res = res.rename(present)

    if coords:
        res = res.assign_coords(
            {k: v for k, v in coords.items() if k in res.dims}
        )

    for dim, values in (expand or {}).items():
        if dim not in res.dims:
            res = res.expand_dims({dim: values})

    return res
