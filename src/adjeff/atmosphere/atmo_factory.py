"""Multi-profile Smart-G atmosphere.

Functions
---------
    create_atmosphere
        Merged Smart-G atmosphere, one profile per parameter set.
    parse_params
        Check the parameters and split them into one dict per profile.
    create_atmafgl
        Smart-G profile of one parameter set.
    surface_pressure
        Surface pressure at a ground elevation.
    grids
        Vertical grids Smart-G samples the optical properties on.
"""

from typing import cast

import numpy as np
import xarray as xr

from adjeff.exceptions import ConfigurationError, MissingVariableError

from .._logging import get_logger

logger = get_logger(__name__)


def create_atmosphere(
    atmo_params: dict[str, xr.DataArray],
    species: dict[str, float],
    afgl_type: str = "afgl_exp_h8km",
    remove_rayleigh: bool = False,
    wl_ref_nm: float = 560.0,
) -> xr.Dataset:
    """Return a multi-profile Smart-G atmosphere.

    One profile is built per element along the dimension shared by the
    parameters, then all are merged with ``multi_profiles``.

    Parameters
    ----------
    atmo_params : dict[str, xr.DataArray]
        1-D arrays on one shared dimension, with keys ``"wl"`` [nm],
        ``"aot"``, ``"rh"`` [%], ``"h"`` [km] and ``"href"`` [km].
    species : dict[str, float]
        OPAC species and their fractions, summing to 1.
    afgl_type : str, optional
        AFGL standard atmosphere profile, ``"afgl_exp_h8km"`` by default.
    remove_rayleigh : bool, optional
        Set the Rayleigh optical thickness to zero.
    wl_ref_nm : float, optional
        Wavelength the AOT refers to [nm], 560 by default.

    Raises
    ------
    MissingVariableError
        If a required key is missing.
    ConfigurationError
        If the arrays do not share a single dimension of one size.
    """
    from smartg.smartg import multi_profiles

    params_li = parse_params(atmo_params)
    grid, pfgrid = grids()

    all_atm = []
    for params in params_li:
        logger.debug("atmosphere.build", **params)

        atm: xr.Dataset = create_atmafgl(
            height=params["h"],
            aot=params["aot"],
            rh=params["rh"],
            wl=params["wl"],
            zmix=params["href"],
            species=species,
            afgl_type=afgl_type,
            remove_rayleigh=remove_rayleigh,
            wl_ref=wl_ref_nm,
            grid=grid,
            pfgrid=pfgrid,
        )
        all_atm.append(atm)

    logger.debug("atmosphere.merge", profiles=len(all_atm))
    return cast(xr.Dataset, multi_profiles(all_atm))


def parse_params(params: dict[str, xr.DataArray]) -> list[dict[str, float]]:
    """Return one ``{name: value}`` dict per element of the shared dimension.

    Raises
    ------
    MissingVariableError
        If a required key is missing.
    ConfigurationError
        If the arrays do not share a single dimension of one size.
    """
    mandatory: list[str] = ["aot", "rh", "wl", "href", "h"]

    # Check missing
    missing = [m for m in mandatory if m not in params]
    if missing:
        raise MissingVariableError(f"Missing parameters: {missing}")

    # Reference dimension
    first = next(iter(params.values()))
    if len(first.dims) != 1:
        raise ConfigurationError("Each parameter must have exactly one dimension")

    dim = first.dims[0]
    size = first.sizes[dim]

    # Check consistency
    for name, arr in params.items():
        if arr.dims != (dim,):
            raise ConfigurationError(f"{name} has dims {arr.dims}, expected {(dim,)}")
        if arr.sizes[dim] != size:
            raise ConfigurationError(
                f"{name} has size {arr.sizes[dim]}, expected {size}"
            )

    # Build list of dicts
    result = []
    for i in range(size):
        result.append({name: float(arr.data[i]) for name, arr in params.items()})

    return result


def create_atmafgl(
    height: float,
    aot: float,
    rh: float,
    wl: float,
    species: dict[str, float],
    grid: np.ndarray,
    pfgrid: np.ndarray,
    zmix: float,
    remove_rayleigh: bool,
    afgl_type: str,
    wl_ref: float,
) -> xr.Dataset:
    """Return the Smart-G profile of one parameter set.

    *height*, *aot*, *rh*, *wl* and *zmix* are one value of ``h``,
    ``aot``, ``rh``, ``wl`` and ``href`` (see :func:`create_atmosphere`);
    *grid* and *pfgrid* come from :func:`grids`.
    """
    # Deferred like every other Smart-G import in the package: importing
    # smartg raises unless SMARTG_DIR_AUXDATA is set, and half of adjeff
    # never touches the radiative transfer at all.
    from smartg.atmosphere import AerOPAC, Atm1D

    aer_mix: list[AerOPAC] = [
        AerOPAC(
            fname=aer,
            tau_ref=aot * prop,
            w_ref=wl_ref,
            rh_mix=rh,
            z_mix=zmix,
        )
        for (aer, prop) in species.items()
    ]

    atm = Atm1D(
        fname=afgl_type,
        comp=aer_mix,
        grid=grid,
        pfgrid=pfgrid,
        rh_cst=rh,
        p0=surface_pressure(height),
        tau_r=0.0 if remove_rayleigh else None,
    ).calc(wl)
    return cast(xr.Dataset, atm)


def surface_pressure(height: float) -> float:
    """Return the standard-atmosphere surface pressure [hPa] at *height* [km]."""
    P0: float = 1013.25
    return float(P0 * (1.0 - 6.5 * height / 288.15) ** 5.255)


def grids() -> tuple[np.ndarray, np.ndarray]:
    """Return the vertical grids [km] of the optical properties and phase matrix."""
    base = np.arange(10.0, -0.01, -0.25)
    grid = np.concatenate((np.linspace(100.0, 11.0, num=90), base))
    pfgrid = np.concatenate((np.array([100.0, 20.0]), base))
    return grid, pfgrid
