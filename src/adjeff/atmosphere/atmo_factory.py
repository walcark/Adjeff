"""Functions to instantiate a multi-profile atmosphere."""

import numpy as np
import structlog
import xarray as xr
from luts.luts import MLUT  # type: ignore[import-untyped]
from smartg.atmosphere import AerOPAC, AtmAFGL

from adjeff.exceptions import MissingVariableError

logger = structlog.get_logger(__name__)


def create_atmosphere(
    atmo_params: dict[str, xr.DataArray],
    species: dict[str, float],
    afgl_type: str = "afgl_exp_h8km",
    remove_rayleigh: bool = False,
    wl_ref_nm: float = 560.0,
) -> MLUT:
    """Create a multi-profile Smart-G atmosphere from atmospheric parameters.

    Each parameter DataArray must share the same single named dimension
    (e.g. ``"index"``).  One :class:`~smartg.atmosphere.AtmAFGL` instance
    is built per element along that dimension; all instances are then merged
    into a single ``MLUT`` via ``multi_profiles``.

    Parameters
    ----------
    atmo_params : dict[str, xr.DataArray]
        Mapping of parameter names to 1-D DataArrays.  Required keys:
        ``"wl"`` (wavelength [nm]), ``"aot"`` (aerosol optical thickness),
        ``"rh"`` (relative humidity [%]), ``"h"`` (ground elevation [km]),
        ``"href"`` (reference height of the aerosol profile [km]).
    species : dict[str, float]
        OPAC aerosol species and their fractional contributions.
        Values must sum to 1.
    afgl_type : str, optional
        Identifier of the AFGL standard atmosphere profile file,
        by default ``"afgl_exp_h8km"``.
    remove_rayleigh : bool, optional
        If ``True``, Rayleigh optical depth is set to zero, by default
        ``False``.
    wl_ref_nm : float, optional
        Reference wavelength [nm] used to scale the aerosol optical
        thickness, by default 560.0.

    Returns
    -------
    MLUT
        A merged multi-profile Smart-G atmosphere ready for simulation.

    Raises
    ------
    MissingVariableError
        If any of the required keys is absent from *atmo_params*.
    ValueError
        If the DataArrays do not share a single common dimension or if
        their sizes differ.
    """
    from smartg.smartg import multi_profiles

    params_li = parse_params(atmo_params)
    grid, pfgrid = grids()

    all_atm = []
    for params in params_li:
        logger.debug(
            "Atmosphere LUT generation. %s",
            ", ".join(f"{n}={v}" for n, v in params.items()),
        )

        atm: MLUT = create_atmafgl(
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

    logger.debug("Merge all atmospheres with multi-profile.")
    return multi_profiles(all_atm)


def parse_params(params: dict[str, xr.DataArray]) -> list[dict[str, float]]:
    """Ensure all input parameters are present with a single same dim.

    Parameters
    ----------
    params : dict[str, xr.DataArray]
        Input atmospheric parameters.

    Returns
    -------
    list[dict[str, float]]
        List of attributes name and values.
    """
    mandatory: list[str] = ["aot", "rh", "wl", "href", "h"]

    # Check missing
    missing = [m for m in mandatory if m not in params]
    if missing:
        raise MissingVariableError(f"Missing parameters: {missing}")

    # Reference dimension
    first = next(iter(params.values()))
    if len(first.dims) != 1:
        raise ValueError("Each parameter must have exactly one dimension")

    dim = first.dims[0]
    size = first.sizes[dim]

    # Check consistency
    for name, arr in params.items():
        if arr.dims != (dim,):
            raise ValueError(f"{name} has dims {arr.dims}, expected {(dim,)}")
        if arr.sizes[dim] != size:
            raise ValueError(
                f"{name} has size {arr.sizes[dim]}, expected {size}"
            )

    # Build list of dicts
    result = []
    for i in range(size):
        result.append(
            {name: float(arr.data[i]) for name, arr in params.items()}
        )

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
) -> MLUT:
    """Calculate an AtmAFGL instance for a set of atmospheric parameters.

    Parameters
    ----------
    height : float
        Ground elevation [km].
    aot : float
        Aerosol optical thickness.
    rh : float
        Relative humidity [%].
    wl : float
        Wavelength [nm].
    zmix : float
        Typical height of aerosol in the atmosphere.
    species : dict[str, float]
        Proportion of aerosol species.
    grid : np.ndarray
        Vertical sampling grid for scalar optical properties in Smart-G.
    pfgrid : np.ndarray
        Vertical sampling grid for the scattering matrix in Smart-G.
    remove_rayleigh : bool
        Assumes no rayleigh scattering if set to True.
    afgl_type : str
        Filename for the AFGL profile.
    wl_ref : float
        Reference wavelength for the LUT computation [nm].

    Returns
    -------
    MLUT
        The multi-LUT representing the Smart-G atmosphere instance.
    """
    aer_mix: list[AerOPAC] = [
        AerOPAC(
            filename=aer,
            tau_ref=aot * prop,
            w_ref=wl_ref,
            rh_mix=rh,
            Z_mix=zmix,
        )
        for (aer, prop) in species.items()
    ]

    return AtmAFGL(
        atm_filename=afgl_type,
        comp=aer_mix,
        grid=grid,
        pfgrid=pfgrid,
        RH_cst=rh,
        P0=surface_pressure(height),
        tauR=0.0 if remove_rayleigh else None,
    ).calc(wl)


def surface_pressure(height: float) -> float:
    """Return the surface pressure for a ground elevation.

    Parameters
    ----------
    height : float
        Ground elevation [km].

    Returns
    -------
    float
        Surface pressure [hPa].
    """
    P0: float = 1013.25
    return float(P0 * (1.0 - 6.5 * height / 288.15) ** 5.255)


def grids() -> tuple[np.ndarray, np.ndarray]:
    """Construct vertical grids to sample optical properties in Smart-G.

    Grid is used to sample scalar optical properties, and pfgrid is used
    to sample the scattering matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The grid and pfgrid numpy arrays.
    """
    base = np.arange(10.0, -0.01, -0.25)
    grid = np.concatenate((np.linspace(100.0, 11.0, num=90), base))
    pfgrid = np.concatenate((np.array([100.0, 20.0]), base))
    return grid, pfgrid
