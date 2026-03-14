"""Define a configuration for atmospheric parameters."""

from __future__ import annotations

from math import isclose
from typing import Annotated, Self

import xarray as xr
from pydantic.functional_validators import BeforeValidator as Before
from pydantic.functional_validators import model_validator

from ._config import _Config, to_arr


class AtmoConfig(_Config):
    """Pydantic model for the atmosphere parameters.

    Parameters
    ----------
    aot : xr.DataArray
        Aerosol optical thickness.
    h : xr.DataArray
        Ground elevation [km].
    rh : xr.DataArray
        Relative humidity [%].
    href : xr.DataArray
        Reference height of the exponential aerosol vertical profile.
    wl : xr.DataArray
        Wavelength [nm].
    species : dict[str, float]
        Dictionnary of species concentrations.
    """

    aot: Annotated[xr.DataArray, Before(to_arr("aot", ge=0.0))]
    h: Annotated[xr.DataArray, Before(to_arr("h", ge=0.0, le=9.0))]
    rh: Annotated[xr.DataArray, Before(to_arr("rh", ge=0.0, le=100.0))]
    href: Annotated[xr.DataArray, Before(to_arr("href", ge=0.1))]
    wl: Annotated[xr.DataArray, Before(to_arr("wl", ge=300.0))]
    species: dict[str, float]

    @model_validator(mode="after")
    def check_species_concentrations(self) -> Self:
        """Ensure that the species mix is equal to 1.0."""
        prop_sum: float = sum(self.species.values())
        if not isclose(prop_sum, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"Species proportions should be 1.0, not {prop_sum}."
            )
        return self
