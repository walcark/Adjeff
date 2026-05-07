"""Tests for adjeff.sweep (SweepBundle and UniqueIndex)."""

import numpy as np
import pytest
import xarray as xr

from adjeff.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# SweepBundle
# ---------------------------------------------------------------------------


def test_sweep_bundle_scalar_multidim_raises():
    """SweepBundle raises ConfigurationError for a scalar with ndim > 1."""
    from adjeff.sweep.bundle import SweepBundle

    da_2d = xr.DataArray(
        np.ones((2, 3)),
        dims=["aot", "rh"],
        coords={"aot": [0.1, 0.2], "rh": [50.0, 70.0, 90.0]},
    )
    with pytest.raises(ConfigurationError, match="dimensions"):
        SweepBundle(scalars={"aot": da_2d}, vectors={})


def test_aggregate_missing_name_raises():
    """_aggregate raises ConfigurationError when a requested name is absent."""
    from adjeff.sweep.bundle import _aggregate

    class FakeConfig:
        _arrays = {"aot": xr.DataArray([0.1], dims=["aot"])}
        _non_arrays: dict = {}

    with pytest.raises(ConfigurationError, match="not found in any config"):
        _aggregate([FakeConfig()], ["aot", "missing_field"])


# ---------------------------------------------------------------------------
# UniqueIndex
# ---------------------------------------------------------------------------


def test_unique_index_no_matching_dims_raises():
    """UniqueIndex.build raises ConfigurationError when no array has the dims."""
    from adjeff.sweep._dedup import UniqueIndex

    arrays = {"aot": xr.DataArray([0.1, 0.2], dims=["aot"])}
    with pytest.raises(ConfigurationError, match="No arrays carry any of dims"):
        UniqueIndex.build(arrays, dims=["x", "y"])
