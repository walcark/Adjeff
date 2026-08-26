"""Tests for the loss and encircled-energy landscapes.

These run on the CPU: mapping a surface is a convolution per candidate,
never a Smart-G call.  The module sat at 19% coverage while carrying the
figures of Section 2 of the manuscript, and its docstring promised more
than it delivered.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from adjeff.core import GaussPSF, KingPSF, PSFGrid, S2Band
from adjeff.optim import (
    Loss,
    Metric,
    TrainingImages,
    energy_radius_landscape,
    loss_landscape,
)

BAND = S2Band.B03
RES_KM = 0.5
N = 33


@pytest.fixture
def scene():
    """Return one scene carrying everything Unif2Surface reads."""
    from adjeff.core import ImageDict, gaussian_image_dict

    images = gaussian_image_dict(
        sigma=2.0, res_km=RES_KM, bands=[BAND], n=N, var="rho_s"
    )
    ds = images[BAND]
    rho_s = ds["rho_s"]
    ds["rho_unif"] = rho_s * 0.9 + 0.01
    for name, value in (
        ("tdir_up", 0.7),
        ("tdif_up", 0.2),
        ("sph_alb", 0.1),
    ):
        ds[name] = xr.DataArray(value)
    return ImageDict({BAND: ds})


@pytest.fixture
def train_images(scene):
    """Return a single-scene training set."""
    return TrainingImages(images=[scene], weights=[1.0])


def _grid(sigmas):
    """Return one Gaussian PSF per sigma."""
    return [
        GaussPSF(PSFGrid(RES_KM, N), BAND, sigma=float(s)) for s in sigmas
    ]


# --- loss_landscape ---


def test_loss_landscape_returns_one_value_per_candidate(train_images):
    """The output is flat: the caller owns the parameter grid's shape."""
    losses = loss_landscape(
        train_images, BAND, _grid([0.5, 1.0, 2.0]), Loss(Metric.RMSE_RAD)
    )

    assert losses.shape == (3,)
    assert np.isfinite(losses).all()


def test_loss_landscape_accepts_any_callable_loss(train_images):
    """A loss is anything `fit` could call, not only the built-in one.

    The function used to read `loss.mask_on` and `loss.metric` off the
    object it was handed, which quietly restricted it to `Loss`.
    """

    def flat_mae(forward_fn, samples):
        """Unweighted, unmasked mean absolute error."""
        return torch.stack(
            [
                (forward_fn(s.inputs) - s.target).abs().mean() * s.weight
                for s in samples
            ]
        ).sum()

    losses = loss_landscape(train_images, BAND, _grid([0.5, 2.0]), flat_mae)

    assert losses.shape == (2,)
    assert np.isfinite(losses).all()


def test_loss_landscape_runs_through_a_model(train_images):
    """With a model, the variable names come from it rather than a list."""
    from adjeff.api import make_model
    from adjeff.modules.models import Unif2Surface

    model = make_model(
        Unif2Surface, GaussPSF, [BAND], RES_KM, N, {"sigma": 1.0},
        device="cpu",
    )

    through_model = loss_landscape(
        train_images,
        BAND,
        _grid([0.5, 2.0]),
        Loss(Metric.RMSE_RAD),
        model=model,
    )
    direct = loss_landscape(
        train_images, BAND, _grid([0.5, 2.0]), Loss(Metric.RMSE_RAD)
    )

    # Unif2Surface is exactly what the modelless path hardcodes, so the
    # two must agree to numerical noise.
    np.testing.assert_allclose(through_model, direct, rtol=1e-5)


def test_loss_landscape_is_empty_for_no_candidate(train_images):
    """An empty parameter grid is a valid, if useless, request."""
    assert loss_landscape(
        train_images, BAND, [], Loss(Metric.RMSE_RAD)
    ).shape == (0,)


# --- energy_radius_landscape ---


def test_energy_radii_grow_with_the_requested_fraction():
    """More energy means a larger circle, for every candidate."""
    radii = energy_radius_landscape(_grid([1.0, 2.0]), fractions=[0.1, 0.5, 0.9])

    assert set(radii) == {"EE10%", "EE50%", "EE90%"}
    assert (radii["EE10%"] < radii["EE50%"]).all()
    assert (radii["EE50%"] < radii["EE90%"]).all()


def test_energy_radii_grow_with_the_psf_width():
    """A wider Gaussian encircles its energy further out."""
    radii = energy_radius_landscape(_grid([0.5, 1.0, 2.0]))

    assert (np.diff(radii["EE50%"]) > 0).all()


def test_energy_radii_follow_the_king_tail():
    """A shallower King tail pushes the outer radius out.

    The 99% radius is the one that reads the tail, so it must separate
    two profiles that share a core width.
    """
    kings = [
        KingPSF(PSFGrid(RES_KM, N), BAND, sigma=1.0, gamma=g)
        for g in (1.1, 3.0)
    ]

    radii = energy_radius_landscape(kings, fractions=[0.99])

    assert radii["EE99%"][0] > radii["EE99%"][1]


def test_energy_radii_are_empty_for_no_candidate():
    """An empty list returns empty arrays, one per fraction."""
    radii = energy_radius_landscape([], fractions=[0.5])

    assert radii["EE50%"].shape == (0,)
