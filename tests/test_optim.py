"""Tests for adjeff.optim (Loss, metrics, training set, optimizers)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from adjeff.exceptions import ConfigurationError, OptimizationWarning

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _sample(n: int = 8):
    """Return one training sample on a small square grid."""
    from adjeff.optim.training_set import TrainingSample

    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float32),
        torch.arange(n, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.hypot(yy - n // 2, xx - n // 2)
    return TrainingSample(
        inputs={"rho_unif": torch.rand(n, n)},
        target=torch.rand(n, n),
        dist=dist,
        weight=1.0,
    )


def test_loss_unknown_mask_variable_names_what_is_available():
    """An unknown mask variable must say what the sample does carry.

    The name is no longer restricted to `rho_unif`, so a typo can only
    be caught when the training data is in hand.
    """
    from adjeff.optim import Loss, Metric

    loss = Loss(metric=Metric.MSE_RAD, mask_on="rho_uniff")

    with pytest.raises(ConfigurationError, match="rho_unif"):
        loss._mask_for(_sample())


def test_loss_rejects_a_non_positive_radius():
    """A radius mask must be a positive number of kilometres."""
    from adjeff.optim import Loss, Metric

    with pytest.raises(ConfigurationError, match="radius"):
        Loss(metric=Metric.MSE_RAD, mask_on=0.0)


def test_loss_radius_mask_is_a_disc_that_does_not_move():
    """A radius mask depends on the grid alone, not on the prediction."""
    from adjeff.optim import Loss, Metric

    sample = _sample()
    mask = Loss(metric=Metric.MSE_RAD, mask_on=2.0)._mask_for(sample)

    assert mask is not None
    assert mask.dtype == torch.bool
    assert bool(mask[sample.dist <= 2.0].all())
    assert not bool(mask[sample.dist > 2.0].any())


def test_loss_variable_mask_is_the_field_itself():
    """A named mask hands the metric the field, which derives the cut."""
    from adjeff.optim import Loss, Metric

    sample = _sample()
    mask = Loss(metric=Metric.MSE_RAD, mask_on="rho_unif")._mask_for(sample)

    assert mask is sample.inputs["rho_unif"]


# ---------------------------------------------------------------------------
# LBFGSStage — OptimizationWarning on degenerated line search
# ---------------------------------------------------------------------------


def test_lbfgs_degenerated_line_search_warns():
    """LBFGSStage issues OptimizationWarning when L-BFGS line search fails."""
    from adjeff.core import S2Band
    from adjeff.optim import Loss, Metric
    from adjeff.optim.lbfgs_optimizer import LBFGSConfig, LBFGSStage

    config = LBFGSConfig(
        min_steps=0,
        max_steps=5,
        loss_relative_tolerance=1e-4,
        loss=Loss(Metric.MSE),
    )
    stage = LBFGSStage(config)

    # Minimal real parameter so LBFGS can be instantiated.
    param = nn.Parameter(torch.tensor(1.0))
    psf_mock = MagicMock(spec=nn.Module)
    psf_mock.parameters.return_value = [param]

    model = MagicMock()
    model.psf_modules = {S2Band.B02.id: psf_mock}

    with (
        pytest.warns(OptimizationWarning, match="degenerated"),
        patch("torch.optim.LBFGS.step", side_effect=IndexError("bracket collapse")),
        patch.object(
            stage, "_total_loss", return_value=torch.tensor(0.5, requires_grad=True)
        ),
    ):
        stage._run_combo(model, S2Band.B02, MagicMock(), "aot=0.1")


def test_radial_metric_accepts_a_ready_made_mask():
    """A boolean mask is used as is, a float field drives a CDF cut.

    The two are told apart by dtype, so a fixed radius and an energy
    fraction can share one parameter without a second argument.
    """
    from adjeff.optim import Metric

    sample = _sample(16)
    pred = sample.target + 0.1

    inside = sample.dist <= 3.0
    on_disc = float(
        Metric.MSE_RAD(pred, sample.target, sample.dist, inside)
    )
    everywhere = float(
        Metric.MSE_RAD(pred, sample.target, sample.dist, None)
    )

    # A constant offset gives the same mean square error either way, so
    # the mask changes which pixels are averaged, not the value.
    assert on_disc == pytest.approx(everywhere, rel=1e-5)

    # A defect confined outside the disc must be invisible to the mask.
    spoiled = pred.clone()
    spoiled[sample.dist > 5.0] += 10.0
    assert float(
        Metric.MSE_RAD(spoiled, sample.target, sample.dist, inside)
    ) == pytest.approx(on_disc, rel=1e-5)
    assert float(
        Metric.MSE_RAD(spoiled, sample.target, sample.dist, None)
    ) > 10.0 * everywhere
