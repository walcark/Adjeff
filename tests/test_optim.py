"""Tests for adjeff.optim (Loss, metrics, training set, optimizers)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from adjeff.exceptions import ConfigurationError, OptimizationWarning

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_loss_invalid_mask_on_raises():
    """Loss raises ConfigurationError for an invalid mask_on value."""
    from adjeff.optim import Loss, Metric

    with pytest.raises(ConfigurationError, match="mask_on"):
        Loss(metric=Metric.MSE_RAD, mask_on="invalid")


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


def test_residual_scale_ignores_the_prediction():
    """A defect far from the pixels under test must not shrink the error.

    The residual used to be divided by ``max(|pred|, |truth|)``, so a
    prediction going wrong anywhere raised the scale everywhere, and the
    error measured on untouched pixels fell with it.  The reference
    alone sets the scale, and it does not move while the optimiser
    searches.
    """
    from adjeff.optim import Metric

    sample = _sample(16)
    pred = sample.target + 0.1
    inside = sample.dist <= 3.0

    clean = float(Metric.MSE_RAD(pred, sample.target, sample.dist, None))

    spoiled = pred.clone()
    spoiled[sample.dist > 5.0] += 10.0
    scaled = float(Metric.MSE_RAD(spoiled, sample.target, sample.dist, None))

    # The defect is real, so the unmasked error must grow, never shrink.
    assert scaled > clean
    assert bool(torch.equal(pred[inside], spoiled[inside]))


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
