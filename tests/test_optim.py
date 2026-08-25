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
