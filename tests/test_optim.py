"""Tests for adjeff.optim (Loss, metrics, training set)."""

import pytest

from adjeff.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_loss_invalid_mask_on_raises():
    """Loss raises ConfigurationError for an invalid mask_on value."""
    from adjeff.optim import Loss, Metric

    with pytest.raises(ConfigurationError, match="mask_on"):
        Loss(metric=Metric.MSE_RAD, mask_on="invalid")
