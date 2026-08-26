"""Adjeff exceptions hierarchy."""


class AdjeffError(Exception):
    """Base exception for all Adjeff errors."""


class MissingVariableError(AdjeffError):
    """Required variable missing from one or more band Datasets."""


class AdjeffAccessorError(AdjeffError):
    """An error occurred during a call to the xarray adjeff accessor."""


class ConfigurationError(AdjeffError):
    """Invalid or missing configuration parameters."""


class ImageIOError(AdjeffError):
    """Satellite image or product file read failure."""


class ComputationError(AdjeffError):
    """A module produced a result that cannot be used.

    Raised before anything is written to the cache, so that a bad result
    is a failure at the place that caused it rather than a value later
    runs keep reading back.
    """


class OptimizationWarning(UserWarning):
    """PSF optimisation did not converge or terminated early."""
