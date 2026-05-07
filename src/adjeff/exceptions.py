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


class OptimizationWarning(UserWarning):
    """PSF optimisation did not converge or terminated early."""
