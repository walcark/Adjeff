"""Exception hierarchy of the package.

Classes
-------
    AdjeffError
        Base of every error below.
    MissingVariableError
        A required variable is absent from a band.
    AdjeffAccessorError
        The ``.adjeff`` accessor was misused.
    ConfigurationError
        Invalid or missing configuration.
    ImageIOError
        A product file could not be read.
    ComputationError
        A module produced an unusable result.
    OptimizationWarning
        A PSF fit did not converge.
"""


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

    Raised before the cache is written, so a bad result fails where it
    was caused rather than being read back by later runs.
    """


class OptimizationWarning(UserWarning):
    """PSF optimisation did not converge or terminated early."""
