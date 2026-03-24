"""Adjeff exceptions hierarchy."""


class AdjeffError(Exception):
    """Base exception for all Adjeff errors."""


class MissingVariableError(AdjeffError):
    """Required variable missing from one or more band Datasets."""


class AdjeffAccessorError(AdjeffError):
    """An error occured during the call to the xarray adjeff accessor."""


class ConfigurationError(AdjeffError):
    """Invalid or missing configuration parameters."""
