"""Trainable models convolving a scene with a learnable PSF.

Classes
-------
    Unif2Surface
        ``rho_s`` from ``rho_unif``: PSF convolution, then the 5S formula.

Example
-------
::

    model = make_model(Unif2Surface, GaussPSF, bands, res_km, n, {"sigma": 0.1})
    tree = fit(model, train_images)
"""

from .unif2surface import Unif2Surface

__all__ = ["Unif2Surface"]
