"""Trainable scene modules applying a learnable PSF convolution.

Currently contains one model:

- :class:`Unif2Surface`: estimates surface reflectance ``rho_s``
  from uniform reflectance ``rho_unif`` by convolving with a learnable
  PSF and applying the 5S formula.

Typical workflow::

    model = make_model(Unif2Surface, GaussPSF, bands, res_km, n, {"sigma": 0.1})
    tree = fit(model, train_images)  # returns a frozen PSF tree
"""

from .unif2surface import Unif2Surface

__all__ = ["Unif2Surface"]
