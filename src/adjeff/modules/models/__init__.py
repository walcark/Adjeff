"""Trainable scene modules applying a learnable PSF convolution.

Currently contains one model:

- :class:`Unif2Surface` — estimates surface reflectance ``rho_s``
  from uniform reflectance ``rho_unif`` by convolving with a learnable
  PSF and applying the 5S formula.

Typical workflow::

    psf_dict = init_psf_dict(grids, GaussPSF, {"sigma": 0.1})
    model = Unif2Surface(psf_dict)
    optimizer = LBFGSOptimizer(train_images, config)
    result = optimizer.run(model)  # returns a frozen PSFDict
"""

from .unif2surface import Unif2Surface

__all__ = ["Unif2Surface"]
