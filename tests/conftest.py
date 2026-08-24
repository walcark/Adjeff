"""Test whether the working environment has Dask or Cuda."""
import pytest


# Importing pycuda succeeds wherever it is installed, GPU or not: a CI
# runner has the package and no device.  Initialising the driver and
# counting devices is what actually tells the two apart.
try:
    import pycuda.driver as _drv

    _drv.init()
    _HAS_CUDA = _drv.Device.count() > 0
except Exception:
    _HAS_CUDA = False

try:
    import dask  # noqa: F401
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False


requires_dask = pytest.mark.skipif(not _HAS_DASK, reason="dask not available")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
