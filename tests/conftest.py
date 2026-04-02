"""Test whether the working environment has Dask or Cuda."""
import pytest


try:
    import pycuda.driver  # noqa: F401
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False

try:
    import dask  # noqa: F401
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False


requires_dask = pytest.mark.skipif(not _HAS_DASK, reason="dask not available")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
