import pytest


try:
    import pycuda.driver  # noqa: F401
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
