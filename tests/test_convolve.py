import pytest

import numpy as np
import scipy
import torch
import xarray as xr

from adjeff.utils import fft_convolve_2D


def test_raise_error_wrong_conv_type():
    """Assert that a ValueError is raised if a wrong conv_type is specified."""
    np.random.seed(0)
    arr = np.random.rand(8, 8)
    kernel = np.random.rand(3, 3)
    
    with pytest.raises(ValueError):
        fft_convolve_2D(
            arr, 
            kernel, 
            padding="constant", 
            conv_type="random_conv_name", 
            device="cpu",
        )


def test_fft_convolve_vs_scipy():
    """Compare PyTorch FFT convolution against Scipy 2D convolution."""
    np.random.seed(0)
    arr = np.random.rand(8, 8)
    kernel = np.random.rand(3, 3)

    result_fft = fft_convolve_2D(
        arr, 
        kernel, 
        padding="constant", 
        conv_type="valid", 
        device="cpu",
    )
    result_scipy = scipy.signal.convolve2d(arr, kernel, mode="valid")

    assert np.allclose(result_fft, result_scipy, rtol=1e-5), (
        "FFT convolution differs from scipy"
    )


def test_fft_convolve_vs_scipy_even_kernel():
    """Compare PyTorch FFT convolution against Scipy 2D convolution."""
    np.random.seed(0)
    arr = np.random.rand(8, 8)
    kernel = np.random.rand(4, 4)

    result_fft = fft_convolve_2D(
        arr, 
        kernel, 
        padding="constant", 
        conv_type="valid", 
        device="cpu",
    )
    result_scipy = scipy.signal.convolve2d(arr, kernel, mode="valid")

    assert np.allclose(result_fft, result_scipy, rtol=1e-5), (
        "FFT convolution differs from scipy"
    )


def test_fft_convolve_xarray():
    """Check that fft_convolve_2D works with xarray.DataArray inputs."""
    da = xr.DataArray(np.random.rand(10, 10))
    kernel = xr.DataArray(np.ones((3, 3)))
    result = fft_convolve_2D(
        da, 
        kernel, 
        padding="constant", 
        conv_type="same", 
        device="cpu"
    )
    assert isinstance(result, xr.DataArray), (
        "Result should be numpy array"
    )
    assert result.shape == da.shape, (
        "Shape mismatch for 'same' conv_type"
    )


def test_fft_convolve_numpy():
    """Check that fft_convolve_2D works with numpy.ndarray inputs."""
    da = np.random.rand(10, 10)
    kernel = np.ones((3, 3))
    result = fft_convolve_2D(
        da, 
        kernel, 
        padding="constant", 
        conv_type="same", 
        device="cpu"
    )
    assert isinstance(result, np.ndarray), (
        "Result should be numpy array"
    )
    assert result.shape == da.shape, (
        "Shape mismatch for 'same' conv_type"
    )


def test_fft_convolve_gpu_cpu():
    """Compare fft_convolve_2D results on CPU vs GPU for consistency."""
    arr = np.random.rand(16, 16)
    kernel = np.random.rand(5, 5)
    result_cpu = fft_convolve_2D(
        arr, 
        kernel, 
        padding="constant", 
        conv_type="valid", 
        device="cpu",
    )
    if torch.cuda.is_available():
        result_gpu = fft_convolve_2D(
            arr, 
            kernel, 
            padding="constant", 
            conv_type="valid", 
            device="cuda",
        )
        assert np.allclose(result_cpu, result_gpu, rtol=1e-5), (
            "CPU vs GPU mismatch"
        )


@pytest.mark.parametrize("conv_type", ["valid", "same"])
@pytest.mark.parametrize("padding", ["constant", "reflect"])
def test_fft_various(conv_type, padding):
    """Test fft_convolve_2D with various padding and output modes."""
    arr = np.random.rand(8, 8)
    kernel = np.random.rand(3, 3)
    fft_convolve_2D(
        arr, 
        kernel, 
        conv_type=conv_type, 
        padding=padding, 
        device="cpu",
    )
