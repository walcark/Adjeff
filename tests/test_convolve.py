import pytest

import numpy as np
import scipy
import torch
import xarray as xr

from adjeff.utils import fft_convolve_2D


def _make_inputs(
    arr: np.ndarray, kernel: np.ndarray
) -> tuple[xr.DataArray, xr.DataArray]:
    return (
        xr.DataArray(arr, dims=["y", "x"]),
        xr.DataArray(kernel, dims=["y_psf", "x_psf"]),
    )


def test_raise_error_wrong_conv_type():
    """Assert that a ValueError is raised if a wrong conv_type is specified."""
    np.random.seed(0)
    arr, kernel = _make_inputs(np.random.rand(8, 8), np.random.rand(3, 3))

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
    raw_arr = np.random.rand(8, 8)
    raw_kernel = np.random.rand(3, 3)
    arr, kernel = _make_inputs(raw_arr, raw_kernel)

    result_fft = fft_convolve_2D(
        arr,
        kernel,
        padding="constant",
        conv_type="valid",
        device="cpu",
    )
    result_scipy = scipy.signal.convolve2d(raw_arr, raw_kernel, mode="valid")

    assert np.allclose(result_fft.values, result_scipy, rtol=1e-5), (
        "FFT convolution differs from scipy"
    )


def test_fft_convolve_vs_scipy_even_kernel():
    """Compare PyTorch FFT convolution against Scipy 2D convolution."""
    np.random.seed(0)
    raw_arr = np.random.rand(8, 8)
    raw_kernel = np.random.rand(4, 4)
    arr, kernel = _make_inputs(raw_arr, raw_kernel)

    result_fft = fft_convolve_2D(
        arr,
        kernel,
        padding="constant",
        conv_type="valid",
        device="cpu",
    )
    result_scipy = scipy.signal.convolve2d(raw_arr, raw_kernel, mode="valid")

    assert np.allclose(result_fft.values, result_scipy, rtol=1e-5), (
        "FFT convolution differs from scipy"
    )


def test_fft_convolve_xarray():
    """Check that fft_convolve_2D works with xarray.DataArray inputs."""
    arr, kernel = _make_inputs(np.random.rand(10, 10), np.ones((3, 3)))
    result = fft_convolve_2D(
        arr,
        kernel,
        padding="constant",
        conv_type="same",
        device="cpu",
    )
    assert isinstance(result, xr.DataArray), "Result should be a DataArray"
    assert result.shape == arr.shape, "Shape mismatch for 'same' conv_type"


def test_fft_convolve_gpu_cpu():
    """Compare fft_convolve_2D results on CPU vs GPU for consistency."""
    arr, kernel = _make_inputs(np.random.rand(16, 16), np.random.rand(5, 5))
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
        assert np.allclose(result_cpu.values, result_gpu.values, rtol=1e-5), (
            "CPU vs GPU mismatch"
        )


@pytest.mark.parametrize("conv_type", ["valid", "same"])
@pytest.mark.parametrize("padding", ["constant", "reflect"])
def test_fft_various(conv_type, padding):
    """Test fft_convolve_2D with various padding and output modes."""
    arr, kernel = _make_inputs(np.random.rand(8, 8), np.random.rand(3, 3))
    fft_convolve_2D(
        arr,
        kernel,
        conv_type=conv_type,
        padding=padding,
        device="cpu",
    )
