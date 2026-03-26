"""Implement FFT convolution with PyTorch GPU Optimization.

The main method `fft_convolve_2D_torch` operates on torch.Tensor objects.
The wrapper `fft_convolve_2D` operates on xr.DataArray objects and handles
arbitrary extra dimensions via ``xr.apply_ufunc``.

"""

from typing import Literal, cast

import numpy as np
import torch
import xarray as xr

from .logger import adjeff_logging

logger = adjeff_logging.get_logger()


@adjeff_logging.log_execution_time
def fft_convolve_2D(
    in1: xr.DataArray,
    in2: xr.DataArray,
    *,
    padding: Literal["constant", "reflect", "replicate"],
    const_padding_values: float = 0.0,
    conv_type: str = "valid",
    device: torch.device | str = "cuda",
) -> xr.DataArray:
    """Perform a 2D convolution on xarray DataArrays using PyTorch FFT.

    This is a wrapper around `fft_convolve_2D_torch` that handles arbitrarily
    many extra dimensions in ``in1`` (e.g. ``aot``, ``wl``) by sweeping over
    them via ``xr.apply_ufunc`` and reconstructing the output with the same
    shape and coordinates. The following naming conventions are required:

    - Spatial dimensions of ``in1`` must be named ``"y"`` and ``"x"``.
    - Spatial dimensions of ``in2`` (the kernel) must be named ``"y_psf"``
      and ``"x_psf"``.

    Parameters
    ----------
    in1 : xr.DataArray
        Input array. May have extra dimensions beyond ``(y, x)``.
    in2 : xr.DataArray
        Convolution kernel with spatial dims ``(y_psf, x_psf)``.
    padding : {"constant", "reflect", "replicate"}
        Padding method. Only `"constant"` uses `const_padding_values`.
    const_padding_values : float, optional
        Value used for constant padding (default 0.0).
    conv_type : {"valid", "same"}, optional
        Output size mode.
    device : torch.device or str, optional
        Device to perform computation on (default ``"cuda"``).

    Returns
    -------
    xr.DataArray
        The convolved array with the same extra dimensions as ``in1``.

    """

    def _convolve_slice(arr: np.ndarray, k: np.ndarray) -> np.ndarray:
        in1_t = torch.tensor(arr, device=device, dtype=torch.float32)
        in2_t = torch.tensor(k, device=device, dtype=torch.float32)
        return (
            fft_convolve_2D_torch(
                in1_t,
                in2_t,
                padding=padding,
                const_padding_values=const_padding_values,
                conv_type=conv_type,
            )
            .detach()
            .cpu()
            .numpy()
        )

    result = xr.apply_ufunc(
        _convolve_slice,
        in1,
        in2,
        input_core_dims=[["y", "x"], ["y_psf", "x_psf"]],
        output_core_dims=[["y_out", "x_out"]],
        vectorize=True,
    ).rename({"y_out": "y", "x_out": "x"})

    n_out = result.sizes["y"]
    half = (in1.sizes["y"] - n_out) // 2
    return cast(
        xr.DataArray,
        result.assign_coords(
            y=in1.coords["y"].values[half : half + n_out],
            x=in1.coords["x"].values[half : half + n_out],
        ),
    )


def fft_convolve_2D_torch(
    in1: torch.Tensor,
    in2: torch.Tensor,
    *,
    padding: Literal["constant", "reflect", "replicate"],
    const_padding_values: float = 0.0,
    conv_type: str = "valid",
) -> torch.Tensor:
    """Compute a **true linear 2D FFT convolution** between two torch tensors.

    Unlike a standard FFT-based convolution, which is circular and can produce
    wrap around artifacts at the tensor edges, the linear convolution is
    computed by:

    1) Extending the input to avoid wrap-around,
    2) Zero-padding the kernel to match the extended input,
    3) Multiplying in the Fpurier domain (rFFT -> iFFT),
    4) Cropping the result according to `conv_type`.

    This method is GPU-efficient and minimizes temporary allocations.

    Parameters
    ----------
    in1 : torch.Tensor
        2D input tensor of shape (N, N).
    in2 : torch.Tensor
        2D convolution kernel of shape (K, K).
    padding : {"constant", "reflect", "replicate"}
        How to pad the input before FFT. Only `"constant"` uses
        `const_padding_values`.
    const_padding_values : float, optional
        Constant value used when `padding="constant"` (default 0.0).
    conv_type : {"valid", "same"}, optional
        Determines output size: `"valid"`: only positions where the
        kernel fully overlaps input (N-K+1 × N-K+1), and `"same"`
        output has the same shape as input (N × N).

    Returns
    -------
    torch.Tensor
        The convolved 2D tensor, with shape determined by `mode_out`.

    """
    n = in1.shape[0]  # input size
    k = in2.shape[0]  # kernel size
    ext = n + k - 1  # Full linear extension

    # Simply pad the input with the constant value in constant
    # mode, else cast to 3D for `reflect` and `replicate`.
    pad = (0, k - 1, 0, k - 1)
    if padding == "constant":
        in1_ext = torch.nn.functional.pad(
            in1,
            pad,
            mode="constant",
            value=const_padding_values,
        )
    else:
        in1_ext = in1.unsqueeze(0).unsqueeze(0)
        in1_ext = torch.nn.functional.pad(
            in1_ext,
            pad,
            mode=padding,
        )
        in1_ext = in1_ext.squeeze(0).squeeze(0)

    # Ensure width even for rfft
    ext_fft = ext + (ext % 2)
    add_col = ext_fft != ext
    if add_col:
        in1_ext = torch.nn.functional.pad(
            in1_ext,
            (0, 1, 0, 0),
            mode="constant",
            value=0.0,
        )

    # Kernel padded
    in2_ext = torch.zeros((ext, ext_fft), dtype=in1.dtype, device=in1.device)
    in2_ext[:k, :k] = in2

    # FFT-based linear conv
    in1_fft = torch.fft.rfftn(in1_ext, s=(ext, ext_fft), dim=(0, 1))
    in2_fft = torch.fft.rfftn(in2_ext, s=(ext, ext_fft), dim=(0, 1))
    Y = torch.fft.irfftn(in1_fft * in2_fft, s=(ext, ext_fft), dim=(0, 1))
    if add_col:
        Y = Y[:, :ext]

    # Crop according to `conv_type`
    if conv_type == "valid":
        out_n = n - k + 1
        top = k - 1
    elif conv_type == "same":
        out_n = n
        top = (ext - n) // 2
    else:
        raise ValueError("Mode should either be 'same' or 'valid'.")

    out = Y[top : top + out_n, top : top + out_n].contiguous()
    return cast(torch.Tensor, out)
