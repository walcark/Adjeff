"""Implement FFT convolution with PyTorch GPU Optimization.

The main method `fft_convolve_pytorch` uses an input torch.Tensor and returns
another torch.Tensor. A wrapper method `fft_convolve` allows to perform the
convolution on either a numpy.ndarray or xarray.DataArray.

"""

from typing import Literal, cast

import numpy as np
import torch
import xarray as xr

from .logger import adjeff_logging

logger = adjeff_logging.get_logger()


@adjeff_logging.log_execution_time
def fft_convolve_2D(
    in1: np.ndarray | xr.DataArray,
    in2: np.ndarray | xr.DataArray,
    *,
    padding: Literal["constant", "reflect", "replicate"],
    const_padding_values: float = 0.0,
    conv_type: str = "valid",
    device: torch.device | str = "cuda",
) -> np.ndarray | xr.DataArray:
    """Perform a 2D convolution on numpy or xarray objects using PyTorch FFT.

    This is a wrapper around `fft_convolve_2D_torch` that allows passing numpy
    arrays or xarray.DataArray as input. The actual computation is performed
    on a PyTorch tensor on the specified device.

    Parameters
    ----------
    in1 : np.ndarray or xarray.DataArray
        Input 2D array.
    in2 : np.ndarray or xarray.DataArray
        2D convolution kernel.
    padding : {"constant", "reflect", "replicate"}
        Padding method before convolution. Only `"constant"` uses
        `const_padding_values`.
    const_padding_values : float, optional
        Value used for constant padding (default 0.0).
    conv_type : {"valid", "same"}, optional
        Output size mode.
    device : torch.device or str, optional
        Device to perform computation on `"cpu"` or `"cuda"`, with
        `"cuda"` by default.

    Returns
    -------
    np.ndarray or xarray.DataArray
        The convolved array, same type as input.

    """
    # Save metadata if input is xarray
    if isinstance(in1, xr.DataArray):
        dims, coords = (in1.dims, in1.coords)
    elif isinstance(in1, np.ndarray):
        dims, coords = (None, None)
    else:
        raise ValueError(f"Wrong input type: {type(in1)}.")

    # Convert input to PyTorch tensor
    in1_tensor = torch.tensor(
        np.asarray(in1), device=device, dtype=torch.float32
    )
    in2_tensor = torch.tensor(
        np.asarray(in2), device=device, dtype=torch.float32
    )

    # Perform convolution
    result = (
        fft_convolve_2D_torch(
            in1_tensor,
            in2_tensor,
            padding=padding,
            const_padding_values=const_padding_values,
            conv_type=conv_type,
        )
        .detach()
        .cpu()
        .numpy()
    )

    # Return as xarray if needed
    if isinstance(in1, xr.DataArray):
        return xr.DataArray(result, dims=dims, coords=coords)
    else:
        return result


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
