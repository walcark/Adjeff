"""Linear 2-D convolution by FFT, on tensors or DataArrays.

Functions
---------
    fft_convolve_2D
        Convolution of DataArrays, over any extra dims.
    fft_convolve_2D_torch
        Convolution of 2-D tensors, differentiable.
"""

from typing import Literal, cast

import numpy as np
import torch
import xarray as xr


def fft_convolve_2D(
    in1: xr.DataArray,
    in2: xr.DataArray,
    *,
    padding: Literal["constant", "reflect", "replicate"],
    const_padding_values: float = 0.0,
    conv_type: str = "valid",
    device: torch.device | str = "cuda",
) -> xr.DataArray:
    """Convolve *in1* ``(y, x)``, with the kernel *in2* ``(y_psf, x_psf)``.

    Extra dims of *in1* are looped over; the output keeps them and the
    coordinates of *in1*.  See :func:`fft_convolve_2D_torch` for the
    other parameters; *device* is the torch device used.
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
    """Return the linear convolution of the ``(N, N)`` *in1* by the ``(K, K)`` *in2*.

    *in1* is extended by ``K - 1`` with *padding*, so the FFT product
    does not wrap around.

    Parameters
    ----------
    padding : {"constant", "reflect", "replicate"}
        Extension of *in1*; ``"constant"`` uses *const_padding_values*.
    conv_type : {"valid", "same"}, optional
        Output of size ``N - K + 1``, or ``N``.
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

    # Kernel padded — use F.pad so the gradient flows through in2.
    in2_ext = torch.nn.functional.pad(in2, (0, ext_fft - k, 0, ext - k))

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
