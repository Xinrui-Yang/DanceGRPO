from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED = False


def _pad_rows_to_block_size(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    rows = x.shape[0]
    pad_rows = (-rows) % block_size
    if pad_rows == 0:
        return x, 0
    return F.pad(x, (0, 0, 0, pad_rows)), pad_rows


def enable_mxfp8_backward_padding_patch() -> bool:
    """Patch torchao mx_mm.backward to tolerate non-multiple-of-32 token rows.

    The original backward path assumes row dimension is divisible by block_size for
    dim1 casting in grad_weight computation. For variable-length multimodal batches,
    this may fail (e.g., 524 rows). We pad rows with zeros to the next multiple of
    block_size before dim1 casts. Zero-padded rows contribute zero gradient.
    """
    global _PATCHED
    if _PATCHED:
        return True

    if os.getenv("VERL_DISABLE_MXFP8_BACKWARD_PAD", "0") == "1":
        logger.info("Skip MXFP8 backward padding patch: VERL_DISABLE_MXFP8_BACKWARD_PAD=1")
        return False

    try:
        from torchao.prototype.mx_formats.config import MXFP8Dim1CastKernelChoice
        from torchao.prototype.mx_formats.mx_linear import mx_mm
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
    except Exception as exc:
        logger.warning("Skip MXFP8 backward padding patch: cannot import torchao mx modules: %s", exc)
        return False

    if getattr(mx_mm, "_verl_mxfp8_pad_patch", False):
        _PATCHED = True
        return True

    def _patched_backward(ctx: Any, grad_output_hp: torch.Tensor):
        input_hp, weight_hp = ctx.saved_tensors
        in_elem_dtype = ctx.in_elem_dtype
        w_elem_dtype = ctx.w_elem_dtype
        grad_elem_dtype = ctx.grad_elem_dtype
        block_size = ctx.block_size
        kernel_preference = ctx.kernel_preference
        mxfp8_dim0_cast_kernel_choice = ctx.mxfp8_dim0_cast_kernel_choice
        mxfp8_dim1_cast_kernel_choice = ctx.mxfp8_dim1_cast_kernel_choice
        scale_calculation_mode = ctx.scale_calculation_mode

        grad_output_orig_shape = grad_output_hp.shape
        grad_output_hp_r = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

        input_hp_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_hp_orig_shape[-1])

        # grad_output @ weight = grad_input (unchanged path)
        grad_output_mx_dim0 = MXTensor.to_mx(
            grad_output_hp_r,
            grad_elem_dtype,
            block_size,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
        )

        if mxfp8_dim1_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                weight_hp,
                block_size,
                w_elem_dtype,
                weight_hp.dtype,
                kernel_preference,
                mxfp8_dim1_cast_kernel_choice,
                scale_calculation_mode,
            )
        else:
            weight_hp_t_c = weight_hp.t().contiguous()
            weight_mx_dim1 = MXTensor.to_mx(
                weight_hp_t_c,
                w_elem_dtype,
                block_size,
                kernel_preference=kernel_preference,
                scaling_mode=scale_calculation_mode,
                mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
            )
        grad_input = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())
        grad_input = grad_input.reshape(*grad_output_orig_shape[:-1], grad_input.shape[-1])

        # input_t @ grad_output = grad_weight
        # Pad dynamic token rows to satisfy dim1 cast alignment requirements.
        if grad_output_hp_r.shape[0] != input_hp_r.shape[0]:
            raise AssertionError(
                f"Expected matched flattened rows, got grad_output={grad_output_hp_r.shape[0]} "
                f"and input={input_hp_r.shape[0]}"
            )
        grad_output_hp_r_for_weight, pad_rows = _pad_rows_to_block_size(grad_output_hp_r, block_size)
        input_hp_r_for_weight, input_pad_rows = _pad_rows_to_block_size(input_hp_r, block_size)
        if input_pad_rows != pad_rows:
            raise AssertionError(f"Padding mismatch: grad_output pad={pad_rows}, input pad={input_pad_rows}")
        if pad_rows > 0:
            logger.debug(
                "MXFP8 backward row padding applied: %s -> %s (block_size=%s)",
                grad_output_hp_r.shape[0],
                grad_output_hp_r_for_weight.shape[0],
                block_size,
            )

        if mxfp8_dim1_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            grad_output_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                grad_output_hp_r_for_weight,
                block_size,
                grad_elem_dtype,
                grad_output_hp_r_for_weight.dtype,
                kernel_preference,
                mxfp8_dim1_cast_kernel_choice,
                scale_calculation_mode,
            )
        else:
            grad_output_mx_dim1 = MXTensor.to_mx(
                grad_output_hp_r_for_weight.t().contiguous(),
                grad_elem_dtype,
                block_size,
                kernel_preference=kernel_preference,
                scaling_mode=scale_calculation_mode,
                mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
            )

        if mxfp8_dim1_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            input_t_mx_dim0_tmp = _to_mxfp8_dim1_kernel_wrapper(
                input_hp_r_for_weight,
                block_size,
                in_elem_dtype,
                input_hp_r_for_weight.dtype,
                kernel_preference,
                mxfp8_dim1_cast_kernel_choice,
                scale_calculation_mode,
            )
            input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
        else:
            input_t_mx_dim0_tmp = MXTensor.to_mx(
                input_hp_r_for_weight.t().contiguous(),
                in_elem_dtype,
                block_size,
                kernel_preference=kernel_preference,
                scaling_mode=scale_calculation_mode,
                mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
            )
            input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
        grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)

        return grad_input, grad_weight, None, None, None, None, None, None, None, None

    mx_mm.backward = staticmethod(_patched_backward)
    setattr(mx_mm, "_verl_mxfp8_pad_patch", True)
    _PATCHED = True
    logger.info("Enabled MXFP8 backward row padding patch")
    return True