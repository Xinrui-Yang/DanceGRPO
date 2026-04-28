# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
# from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_wan_2_1_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from contextlib import contextmanager
from safetensors.torch import save_file
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
import json

from torchtitan.config import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from fastvideo.utils.mxfp8_backward_padding import enable_mxfp8_backward_padding_patch
from torchao.prototype.mx_formats.mx_linear import MXLinear
from torchao.prototype.mx_formats.config import MXLinearConfig
from torchao.prototype.mx_formats.config import MXLinearRecipeName
from functools import wraps

def create_job_config() -> JobConfig:
    job_config = JobConfig()
    job_config.model.converters = ["quantize.linear.mx"]

    job_config.parallelism.tensor_parallel_degree = 1
    job_config.parallelism.context_parallel_degree =1
    job_config.parallelism.data_parallel_shard_degree = -1
    job_config.parallelism.data_parallel_replicate_degree = 1
    job_config.parallelism.fsdp_reshard_after_forward = "always"
    
    job_config.quantize.linear.mx.filter_fqns = ["output", "condition_embedder"]
    job_config.quantize.linear.mx.mxfp8_dim1_cast_kernel_choice = "cuda"
    job_config.quantize.linear.mx.recipe_name = "mxfp8_cublas"
        
    return job_config


def create_parallel_dims(job_config, world_size: int) -> ParallelDims:
    return ParallelDims(
        dp_shard = job_config.parallelism.data_parallel_shard_degree,
        dp_replicate = job_config.parallelism.data_parallel_replicate_degree,
        cp = job_config.parallelism.context_parallel_degree,
        tp = job_config.parallelism.tensor_parallel_degree,
        pp = job_config.parallelism.pipeline_parallel_degree,
        ep = job_config.parallelism.expert_parallel_degree,
        etp = job_config.parallelism.expert_tensor_parallel_degree,
        world_size = world_size,
    )


def build_mx_linear_converters(
    transformer,
    world_size: int,
):
    """Convert the linear layers to TorchAO MXLinear."""
    enable_mxfp8_backward_padding_patch()
    
    job_config = create_job_config()
    parallel_dims = create_parallel_dims(job_config, world_size)

    from torchtitan.protocols.model_converter import build_model_converters
    
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(transformer)

    return transformer

@dataclass(frozen=True)
class ResolvedMXLinearConfig:
    precision: str
    config: MXLinearConfig
    recipe_name: str
    fallback_from: str | None = None

_MX_LINEAR_CONFIG_CACHE: dict[str, ResolvedMXLinearConfig] = {}

def _resolve_mx_linear_config(precision: str) -> ResolvedMXLinearConfig:
    if precision == "mxfp8":
        recipe = MXLinearRecipeName.MXFP8_CUBLAS
        return ResolvedMXLinearConfig(
            precision=precision,
            config=MXLinearConfig.from_recipe_name(recipe),
            recipe_name=recipe.value,
        )
    if precision == "mxfp4":
        requested_recipe = MXLinearRecipeName.MXFP4_CUTLASS
        return ResolvedMXLinearConfig(
            precision=precision,
            config=MXLinearConfig.from_recipe_name(requested_recipe),
            recipe_name=requested_recipe.value,
        )
    if precision == "nvfp4":
        placeholder_recipe = MXLinearRecipeName.MXFP4_CUTLASS
        return ResolvedMXLinearConfig(
            precision=precision,
            # NVFP4 rollout is handled by a dedicated inference-only fast path
            # in MXLinear.forward. We keep an MXFP4 placeholder config here so
            # the same MXLinear modules can still participate in runtime
            # precision switching without altering the model conversion flow.
            config=MXLinearConfig.from_recipe_name(placeholder_recipe),
            recipe_name="nvfp4_flashinfer",
        )
    raise ValueError(f"Unsupported MX precision: {precision}")


def _patch_rms_norm_forward_for_mixed_precision() -> None:
    if getattr(torch.nn.RMSNorm, "_dancegrpo_mixed_precision_patch_applied", False):
        return

    original_forward = torch.nn.RMSNorm.forward

    @wraps(original_forward)
    def patched_forward(self, x):
        weight = self.weight
        if weight is not None and x.is_cuda and x.dtype != weight.dtype:
            return F.rms_norm(
                x,
                self.normalized_shape,
                weight.to(dtype=x.dtype),
                self.eps,
            )
        return original_forward(self, x)

    torch.nn.RMSNorm.forward = patched_forward
    torch.nn.RMSNorm._dancegrpo_mixed_precision_patch_applied = True


def _get_resolved_mx_linear_config(precision: str) -> ResolvedMXLinearConfig:
    config = _MX_LINEAR_CONFIG_CACHE.get(precision)
    if config is None:
        config = _resolve_mx_linear_config(precision)
        _MX_LINEAR_CONFIG_CACHE[precision] = config
    return config

def _get_transformer_module(transformer):
    return transformer.module if isinstance(transformer, FSDP) else transformer

def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_mxlinear_modules(transformer) -> tuple[MXLinear, ...]:
    module = _get_transformer_module(transformer)
    mxlinear_modules = getattr(module, "_dancegrpo_mxlinear_modules", None)
    if mxlinear_modules is None:
        mxlinear_modules = tuple(
            mod for mod in module.modules() if isinstance(mod, MXLinear)
        )
        module._dancegrpo_mxlinear_modules = mxlinear_modules
    return mxlinear_modules


def _patch_mxlinear_forward_for_precision_switch() -> None:
    if getattr(MXLinear, "_dancegrpo_precision_switch_patch_applied", False):
        return

    original_forward = MXLinear.forward

    @wraps(original_forward)
    def patched_forward(self, x):
        if getattr(self, "_dancegrpo_runtime_precision", None) == "bf16":
            return F.linear(x, self.weight, self.bias)
        return original_forward(self, x)

    MXLinear.forward = patched_forward
    MXLinear._dancegrpo_precision_switch_patch_applied = True


def _needs_mx_linear_conversion(train_precision: str, rollout_precision: str) -> bool:
    return (train_precision, rollout_precision) != ("bf16", "bf16")

class PrecisionSwitchController:
    def __init__(self, transformer, train_precision: str, rollout_precision: str):
        self.module = _get_transformer_module(transformer)
        self.train_precision = train_precision
        self.rollout_precision = rollout_precision
        self.current_precision = None
        self.mxlinear_modules = (
            _get_mxlinear_modules(self.module)
            if _needs_mx_linear_conversion(train_precision, rollout_precision)
            else ()
        )
        self._mx_configs = {}

        if self.mxlinear_modules:
            _patch_mxlinear_forward_for_precision_switch()
            for precision in {train_precision, rollout_precision}:
                if precision != "bf16":
                    self._mx_configs[precision] = _get_resolved_mx_linear_config(precision)

        self.switch_to_train()

    def _clear_nvfp4_weight_caches(self) -> None:
        if not self.mxlinear_modules:
            return

        cleared = False
        for mod in self.mxlinear_modules:
            if hasattr(mod, "_dancegrpo_nvfp4_weight_cache"):
                delattr(mod, "_dancegrpo_nvfp4_weight_cache")
                cleared = True
            if hasattr(mod, "_dancegrpo_nvfp4_weight_cache_key"):
                delattr(mod, "_dancegrpo_nvfp4_weight_cache_key")

        if cleared and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def switch_to(self, precision: str) -> None:
        if self.current_precision == precision:
            return

        previous_precision = self.current_precision

        if precision == "bf16":
            for mod in self.mxlinear_modules:
                mod._dancegrpo_runtime_precision = "bf16"
        else:
            mx_config = self._mx_configs[precision].config
            for mod in self.mxlinear_modules:
                mod.config = mx_config
                mod._dancegrpo_runtime_precision = precision

        if (
            previous_precision == "nvfp4"
            and precision != "nvfp4"
            and _env_flag("DANCEGRPO_NVFP4_CLEAR_CACHE_ON_TRAIN", True)
        ):
            self._clear_nvfp4_weight_caches()

        self.current_precision = precision

    def switch_to_train(self) -> None:
        self.switch_to(self.train_precision)

    def switch_to_rollout(self) -> None:
        self.switch_to(self.rollout_precision)


def configure_precision_switch_controller(transformer, train_precision: str, rollout_precision: str):
    controller = PrecisionSwitchController(transformer, train_precision, rollout_precision)
    module = _get_transformer_module(transformer)
    module._dancegrpo_precision_controller = controller
    return controller


def _get_precision_switch_controller(transformer):
    module = _get_transformer_module(transformer)
    return getattr(module, "_dancegrpo_precision_controller", None)


def use_reference_old_log_probs(args) -> bool:
    if args.old_log_prob_source == "reference":
        return True
    if args.old_log_prob_source == "behavior":
        return False
    return args.train_precision != args.rollout_precision


def should_compute_reference_log_probs(args) -> bool:
    return use_reference_old_log_probs(args) or args.reference_policy_kl_coef > 0

def should_apply_rollout_correction(args) -> bool:
    return args.rollout_correction_mode == "timestep_tis"

def compute_rollout_tis_weights(
    args,
    old_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
) -> torch.Tensor:
    corr_log_ratio = (old_log_probs.to(torch.float32) - behavior_log_probs.to(torch.float32)).clamp(
        min=-args.rollout_is_log_bound,
        max=args.rollout_is_log_bound,
    )
    corr_weight = torch.exp(corr_log_ratio).clamp(max=args.rollout_is_threshold)
    if args.rollout_is_batch_normalize:
        corr_weight = corr_weight / corr_weight.mean().clamp_min(1e-6)
    return corr_weight.detach()


def summarize_rollout_tis_weights(
    corr_weights: torch.Tensor,
    old_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    with torch.no_grad():
        gathered_weights = gather_tensor(corr_weights.detach().reshape(-1).to(torch.float32))
        gathered_gap = gather_tensor(
            (old_log_probs.detach() - behavior_log_probs.detach()).abs().reshape(-1).to(torch.float32)
        )
        gathered_clipped = gather_tensor(
            (corr_weights.detach().reshape(-1).to(torch.float32) >= threshold).to(torch.float32)
        )

        weight_sum = gathered_weights.sum()
        weight_sq_sum = gathered_weights.square().sum()
        count = max(int(gathered_weights.numel()), 1)
        ess_ratio = (weight_sum.square() / (weight_sq_sum.clamp_min(1e-12) * count)).item()

        return {
            "corr_weight_mean": gathered_weights.mean().item(),
            "corr_weight_p95": torch.quantile(gathered_weights, 0.95).item(),
            "corr_weight_max": gathered_weights.max().item(),
            "corr_weight_clipped_fraction": gathered_clipped.mean().item(),
            "mean_abs_logprob_gap": gathered_gap.mean().item(),
            "ess_ratio": ess_ratio,
        }


def set_model_precision(transformer, precision: str):
    controller = _get_precision_switch_controller(transformer)
    if controller is None:
        controller = configure_precision_switch_controller(transformer, precision, precision)
    controller.switch_to(precision)


@contextmanager
def rollout_precision_context(transformer):
    controller = _get_precision_switch_controller(transformer)
    was_training = transformer.training
    if controller is None:
        transformer.eval()
        try:
            yield
        finally:
            transformer.train(was_training)
        return

    controller.switch_to_rollout()
    transformer.eval()
    try:
        yield
    finally:
        controller.switch_to_train()
        transformer.train(was_training)


@contextmanager
def reference_policy_context(transformer):
    controller = _get_precision_switch_controller(transformer)
    was_training = transformer.training
    if controller is None:
        transformer.eval()
        try:
            yield
        finally:
            transformer.train(was_training)
        return

    controller.switch_to_train()
    transformer.eval()
    try:
        yield
    finally:
        controller.switch_to_train()
        transformer.train(was_training)


def video_first_frame_to_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频的第一帧")
        cap.release()
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(frame_rgb)

    cap.release()

    return pil_image

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        negative_prompt_embeds, 
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            if args.cfg_infer>1:
                with torch.autocast("cuda", torch.bfloat16):
                    pred= transformer(
                        hidden_states=torch.cat([z,z],dim=0),
                        timestep=torch.cat([timesteps,timesteps],dim=0),
                        encoder_hidden_states=torch.cat([encoder_hidden_states,negative_prompt_embeds],dim=0),
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    model_pred, uncond_pred = pred.chunk(2)
                    pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
            else:
                 with torch.autocast("cuda", torch.bfloat16):
                    pred= transformer(
                        hidden_states=z,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            # Keep rollout trajectory tensors in float32 for stable PPO/reference log-prob reuse.
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            negative_prompt_embeds, 
            transformer,
            timesteps,
            i,
            sigma_schedule,
):
    if args.cfg_infer>1:
        with torch.autocast("cuda", torch.bfloat16):
            pred= transformer(
                hidden_states=torch.cat([latents,latents],dim=0),
                timestep=torch.cat([timesteps,timesteps],dim=0),
                encoder_hidden_states=torch.cat([encoder_hidden_states,negative_prompt_embeds],dim=0),
                attention_kwargs=None,
                return_dict=False,
            )[0]
            model_pred, uncond_pred = pred.chunk(2)
            pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
    else:
        with torch.autocast("cuda", torch.bfloat16):
            pred= transformer(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
            )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob


def compute_reference_log_probs(
    args,
    transformer,
    sigma_schedule,
    all_latents,
    timesteps,
    encoder_hidden_states,
    negative_prompt_embeds,
):
    reference_log_probs = []
    with reference_policy_context(transformer):
        with torch.no_grad():
            for step_idx in range(timesteps.shape[1]):
                log_prob = grpo_one_step(
                    args,
                    all_latents[:, step_idx],
                    all_latents[:, step_idx + 1],
                    encoder_hidden_states,
                    negative_prompt_embeds,
                    transformer,
                    timesteps[:, step_idx],
                    step_idx,
                    sigma_schedule,
                )
                reference_log_probs.append(log_prob)
    return torch.stack(reference_log_probs, dim=1)



def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []  
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    with rollout_precision_context(transformer):
        for index, batch_idx in enumerate(batch_indices):
            batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
            batch_negative_prompt_embeds = negative_prompt_embeds[batch_idx]
            batch_caption = [caption[i] for i in batch_idx]
            if not args.init_same_noise:
                input_latents = torch.randn(
                        (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                        device=device,
                        dtype=torch.bfloat16,
                    )
            grpo_sample=True
            progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
            with torch.no_grad():
                z, latents, batch_latents, batch_log_probs = run_sample_step(
                    args,
                    input_latents.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,     
                    batch_encoder_hidden_states,
                    batch_negative_prompt_embeds, 
                    grpo_sample,
                )
            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)
            vae.enable_tiling()
            
            video_processor = VideoProcessor(vae_scale_factor=8)
            rank = int(os.environ["RANK"])

            
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                        latents.device, latents.dtype
                    )
                    latents = latents / latents_std + latents_mean
                    video = vae.decode(latents, return_dict=False)[0]
                    decoded_video = video_processor.postprocess_video(video)
            export_to_video(decoded_video[0], f"./videos/wan_2_1_{rank}_{index}.mp4", fps=24)

            if args.use_hpsv2:
                with torch.no_grad():
                    image_path = video_first_frame_to_pil(f"./videos/wan_2_1_{rank}_{index}.mp4")
                    image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                    # Process the prompt
                    text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)
                    # Calculate the HPS
                    with torch.amp.autocast('cuda'):
                        outputs = reward_model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image)
                    all_rewards.append(hps_score)

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)

    return all_rewards, all_latents, all_log_probs, sigma_schedule


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def normalize_caption_batch(caption) -> list[str]:
    """
    Normalize caption batches from the dataloader/collate path.

    The custom collate functions build captions with `zip(*batch)`, which yields
    a tuple. Some older environments happened to tolerate this downstream, but
    the training step should not rely on the exact sequence container type.
    """
    if isinstance(caption, str):
        return [caption]
    if isinstance(caption, Sequence):
        return list(caption)
    raise ValueError(f"Unsupported caption type: {type(caption)}")

def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    caption,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
):
    total_loss = 0.0
    optimizer.zero_grad()
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        negative_prompt_embeds = repeat_tensor(negative_prompt_embeds)

        caption = [
            item
            for item in normalize_caption_batch(caption)
            for _ in range(args.num_generations)
        ]

    reward, all_latents, rollout_log_probs, sigma_schedule = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            encoder_hidden_states, 
            negative_prompt_embeds, 
            reward_model,
            tokenizer,
            caption,
            preprocess_val,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    behavior_log_probs = rollout_log_probs[:, :-1].detach()
    reference_log_probs = None
    if should_compute_reference_log_probs(args):
        reference_log_probs = compute_reference_log_probs(
            args,
            transformer,
            sigma_schedule,
            all_latents[:, :-1],
            timesteps[:, :-1],
            encoder_hidden_states,
            negative_prompt_embeds,
        )
    old_log_probs = reference_log_probs if use_reference_old_log_probs(args) else behavior_log_probs
    corr_weights = None
    corr_stats = {}
    if should_apply_rollout_correction(args):
        corr_weights = compute_rollout_tis_weights(
            args,
            old_log_probs=old_log_probs,
            behavior_log_probs=behavior_log_probs,
        )
        corr_stats = summarize_rollout_tis_weights(
            corr_weights=corr_weights,
            old_log_probs=old_log_probs,
            behavior_log_probs=behavior_log_probs,
            threshold=args.rollout_is_threshold,
        )

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": old_log_probs,
        "behavior_log_probs": behavior_log_probs,
        "rewards": reward.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "negative_prompt_embeds": negative_prompt_embeds,
    }
    if reference_log_probs is not None:
        samples["reference_log_probs"] = reference_log_probs
    if corr_weights is not None:
        samples["corr_weights"] = corr_weights
    gathered_reward = gather_tensor(samples["rewards"])
    reward_mean = gathered_reward.mean().item()
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        with open(args.reward_log_path, 'a') as f:
            f.write(f"{reward_mean}\n")

    #计算advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    keys_to_shuffle = ["timesteps", "latents", "next_latents", "log_probs", "behavior_log_probs"]
    if "reference_log_probs" in samples:
        keys_to_shuffle.append("reference_log_probs")
    if "corr_weights" in samples:
        keys_to_shuffle.append("corr_weights")
    for key in keys_to_shuffle:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    grad_norm = torch.zeros((), device=device, dtype=torch.float32)
    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["negative_prompt_embeds"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ppo_ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])
            corr_weight_t = (
                sample["corr_weights"][:, _].to(torch.float32)
                if "corr_weights" in sample
                else torch.ones_like(ppo_ratio, dtype=torch.float32)
            )

            unclipped_loss = -advantages * ppo_ratio
            clipped_loss = -advantages * torch.clamp(
                ppo_ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            policy_loss = torch.maximum(unclipped_loss, clipped_loss)
            loss = torch.mean(corr_weight_t * policy_loss) / (args.gradient_accumulation_steps * train_timesteps)
            if args.reference_policy_kl_coef > 0:
                reference_log_probs = sample["reference_log_probs"][:,_]
                approx_kl = torch.exp(reference_log_probs - new_log_probs) - (reference_log_probs - new_log_probs) - 1.0
                loss = loss + args.reference_policy_kl_coef * approx_kl.mean() / (
                    args.gradient_accumulation_steps * train_timesteps
                )

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("reward", sample["rewards"].item())
            print("ppo_ratio", ppo_ratio)
            print("corr_weight", corr_weight_t)
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
        dist.barrier()
    return total_loss, grad_norm.item(), reward_mean, corr_stats


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    _patch_rms_norm_forward_for_mixed_precision()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    preprocess_val = None
    processor = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                '/share/models/dancegrpo/hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        cp = "/share/models/dancegrpo/hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()


    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    
    transformer = WanTransformer3DModel.from_pretrained(    
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype = torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    ).to(device)

    if _needs_mx_linear_conversion(args.train_precision, args.rollout_precision):
        main_print(f"--> Applying MX converters (train={args.train_precision}, rollout={args.rollout_precision})")
        transformer = build_mx_linear_converters(
            transformer,
            world_size,
        )

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    # Wan forward queries submodule.parameters() (e.g. time_embedder),
    # so keep original parameter views when using FSDP.
    fsdp_kwargs["use_orig_params"] = True
    transformer = FSDP(transformer, **fsdp_kwargs,)
    configure_precision_switch_controller(
        transformer,
        train_precision=args.train_precision,
        rollout_precision=args.rollout_precision,
    )
    main_print(f"--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    #vae.enable_tiling()

    if rank <= 0:
        project = "wan_2_1"
        wandb.init(project=project, config=args)
        run_id = wandb.run.id if wandb.run is not None else "no_wandb_run"
        args.reward_log_path = f"./reward_{run_id}.txt"
    else:
        args.reward_log_path = "./reward_rank_nonzero.txt"

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )


    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch       
        for step, (prompt_embeds, negative_prompt_embeds, caption) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            start_time = time.time()
            if (step-1) % args.checkpointing_steps == 0 and step!=1:
                cpu_state = transformer.state_dict()
                if rank <= 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{step}-{epoch}")
                    os.makedirs(save_dir, exist_ok=True)
                    # save using safetensors
                    weight_path = os.path.join(save_dir,
                                            "diffusion_pytorch_model.safetensors")
                    save_file(cpu_state, weight_path)
                    config_dict = dict(transformer.config)
                    if "dtype" in config_dict:
                        del config_dict["dtype"]  # TODO
                    config_path = os.path.join(save_dir, "config.json")
                    # save dict as json
                    with open(config_path, "w") as f:
                        json.dump(config_dict, f, indent=4)
                main_print(f"--> checkpoint saved at step {step}")
                dist.barrier()
            if step>args.max_train_steps:
                break
            loss, grad_norm, reward_mean, corr_stats = train_one_step(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                negative_prompt_embeds, 
                caption,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
            )

    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "reward_mean": reward_mean,
                        **corr_stats,
                    },
                    step=step,
                )



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--cfg_infer",
        type = float,
        default=5.0,
        help="cfg for training",
    )
    parser.add_argument(
        "--old_log_prob_source",
        type=str,
        default="auto",
        choices=["auto", "behavior", "reference"],
        help="Source for old-policy log probs. 'auto' uses reference policy when train and rollout precisions differ.",
    )
    parser.add_argument(
        "--reference_policy_kl_coef",
        type=float,
        default=0.0,
        help="Optional KL penalty coefficient to keep the train policy close to the train-precision reference policy.",
    )
    parser.add_argument(
        "--rollout_correction_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "timestep_tis"],
        help="Rollout correction mode. 'auto' disables correction when train and rollout precisions match, otherwise enables timestep TIS with a reference old policy.",
    )
    parser.add_argument(
        "--rollout_is_threshold",
        type=float,
        default=2.0,
        help="Upper cap applied to timestep importance weights for rollout correction.",
    )
    parser.add_argument(
        "--rollout_is_batch_normalize",
        action="store_true",
        default=False,
        help="Normalize timestep importance weights by their batch mean after clipping.",
    )
    parser.add_argument(
        "--rollout_is_log_bound",
        type=float,
        default=20.0,
        help="Symmetric clamp bound applied to log importance ratios before exponentiation.",
    )
    parser.add_argument(
        "--train_precision",
        type=str,
        default="bf16",
        choices=["bf16", "mxfp8", "mxfp4"],
        help="Training precision. Supported combinations are bf16-bf16, bf16-mxfp8, mxfp8-mxfp8, mxfp8-mxfp4, and mxfp8-nvfp4.",
    )
    parser.add_argument(
        "--rollout_precision",
        type=str,
        default="bf16",
        choices=["bf16", "mxfp8", "mxfp4", "nvfp4"],
        help="Rollout precision. Supported combinations are bf16-bf16, bf16-mxfp8, mxfp8-mxfp8, mxfp8-mxfp4, and mxfp8-nvfp4.",
    )
    




    args = parser.parse_args()
    main(args)