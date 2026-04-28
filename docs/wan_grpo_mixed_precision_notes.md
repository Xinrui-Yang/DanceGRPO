# Wan 2.1 GRPO Mixed Precision Notes

## Summary

This document summarizes the current understanding of the Wan 2.1 GRPO training path in this repository, the mixed-precision design we refined during this session, and the code changes that were made.

The primary codepath discussed here is:

- [fastvideo/train_grpo_wan_2_1.py](/workspace/DanceGRPO/fastvideo/train_grpo_wan_2_1.py)

Related launcher and dataset files:

- [scripts/finetune/finetune_wan_2_1_grpo.sh](/workspace/DanceGRPO/scripts/finetune/finetune_wan_2_1_grpo.sh)
- [fastvideo/dataset/latent_wan_2_1_rl_datasets.py](/workspace/DanceGRPO/fastvideo/dataset/latent_wan_2_1_rl_datasets.py)
- [fastvideo/utils/mxfp8_backward_padding.py](/workspace/DanceGRPO/fastvideo/utils/mxfp8_backward_padding.py)


## Project Understanding

### Training setup

`train_grpo_wan_2_1.py` trains Wan 2.1 with GRPO using:

- FSDP for distributed training
- TorchTitan model conversion to replace selected `nn.Linear` modules with TorchAO `MXLinear`
- optional mixed train/rollout precision
- optional rollout correction based on reference-policy log-probs and timestep importance sampling
- optional KL regularization against a reference policy

The script uses a single shared transformer instance. It does not maintain separate train and rollout models.

### Precision design

The current design supports only these four train/rollout precision pairs:

- `bf16 -> bf16`
- `bf16 -> mxfp8`
- `mxfp8 -> mxfp8`
- `mxfp8 -> mxfp4`

This restriction is intentional and is enforced in the script. The goal is to support the combinations that have a clear implementation and validation story without silently enabling more fragile combinations.

### How precision switching works

The selected linear submodules are first converted once into `MXLinear` modules. Runtime precision switching is then handled by a `PrecisionSwitchController`:

- train path uses `train_precision`
- rollout sampling path uses `rollout_precision`
- reference-policy log-prob recomputation uses `train_precision`

The controller changes runtime behavior by:

- routing `bf16` through a normal `F.linear(...)` path
- routing `mxfp8` and `mxfp4` through `MXLinear` with the corresponding MX config

This means we do not swap between two model instances and we do not repeatedly rebuild modules.


## Mixed Precision Path Decisions

### Converter entrypoint

Internally, the MX conversion helper was clarified to mean "convert selected linear layers to `MXLinear`", not "force everything to mxfp8".

Current behavior:

- `build_mx_linear_converters(...)` is the semantic entrypoint
- `build_torchtitan_mxfp8_converters(...)` remains as a backward-compatible wrapper

The external CLI did not change.

### MX recipe resolution

We introduced explicit resolution of MX configs:

- `mxfp8` uses `mxfp8_cublas`
- `mxfp4` prefers `mxfp4_cutlass`

The resolved recipe is now logged at startup so that a run does not silently appear to be using `mxfp4` while actually using the emulated fallback.

### Rollout context granularity

Originally, rollout precision switching happened inside the per-sample loop in `sample_reference_model()`.

This was changed so that:

- one full rollout block enters rollout precision once
- all per-sample generations run under that single rollout context
- the code returns to train precision once after the rollout block finishes

This reduces unnecessary train/rollout/train switching without changing the numerical path.

### Rollout trajectory dtype

We intentionally kept rollout trajectory storage in `float32`.

Reason:

- rollout latents are later reused for PPO log-prob computation and optional reference log-prob recomputation
- keeping these tensors in `float32` is the safer default for consistency and debugging

We explicitly did not add an aggressive memory-saving mode in this round.


## Reference Policy and Correction Logic

### Old log-prob source

The script supports:

- `behavior`
- `reference`
- `auto`

Current intended behavior:

- if train precision equals rollout precision, `auto` resolves to behavior log-probs
- if train precision differs from rollout precision, `auto` resolves to reference log-probs

### Rollout correction mode

The script supports:

- `none`
- `timestep_tis`
- `auto`

Current intended behavior:

- if train precision equals rollout precision, `auto` resolves to `none`
- if train precision differs from rollout precision, `auto` resolves to `timestep_tis`
- mixed precision correction requires reference old log-probs

This keeps the mixed train/rollout path numerically interpretable.


## Runtime and Stability Patches

### RMSNorm mixed-dtype patch

The script patches `torch.nn.RMSNorm.forward` so that if the input activation dtype differs from the weight dtype on CUDA, the weight is cast to the input dtype before calling `F.rms_norm(...)`.

Purpose:

- avoid dtype mismatch failures when mixed train/rollout precision is active

### MXFP8 backward row-padding patch

`fastvideo/utils/mxfp8_backward_padding.py` patches TorchAO backward behavior to tolerate row counts that are not multiples of the MX block size during backward.

Purpose:

- avoid failures in MX backward for variable token row counts

This patch affects backward behavior for MX paths. It does not affect rollout no-grad inference directly.


## Torch 2.10 Issue: Caption Became Tuple

### Symptom

After upgrading Torch to `2.10.0`, training failed with:

```text
ValueError: Unsupported caption type: <class 'tuple'>
```

This happened in `train_one_step()` when `args.use_group` was enabled.

### Why it happened

The Wan RL dataset collate function does:

```python
prompt_embeds, prompt_attention_masks, caption = zip(*batch)
return prompt_embeds, prompt_attention_masks, caption
```

That means `caption` is naturally returned as a `tuple`.

The training code previously assumed:

- `caption` is either `str`
- or `caption` is `list`

and rejected everything else.

That assumption was too narrow. The upgrade made the mismatch visible, but the underlying issue was that the training step relied on a specific container type instead of normalizing the batch input.

### Fix

We added `normalize_caption_batch(...)` to convert:

- `str -> [str]`
- `tuple -> list`
- `list -> list`

Then the `use_group` path repeats the normalized captions without caring whether the original container was a tuple or a list.


## Code Changes Made In This Session

### Mixed precision robustness and clarity

Implemented in `train_grpo_wan_2_1.py`:

- added semantic MX converter entrypoint `build_mx_linear_converters(...)`
- kept `build_torchtitan_mxfp8_converters(...)` as compatibility wrapper
- added resolved MX config bookkeeping with recipe and fallback metadata
- added startup logging for:
  - resolved train precision
  - resolved rollout precision
  - whether MX conversion is enabled
  - whether reference old log-probs are enabled
  - whether rollout TIS correction is enabled
  - which MX recipe each non-bf16 precision actually uses
- moved rollout precision context to wrap the full rollout generation block
- removed the ineffective `z.to(torch.bfloat16)` line and documented why rollout trajectories remain in `float32`

### Torch 2.10 caption compatibility

Implemented in `train_grpo_wan_2_1.py`:

- added `normalize_caption_batch(...)`
- updated the `use_group` caption expansion logic to accept sequence-like batch inputs, especially tuples produced by `zip(*batch)`


## Validation Done During This Session

We performed local checks on the modified training script:

- Python syntax check with `python -m py_compile`
- smoke tests for supported precision pair resolution
- smoke tests for rollout-correction auto-resolution
- toy `MXLinear` controller switching tests for:
  - `mxfp8 -> mxfp4`
  - `bf16 -> mxfp8`
- smoke test for caption normalization with:
  - `str`
  - `list`
  - `tuple`

We did not run a full distributed training job as part of these edits.


## Current Practical Guidance

### If you want mxfp8 training and mxfp4 rollout

Use:

- `--train_precision mxfp8`
- `--rollout_precision mxfp4`

Do not use `--use_torchtitan_mxfp8` for this case, because that flag is intentionally only shorthand for:

- `--train_precision mxfp8 --rollout_precision mxfp8`

### If a run behaves unexpectedly

Check startup logs for:

- resolved train/rollout precisions
- resolved old log-prob source
- resolved rollout correction mode
- resolved MX recipe, especially whether `mxfp4` fell back to `mxfp4_emulated`


## Follow-up Suggestions

The same tuple/list caption assumption appears in other GRPO training scripts in this repository. If those scripts are expected to run under the upgraded Torch environment, the same normalization fix should be applied there as well:

- `fastvideo/train_grpo_flux.py`
- `fastvideo/train_grpo_flux_lora.py`
- `fastvideo/train_grpo_hunyuan.py`
- `fastvideo/train_grpo_qwenimage.py`
- `fastvideo/train_grpo_qwenimage_edit.py`
- `fastvideo/train_grpo_skyreels_i2v.py`
