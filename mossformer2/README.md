# MossFormer2 Training Guide

This folder contains the training and inference code for MossFormer2 speech separation in the current workspace. The codebase now supports two recurrent variants inside the separator:

- `fsmn`: original baseline branch
- `mamba2`: new recurrent branch based on the local `mamba/` package

The default model families are:

- `MossFormer2_SS_8K`
- `MossFormer2_SS_16K`

## 1. What changed in this workspace

Compared with the upstream training code, this workspace now includes:

- `mamba2` integration in the recurrent branch
- preflight checks in `train.py` and `inference.py`
- safer checkpoint resume behavior when switching architectures
- updated `train.sh` supporting both single-GPU and multi-GPU runs
- ready-to-edit Libri2Mix 8k configs for `mamba2`

## 2. Recommended environment

Use:

- Linux
- NVIDIA GPU
- Python `3.9` or `3.10`
- PyTorch with CUDA already working before installing `mamba`

The local `mamba/` package in this repository declares `requires-python >= 3.9`, so `python=3.8` is not recommended for the `mamba2` path.

## 3. Environment setup

Create and activate a clean Conda environment:

```bash
conda create -n mossformer-mamba2 python=3.9 -y
conda activate mossformer-mamba2
```

Install PyTorch with CUDA support first. Example for CUDA 12.1:

```bash
export REPO_ROOT=/path/to/mossformer
cd "$REPO_ROOT"
pip install --upgrade pip setuptools wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the MossFormer2 training dependencies:

```bash
cd "$REPO_ROOT"
pip install -r mossformer2/requirements-mamba2.txt
```

Install the local `mamba/` package:

```bash
cd "$REPO_ROOT"
pip install -e ./mamba --no-build-isolation
```

If `pip install -e ./mamba --no-build-isolation` fails, fix the CUDA / PyTorch environment first before attempting training.

## 4. Expected dataset format

The trainer expects `.scp` files with one mixture per line.

For 2-speaker separation:

```text
path/to/mix.wav path/to/s1.wav path/to/s2.wav
```

For more speakers:

- keep the first column as the mixture path
- append one clean target path per speaker

The dataloader used by speech separation training reads:

- first column as input mixture
- remaining columns as target sources

## 5. Libri2Mix 8k configs already added

Two configs are ready for Libri2Mix 8k training from scratch:

- `config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_small.yaml`
- `config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_full.yaml`

These configs currently assume:

- `train = 30h`
- `dev = 10h`
- `test = 5h`
- `8kHz`
- `4s`

You only need to edit:

- `tr_list`
- `cv_list`
- optionally `tt_list`

The current placeholders are:

- `data/libri2mix_8k_train_30h.scp`
- `data/libri2mix_8k_dev_10h.scp`
- optional `data/libri2mix_8k_test_5h.scp`

## 6. Important Mamba2 preflight checks

When `recurrent_type="mamba2"`, the code now fails early if:

- `use_cuda` is disabled
- `torch.cuda.is_available()` is `False`
- the local `mamba/` package is not importable
- `recurrent_inner_channels * mamba_expand` is not divisible by `mamba_headdim`
- `recurrent_inner_channels * mamba_expand + 2 * mamba_d_state` is not a multiple of `8`

This is intentional. It is better to fail early than to crash deep inside the CUDA / Triton path.

## 7. Training configs

### 7.1. Paper-like full

Use:

- `config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_full.yaml`

This keeps the larger MossFormer2-style setup:

- `encoder_embedding_dim: 512`
- `mossformer_sequence_dim: 512`
- `num_mossformer_layer: 24`
- `recurrent_inner_channels: 256`
- `mamba_d_state: 64`

### 7.2. Smaller practical config

Use:

- `config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_small.yaml`

This is the cheaper and more practical option for initial training runs:

- `encoder_embedding_dim: 384`
- `mossformer_sequence_dim: 384`
- `num_mossformer_layer: 25`
- `recurrent_inner_channels: 256`
- `mamba_d_state: 64`

## 8. Start training

The updated `train.sh` supports both single-GPU and multi-GPU execution.

### 8.1. Single GPU

Small config:

```bash
cd "$REPO_ROOT/mossformer2"
GPU_ID=0 \
N_GPU=1 \
CONFIG_PTH=config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_small.yaml \
CHECKPOINT_DIR=checkpoints/MossFormer2_SS_8K_Libri2Mix_Mamba2_small \
bash train.sh
```

Full config:

```bash
cd "$REPO_ROOT/mossformer2"
GPU_ID=0 \
N_GPU=1 \
CONFIG_PTH=config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_full.yaml \
CHECKPOINT_DIR=checkpoints/MossFormer2_SS_8K_Libri2Mix_Mamba2_full \
bash train.sh
```

### 8.2. Multi GPU

Example with 2 GPUs:

```bash
cd "$REPO_ROOT/mossformer2"
GPU_ID=0,1 \
N_GPU=2 \
CONFIG_PTH=config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_small.yaml \
CHECKPOINT_DIR=checkpoints/MossFormer2_SS_8K_Libri2Mix_Mamba2_small_2gpu \
bash train.sh
```

`train.sh` will automatically:

- run `python train.py` when `N_GPU=1`
- run `torchrun` when `N_GPU>1`

## 9. Checkpoints and resume behavior

Important rules:

- use `train_from_last_checkpoint=1` only when resuming the same architecture
- if you switch between `fsmn` and `mamba2`, prefer:

```bash
TRAIN_FROM_LAST_CHECKPOINT=0
INIT_CHECKPOINT_PATH=/path/to/compatible_checkpoint.pt
```

This workspace already includes safer fallback logic if optimizer state is incompatible, but the recommended workflow is still:

- same architecture: resume
- different architecture: partial load + fresh training state

## 10. Inference

`inference.py` also includes the same `mamba2` preflight checks.

The inference configs live in:

- `config/inference/MossFormer2_SS_8K.yaml`
- `config/inference/MossFormer2_SS_16K.yaml`

Before inference with `mamba2`, make sure the inference config matches the training recurrent settings.

## 11. Practical recommendations

For first runs:

1. Start with the small 8k Libri2Mix config.
2. Keep `tt_list` disabled during training to save time.
3. Confirm:
   - loss is finite
   - checkpoints are written correctly
   - validation runs complete
4. Only move to the full config after the small run is stable.

## 12. Files to know

- `train.py`: training entrypoint
- `inference.py`: inference entrypoint
- `train.sh`: practical launcher for 1 GPU or multi GPU
- `requirements-mamba2.txt`: minimal Python dependencies for this workspace
- `config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_small.yaml`
- `config/train/MossFormer2_SS_8K_Libri2Mix_Mamba2_full.yaml`

## 13. Common issues

### Mamba2 import fails

Check:

- Python version is `>= 3.9`
- PyTorch CUDA build is working
- `pip install -e ./mamba --no-build-isolation` completed successfully

### Config rejected by preflight

Check:

- `mamba_headdim`
- `recurrent_inner_channels`
- `mamba_expand`
- `mamba_d_state`

### Training is too slow

Try:

- the `small` config first
- single-GPU sanity run before long training
- smaller validation during tuning
- no `tt_list` during the training loop
