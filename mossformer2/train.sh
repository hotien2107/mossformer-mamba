#!/bin/sh

set -eu

gpu_id="${GPU_ID:-0}"
n_gpu="${N_GPU:-1}"
network="${NETWORK:-MossFormer2_SS_8K}"
config_pth="${CONFIG_PTH:-config/train/${network}.yaml}"
checkpoint_dir="${CHECKPOINT_DIR:-checkpoints/${network}}"
train_from_last_checkpoint="${TRAIN_FROM_LAST_CHECKPOINT:-0}"
init_checkpoint_path="${INIT_CHECKPOINT_PATH:-None}"
print_freq="${PRINT_FREQ:-50}"
checkpoint_save_freq="${CHECKPOINT_SAVE_FREQ:-2000}"

if [ ! -f "${config_pth}" ]; then
  echo "Config not found: ${config_pth}" >&2
  exit 1
fi

mkdir -p "${checkpoint_dir}"
cp "${config_pth}" "${checkpoint_dir}/config.yaml"

export PYTHONWARNINGS="ignore"
export PYTHONUNBUFFERED="1"

common_args="--config ${config_pth} \
--checkpoint_dir ${checkpoint_dir} \
--train_from_last_checkpoint ${train_from_last_checkpoint} \
--init_checkpoint_path ${init_checkpoint_path} \
--print_freq ${print_freq} \
--checkpoint_save_freq ${checkpoint_save_freq}"

if [ "${n_gpu}" -le 1 ]; then
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python -u -W ignore train.py ${common_args}
else
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  torchrun \
    --nproc_per_node="${n_gpu}" \
    --master_port="$(date '+88%S')" \
    train.py ${common_args}
fi
