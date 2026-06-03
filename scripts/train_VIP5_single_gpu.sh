# Single-GPU convenience launcher.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_VIP5_single_gpu.sh [split] [img_feat_type] [img_feat_size_ratio] [reduction_factor] [epoch]
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_VIP5_single_gpu.sh toys vitb32 2 8 20
#
# Lower the batch size if you hit CUDA out-of-memory, e.g.:
#   BATCH_SIZE=12 CUDA_VISIBLE_DEVICES=0 bash scripts/train_VIP5_single_gpu.sh toys

#!/bin/bash

split=${1:-toys}
img_feat_type=${2:-vitb32}
img_feat_size_ratio=${3:-2}
reduction_factor=${4:-8}
epoch=${5:-20}
batch_size=${BATCH_SIZE:-36}
port=${MASTER_PORT:-13579}

name=$split-$img_feat_type-$img_feat_size_ratio-$reduction_factor-$epoch
output=snap/$name

mkdir -p snap log

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port $port \
    src/train.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train $split \
        --valid $split \
        --batch_size $batch_size \
        --optim adamw \
        --warmup_ratio 0.1 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 5.0 \
        --losses 'sequential,direct,explanation' \
        --backbone 't5-small' \
        --output $output \
        --epoch $epoch \
        --use_adapter \
        --unfreeze_layer_norms \
        --reduction_factor $reduction_factor \
        --use_single_adapter \
        --max_text_length 1024 \
        --gen_max_length 64 \
        --image_feature_type $img_feat_type \
        --image_feature_size_ratio $img_feat_size_ratio \
        --whole_word_embed \
        --category_embed > log/$name.log 2>&1
