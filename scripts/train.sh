#!/usr/bin/env bash
# train.sh — 训练入口
# 用法：
#   bash scripts/train.sh smoke   # 50 步冒烟，约 5 分钟，先跑这个
#   bash scripts/train.sh full    # 正式训练 3 epoch，约 2-2.5 小时
set -e

PROJ=/mnt/workspace/sft-math
cd "$PROJ"

MODE=${1:-smoke}
case "$MODE" in
  smoke) CFG=configs/lora_qwen3_4b_smoke.yaml ;;
  full)  CFG=configs/lora_qwen3_4b.yaml ;;
  *) echo "用法: bash scripts/train.sh [smoke|full]"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mnt/workspace/out
echo "=== 开始训练: $CFG ==="
date
llamafactory-cli train "$CFG" 2>&1 | tee "/mnt/workspace/out/train_${MODE}_$(date +%m%d_%H%M).log"
date
echo
echo "adapter 产物："
find /mnt/workspace/out -name "adapter_model.safetensors" -exec ls -lh {} \;
