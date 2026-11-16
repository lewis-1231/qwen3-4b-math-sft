#!/usr/bin/env bash
# train.sh — 训练入口
# 用法：
#   bash scripts/train.sh smoke   # 50 步冒烟，约 10 分钟，先跑这个
#   bash scripts/train.sh full    # 正式训练 3 epoch ≈ 3750 步 ≈ 11-12 小时（A10 实测 11.2s/步）
# 断点续训（8 小时强制停机后，重启实例先跑这条）：
#   RESUME=/mnt/workspace/out/qwen3-4b-math-lora/checkpoint-3000 bash scripts/train.sh full
#   （checkpoint 序号用 ls /mnt/workspace/out/qwen3-4b-math-lora/ 查看，取最大的）
#   也可用 RESUME=auto 自动取最新 checkpoint
set -e

PROJ=/mnt/workspace/sft-math
cd "$PROJ"

MODE=${1:-smoke}
case "$MODE" in
  smoke) CFG=configs/lora_qwen3_4b_smoke.yaml ;;
  full)  CFG=configs/lora_qwen3_4b.yaml ;;
  *) echo "用法: bash scripts/train.sh [smoke|full]"; exit 1 ;;
esac

# 断点续训：RESUME=auto 或显式 checkpoint 路径
RESUME=${RESUME:-}
if [ "$RESUME" = "auto" ]; then
  RESUME=$(ls -d /mnt/workspace/out/qwen3-4b-math-lora/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [ -z "$RESUME" ] && RESUME=""
fi
EXTRA_ARGS=()
[ -n "$RESUME" ] && EXTRA_ARGS+=("resume_from_checkpoint=$RESUME")

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mnt/workspace/out
echo "=== 开始训练: $CFG ==="
[ -n "$RESUME" ] && echo "=== 从 checkpoint 续训: $RESUME ==="
date
llamafactory-cli train "$CFG" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "/mnt/workspace/out/train_${MODE}.log"
date
echo
echo "adapter 产物："
find /mnt/workspace/out -name "adapter_model.safetensors" -exec ls -lh {} \;
