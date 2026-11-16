#!/usr/bin/env bash
# setup.sh — 魔搭 Notebook 上一键准备环境（约 3-5 分钟）
# 用法：bash scripts/setup.sh
set -e

WS=${WS:-/mnt/workspace}
PROJ=$WS/sft-math
PIP="pip install -q -i https://mirrors.aliyun.com/pypi/simple"

echo "=== [1/4] 基础依赖 ==="
$PIP -U modelscope

echo "=== [2/4] 下载基座 Qwen3-4B-Instruct-2507（内网，~8GB） ==="
if [ -d "$WS/models/qwen3-4b/config.json" ]; then
  echo "  已存在，跳过"
else
  modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir "$WS/models/qwen3-4b"
fi

echo "=== [3/4] LLaMA-Factory ==="
if [ -d "$WS/LLaMA-Factory" ]; then
  echo "  已存在，跳过 clone"
else
  # 国内网络若 GitHub 慢，可换：git clone --depth 1 https://gitee.com/mirrors/LLaMA-Factory.git
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$WS/LLaMA-Factory"
fi
cd "$WS/LLaMA-Factory"
# 注意：不加 [torch]，沿用镜像里预装的 torch，避免重装 CUDA 版本踩坑
$PIP -e ".[metrics]"

echo "=== [4/4] 自检 ==="
cd "$PROJ"
llamafactory-cli version
python - <<'PY'
import torch, transformers, datasets, peft
print("torch      ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
print("peft        ", peft.__version__)
PY
echo
echo "环境就绪。下一步：bash scripts/train.sh smoke"
