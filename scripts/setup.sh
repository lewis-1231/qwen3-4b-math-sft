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
if command -v llamafactory-cli >/dev/null 2>&1; then
  echo "  已安装，跳过"
else
  # 优先走阿里云 PyPI（国内秒装）；失败再回退 GitHub 源码
  if pip install -q -i https://mirrors.aliyun.com/pypi/simple llamafactory; then
    echo "  已通过 PyPI 安装 llamafactory"
  else
    echo "  PyPI 失败，回退源码编译"
    if [ ! -d "$WS/LLaMA-Factory" ]; then
      git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$WS/LLaMA-Factory"
    fi
    cd "$WS/LLaMA-Factory"
    pip install -q -e ".[metrics]" -i https://mirrors.aliyun.com/pypi/simple
  fi
fi

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
# 关键自检：模板是否存在，不存在则 smoke 一定失败（早发现省 5 分钟）
try:
    from llamafactory.data.template import TEMPLATES
    ok = "qwen3_nothink" in TEMPLATES
    print("template qwen3_nothink:", "OK" if ok else "缺失！请改用源码版 LLaMA-Factory")
except Exception as e:
    print("template check skipped:", type(e).__name__, e)
PY
echo
echo "环境就绪。下一步：bash scripts/train.sh smoke"
