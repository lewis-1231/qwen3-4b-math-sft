#!/usr/bin/env bash
# run_eval.sh — 基线与 SFT 后各跑一次 CMath，产出对比
# 用法：bash scripts/run_eval.sh            # 全量 1098 题
#      LIMIT=100 bash scripts/run_eval.sh  # 先抽 100 题验证脚本能跑通
set -e

PROJ=/mnt/workspace/sft-math
cd "$PROJ"
MODEL=/mnt/workspace/models/qwen3-4b
ADAPTER=${ADAPTER:-/mnt/workspace/out/qwen3-4b-math-lora}
LIMIT=${LIMIT:-0}

mkdir -p results

echo "=== 基线（原基座，无 adapter） ==="
python scripts/eval.py --model "$MODEL" \
  --data data/cmath/cmath_test.jsonl \
  --out results/base.csv --limit "$LIMIT"

echo
echo "=== SFT 后（基座 + LoRA） ==="
python scripts/eval.py --model "$MODEL" --adapter "$ADAPTER" \
  --data data/cmath/cmath_test.jsonl \
  --out results/sft.csv --limit "$LIMIT"

echo
echo "=== 对比 ==="
python - <<'PY'
import pandas as pd
b = pd.read_csv("results/base.csv"); s = pd.read_csv("results/sft.csv")
print(f"基线  {b['correct'].mean():.4f}   ({int(b['correct'].sum())}/{len(b)})")
print(f"SFT   {s['correct'].mean():.4f}   ({int(s['correct'].sum())}/{len(s)})")
print(f"提升  {(s['correct'].mean()-b['correct'].mean())*100:+.2f} pt")
print()
t = pd.DataFrame({
    "base": b.groupby("grade")["correct"].mean(),
    "sft": s.groupby("grade")["correct"].mean(),
}).round(4)
t["delta_pt"] = ((t["sft"] - t["base"]) * 100).round(2)
print(t.to_string())
PY
