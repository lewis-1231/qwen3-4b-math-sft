---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
tags:
- lora
- sft
- math
- qwen3
---

# qwen3-4b-math-lora-sft

对 `Qwen/Qwen3-4B-Instruct-2507` 做 LoRA SFT 的实验产物（NuminaMath-CoT 2 万条，3 epoch）。

**结论先说：这是一个负结果。** CMath test（1098 题）准确率从基线 0.9089 **下降到 0.8780（−3.10 pt）**。发布此 adapter 不是作为"更好的模型"，而是作为一次完整后训练实验的可复查证据：强基座 + 3 epoch LoRA SFT 会过拟合，掉分集中在长推理链题目。

## 训练配置

| 项 | 值 |
|---|---|
| 基座 | Qwen/Qwen3-4B-Instruct-2507（非 thinking） |
| 数据 | AI-MO/NuminaMath-CoT 分层采样 20,000 条（seed=42） |
| LoRA | r=16, α=32, target=all, dropout=0.05, 可训练参数 33M（0.81%） |
| 超参 | lr 1e-4 cosine、bs 2×8、cutoff 2048、bf16、3 epoch（3750 步） |
| 硬件 | 单卡 A10 24GB（魔搭免费额度），10.9 s/step |

## 评测结果

CMath test 1098 题，greedy 解码，数值精确匹配：

| 模型 | 准确率 | Δ |
|---|---|---|
| Qwen3-4B-Instruct-2507（基线） | **0.9089** | — |
| + 本 LoRA | 0.8780 | **−3.10 pt** |

分推理步数：1 步题 +2.05 pt，4 步题 −16.47 pt，5 步题 −17.39 pt——多步推理能力被叙述模板替换是主要退化形态。

## 使用

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "<你的用户名>/qwen3-4b-math-lora-sft")
```

## 完整分析

根因定位、分年级/分步数明细、判分脚本与逐题证据见 GitHub 仓库：
**https://github.com/lewis-1231/qwen3-4b-math-sft**
