# Qwen3-4B 数学推理 SFT（LoRA）

对 Qwen3-4B-Instruct-2507 做 LoRA 后训练，提升其在中文小学数学题上的答案准确率，并在公开基准 CMath 上给出「训前 vs 训后」对照。

本仓库只做**后训练工程**：数据构建 → LoRA 训练 → 基准评测 → 产物发布。不涉及任何应用层业务逻辑。

---

## 1. 任务定义

| 项 | 内容 |
|---|---|
| 基座 | `Qwen/Qwen3-4B-Instruct-2507`（非 thinking 版） |
| 训练数据 | `AI-MO/NuminaMath-CoT`，86 万条带完整推理链的数学题解 |
| 训练方式 | LoRA（r=16, α=32），rank 只加在注意力 + MLP 投影层 |
| 评测基准 | `evalscope/cmath` test 集（1098 题，中文，答案可精确判分） |
| 指标 | 答案准确率（数值精确匹配，非人工评分） |
| 目标 | 让模型从「给出答案但推理不完整/易错」变成「稳定输出推理链并答对」 |

为什么选数学：**答案可程序化判分**。这是唯一不需要人工标注就能得到客观分数、且能与公开榜单对标的任务类型，适合验证后训练链路是否真的有效。

---

## 2. 数据

### 来源

| 用途 | 数据集 | 规模 | 许可 |
|---|---|---|---|
| 训练 | `AI-MO/NuminaMath-CoT` | 859,494 条（本仓库抽 20,000） | Apache-2.0 |
| 评测 | `evalscope/cmath` | test 1098 / dev 600 | CC-BY-4.0 |

### 构建口径（`data/build_dataset.py`）

从 859,494 条原始样本中清洗出 809,083 条可用样本，再分层抽 20,000 条训练 + 600 条验证：

1. **长度过滤**：题面 30–1500 字符、解答 100–4000 字符
   —— 解答超过 2048 token 的样本会被 `cutoff_len` 截断，模型学不到结论部分，反而学会「推理到一半停住」
2. **按题面 MD5 去重**
3. **分层采样**：按 `source` 比例抽，单一 source 上限 6000（防止一个题库独占）
4. **固定随机种子 42**，结果可复现

抽样后的分布：

| source | 条数 | 占比 |
|---|---|---|
| cn_k12 | 6,185 | 30.9% |
| orca_math | 3,759 | 18.8% |
| synthetic_math | 3,733 | 18.7% |
| olympiads | 3,625 | 18.1% |
| synthetic_amc | 1,576 | 7.9% |
| aops_forum | 754 | 3.8% |
| math / amc_aime / gsm8k | 368 | 1.8% |

**注意 `cn_k12` 不等于中文**：该子集是中国中小学题库，但 NuminaMath 发布时已把题干翻译成英文。全库中文字符占比实测 **0.0001**。这一点在下面的「已知限制」里有对应说明。

### 复现

```bash
# 下载原始数据（内网约 1-2 分钟）
modelscope download --dataset AI-MO/NuminaMath-CoT --local_dir data/numina
modelscope download --dataset evalscope/cmath      --local_dir data/cmath

# 生成训练集 + dataset_info.json
python data/build_dataset.py --n-train 20000 --n-val 600
```

---

## 3. 训练

### 环境

| 项 | 内容 |
|---|---|
| 硬件 | 单卡 A10 24GB（魔搭 Notebook 免费额度） |
| 框架 | LLaMA-Factory |
| 精度 | bf16 + gradient checkpointing |
| 显存占用 | 实测约 19–21 GB |

### 超参（`configs/lora_qwen3_4b.yaml`）

| 参数 | 值 | 理由 |
|---|---|---|
| `template` | `qwen3_nothink` | 必须匹配基座，否则训练/推理的对话格式不一致 |
| `lora_rank` / `alpha` | 16 / 32 | α=2r 是常用配比；任务为领域能力注入，不需要更大 rank |
| `lora_target` | `all` | 注意力 + MLP 全覆盖，可训练参数仍只占约 1% |
| `cutoff_len` | 2048 | 覆盖 95% 样本（解答中位 1015 字符 ≈ 500 token） |
| batch | 2 × 8 = 16 | A10 上的稳定值 |
| lr | 1e-4，cosine，warmup 5% | LoRA 常用学习率 |
| epochs | 3 | 20,000 条样本，约 3,750 步 |

### 执行

```bash
bash scripts/setup.sh                 # 装依赖 + 拉基座（3-5 分钟）
bash scripts/train.sh smoke           # 50 步冒烟，5 分钟，先验链路
bash scripts/train.sh full            # 正式训练，约 2-2.5 小时
```

**先 smoke 后 full 是硬性流程**：冒烟通过的标准是 loss 从 2.x 开始下降、显存 < 22GB、`out/smoke` 出现 `adapter_model.safetensors`。跳过这步的代价是「跑 2 小时才发现模板选错」。

---

## 4. 结果

> 待填入。执行 `bash scripts/run_eval.sh` 后自动生成。

| 模型 | CMath 准确率 | grade 1 | grade 2 | grade 3 | grade 4 | grade 5 | grade 6 |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507（基线） | — | — | — | — | — | — | — |
| + LoRA SFT（本仓库） | — | — | — | — | — | — | — |
| Δ | — | — | — | — | — | — | — |

---

## 已知限制

1. **输出语言偏移**：训练语料是英文推理链，SFT 后模型在中文题上可能出现「中文提问 → 英文推理」的混输现象。评测指标（数值答案）不受影响，但对话体验会下降。推理脚本已在 system prompt 中要求中文作答，属缓解而非根治。根治需要中文 CoT 语料（Math23K / APE210K 一类），未纳入本次范围。
2. **只覆盖 SFT 阶段**：无 RL / 偏好优化。下一阶段可用同一批数据做 GRPO —— 数学题的答案对错本身就是可验证奖励，不需要训练奖励模型。
3. **CMath 偏小学**：1–6 年级，对 4B 模型而言上限不高（部分题原本就能做对），提升幅度会小于初中/高中难度的基准。若要更陡的提升曲线，应换 MATH-500 或更难的集合。
4. **单次判分**：不做多采样投票（self-consistency），报的是保守准确率。

---

## 目录

```
sft-math/
├── data/
│   ├── build_dataset.py     # 清洗 + 分层采样 + 导出 sharegpt
│   ├── numina/              # 原始 parquet（git 忽略）
│   └── cmath/               # 评测集（git 忽略）
├── configs/
│   ├── lora_qwen3_4b.yaml
│   └── lora_qwen3_4b_smoke.yaml
├── scripts/
│   ├── setup.sh             # 环境 + 基座
│   ├── train.sh             # 训练入口
│   ├── run_eval.sh          # 基线与 SFT 对照
│   └── eval.py              # CMath 判分
└── README.md
```
