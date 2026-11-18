# Qwen3-4B 数学推理 SFT（LoRA）

对 Qwen3-4B-Instruct-2507 做 LoRA 后训练，在公开基准 CMath 上给出「训前 vs 训后」对照。

**结论先说**：完整链路已跑通（数据构建 → LoRA SFT → 断点续训 → 基准评测），结果是**负向的**——CMath 从基线 0.9089 掉到 0.8780（−3.10 pt）。根因定位为「强基座 + 3 epoch 过拟合」，完整分析见 [§4](#4-结果) 与 [§5](#5-复盘这个负结果值不值)。

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
| 目标 | 验证「公开 CoT 语料 LoRA 后训练」对强基座是否有效（结论：本配置下无效，见 §4） |

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
bash scripts/train.sh full            # 正式训练
```

**先 smoke 后 full 是硬性流程**：冒烟通过的标准是 loss 从 2.x 开始下降、显存 < 22GB、`out/smoke` 出现 `adapter_model.safetensors`。跳过这步的代价是「跑 2 小时才发现模板选错」。

### 实测训练耗时与断点续训

A10 单卡实测 **10.9 s/step**，3750 步 ≈ **11.4 小时**（README 早期版本估的 2-2.5h 是错的，已按实测修正）。

因此本项目**必须跨会话训练**：魔搭免费实例单次上限 8 小时，训练在第 2392 步被强制中断，
用 `RESUME=auto` 从 checkpoint-2000 续训，第二段 1750 步用时 2 小时 52 分，全流程墙钟时间约 9 小时（含检查点评测与重启损耗）。

```bash
RESUME=auto bash scripts/train.sh full    # 自动找最新 checkpoint 续训
```

另外两个运维细节（都踩过）：

1. **实例重启后系统盘重置**，`pip` 装的 LLaMA-Factory 会消失，但 `/mnt/workspace` 持久化。`train.sh` 已内置「启动前检测不到 `llamafactory-cli` 就自动重装」的自愈逻辑。
2. **实例停机后再启动可能抢不到 GPU**（`nvidia-smi` 报 `No devices were found`，容器层面就没挂上卡）。这不是软件问题，只能停止再启动重新申请；数据在 `/mnt/workspace` 不会丢。**任何时候都不要删除实例**——删除等于清空持久盘。

---

## 4. 结果

评测口径：CMath test 1098 题，greedy 解码（`do_sample=False`），`max_new_tokens=768`，batch 8，答案数值精确匹配（容差 1e-4）。明细见 `results/base.csv`、`results/sft.csv`。

### 4.1 总体：SFT 后掉 3.1 个点

| 模型 | CMath 准确率 | 对/总 | Δ |
|---|---|---|---|
| Qwen3-4B-Instruct-2507（基线） | **0.9089** | 998 / 1098 | — |
| + LoRA SFT（3 epoch，终态 adapter） | 0.8780 | 964 / 1098 | **−3.10 pt** |

这轮实验的结论是**负向**的：在该基座上，3 epoch、lr 1e-4 的 LoRA SFT 没有提升 CMath 答案准确率，反而造成退化。下面把根因定位过程完整记下来——这是本项目真正的产出。

### 4.2 分年级：掉分随难度单调放大

| grade | 基线 | SFT | Δ (pt) | n |
|---|---|---|---|---|
| 1 | 0.9939 | 0.9878 | −0.61 | 163 |
| 2 | 0.9723 | 0.9634 | −0.89 | 253 |
| 3 | 0.9283 | 0.9146 | −1.37 | 438 |
| 4 | 0.9000 | 0.8692 | −3.08 | 130 |
| 5 | 0.7937 | 0.7619 | −3.17 | 63 |
| 6 | 0.8131 | 0.7323 | **−8.08** | 51 |

### 4.3 分推理步数：长链题是重灾区

| reasoning_step | 基线 | SFT | Δ (pt) | n |
|---|---|---|---|---|
| 1 | 0.9726 | 0.9932 | +2.05 | 146 |
| 2 | 0.9271 | 0.9167 | −1.04 | 384 |
| 3 | 0.8804 | 0.8674 | −1.30 | 460 |
| 4 | 0.8588 | 0.6941 | **−16.47** | 85 |
| 5 | 0.9130 | 0.7391 | **−17.39** | 23 |

单步题小幅变好、多步题大幅变差，是「推理链能力被替换成固定叙述模板」的典型形态：
模型学会了 NuminaMath 那种「Step 1 / Step 2 / 自我纠错」的行文节奏，但多步题上真正的中间状态跟踪能力反而下降。

### 4.4 根因定位（三步排除）

**第一步：排除判分/截断造成的假失分。**
SFT 模型一共答错 134 题，其中输出尾部找不到 `\boxed{}` 的只有 5 题（3.7%）——即被 `max_new_tokens=768` 截断、判分器退化成「取正文最后一个数字」的情形极少。即使这 5 题全部修好，准确率也只有 0.8798，补不上 3.1 pt 的缺口。**假失分不是主因。**

**第二步：训练侧指标指向过拟合。**
`trainer_log.jsonl` 里的验证集 loss：

| 步数 | epoch | eval_loss |
|---|---|---|
| 2500 | 2.0 | **0.4120**（最低） |
| 3000 | 2.4 | 0.4161 |
| 3500 | 2.8 | 0.4157 |
| 3750 | 3.0 | 0.4157 |

从 epoch 2.0 起验证集 loss 不再下降，而 train_loss 一路降到 **0.1803**：模型在训练集上继续「背熟」，泛化能力已经到顶并在缓慢倒退。第 3 个 epoch 是净负增益。

![train loss](docs/training_loss.png)
![eval loss](docs/training_eval_loss.png)

**第三步：checkpoint 选择实验没能完成（工程教训）。**
本想在 eval_loss 最低的 checkpoint-2500 上回评，但 `save_total_limit=3` 把早期 checkpoint 轮转删除了，最终目录只剩 {3000, 3500, 3750}，最优点已被自己的保存策略删掉。
**教训：做 checkpoint 选择类实验时，`save_total_limit` 必须调大（或设 `None`），否则最优点会在训练结束时被删除。**

### 4.5 下一轮实验配置（未执行，已明确）

| 项 | 本轮 | 下一轮 |
|---|---|---|
| epochs | 3 | **2**（2500 步，止于 eval_loss 最优点附近） |
| lr | 1e-4 | **5e-5** |
| save_total_limit | 3 | **None**（保留全部 checkpoint） |
| 评测 | 仅终态 | **逐 checkpoint 回评，画 checkpoint-准确率曲线** |

判断依据：CMath 掉分集中在长链题 + 验证集 loss 在 epoch 2.0 见底，说明问题不是「数据不够」而是「训太狠」。降低训练强度是成本最低的下一步。

---

## 5. 复盘：这个负结果值不值

值。原因不是自我安慰，而是**它排除了一个真实的错误假设**：

「用 2 万条公开 CoT 语料做 LoRA，就能让 4B 模型在中文数学上更强」——这个假设在基座已经 90.9% 的情况下不成立。这件事只有真跑过才知道，因为：

- 训练侧一切「正常」：loss 从 2.0 降到 0.18，验证集 loss 0.41，没有任何报警
- 如果只报 train_loss，这会被当成一次成功训练
- 评测一跑，才发现「loss 好看」和「能力提升」是两件事：loss 的下降主要来自学会复述推理模板，而不是答对率

顺带得到三条可直接复用的判据：

1. **先看基线天花板**。基线 > 90% 的基准上做 SFT，先算清楚剩余空间是否大于评测噪声（1098 题时约 ±0.9 pt）。
2. **验证集 loss 见底就停**。eval_loss 从 epoch 2.0 起不再下降，第 3 个 epoch 是净损耗；`save_total_limit` 要设 `None`，否则最优点会被删。
3. **看分桶不看总分**。同是 −3.1 pt，均匀掉分是「格式不兼容」，随难度放大是「能力退化」——后者说明该换优化目标（RL/DPO），而不是继续调 SFT 超参。

---

## 已知限制

1. **强基座上的 SFT 可能负向**：本轮实测 3 epoch 掉 3.10 pt（见 §4）。Qwen3-4B-Instruct-2507 已具备完整的推理能力，用风格更整齐但难度分布不同的公开 CoT 语料继续训练，本质是在「覆盖」而非「增强」原有能力。做 SFT 前应先问：基座在这项任务上还有多少提升空间？基线 90.9% 意味着天花板只剩 9 pt，噪声就有 ±0.9 pt（1098 题）。
2. **输出语言偏移**：训练语料是英文推理链，SFT 后模型在中文题上可能出现「中文提问 → 英文推理」的混输现象。评测指标（数值答案）不受影响，但对话体验会下降。推理脚本已在 system prompt 中要求中文作答，属缓解而非根治。根治需要中文 CoT 语料（Math23K / APE210K 一类），未纳入本次范围。
3. **只覆盖 SFT 阶段**：无 RL / 偏好优化。下一阶段可用同一批数据做 GRPO —— 数学题的答案对错本身就是可验证奖励，不需要训练奖励模型。**在负向 SFT 的结论下，RL 比继续调 SFT 超参更值得优先尝试**：RL 的优化目标直接是「答对」，不会像 SFT 那样把答案质量与叙述风格绑定。
4. **CMath 偏小学**：1–6 年级，对 4B 模型而言上限不高（基线已 90.9%），提升幅度天然受限。若要更陡的提升曲线，应换 MATH-500 或更难的集合——那里基座分数低，SFT/RL 的增益空间才大。
5. **单次判分**：不做多采样投票（self-consistency），报的是保守准确率。
6. **checkpoint 选择实验未完成**：最优点 checkpoint-2500 被 `save_total_limit=3` 轮转删除，无法回评，因此「epoch 2 是否优于 epoch 3」只有验证集 loss 的间接证据，没有 CMath 的直接证据。

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
│   ├── train.sh             # 训练入口（支持 RESUME=auto）
│   ├── run_eval.sh          # 基线与 SFT 对照
│   └── eval.py              # CMath 判分
├── docs/
│   ├── training_loss.png        # 训练 loss 曲线（云产图，下载归档）
│   └── training_eval_loss.png   # 验证集 loss 曲线
├── out/                         # adapter 落盘处（git 忽略，与云端目录结构对应）
├── results/
│   ├── base.csv             # 基线 1098 题逐题明细
│   └── sft.csv              # SFT 1098 题逐题明细（含输出尾部，可复查判分）
└── README.md
```

`results/*.csv` 的列：`grade / reasoning_step / question / golden / pred_raw / pred_num / correct / output_tail`。`output_tail` 保留每条输出最后 300 字符，用于人工抽查判分是否误杀。
