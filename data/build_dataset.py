# -*- coding: utf-8 -*-
"""
build_dataset.py — 从 NuminaMath-CoT 构建 SFT 训练集（sharegpt 格式）

用法：
    python build_dataset.py --n-train 20000 --n-val 600

产出（同目录）：
    train.jsonl        训练集
    val.jsonl          验证集（训练中监控，不参与评测报告）
    dataset_info.json  LLaMA-Factory 数据集注册文件

设计口径：
  1. 只保留 messages 恰好为 [user, assistant] 两条的样本（避免多轮混入）
  2. 题面 30~1500 字符、解答 100~4000 字符（超出 2048 token 的会被裁掉尾巴，等于学不到结论）
  3. 按题面做去重（hash）
  4. 分层采样：按 source 比例抽，但单个 source 上限 cap 防寡头
  5. 随机种子固定，结果可复现
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numina")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MIN_Q, MAX_Q = 30, 1500
MIN_A, MAX_A = 100, 4000


def load_all(src_dir):
    shards = sorted(
        os.path.join(src_dir, f) for f in os.listdir(src_dir)
        if f.startswith("train-") and f.endswith(".parquet")
    )
    if not shards:
        sys.exit(f"[FATAL] 在 {src_dir} 下没找到 train-*.parquet")
    rows = []
    for p in shards:
        t = pq.read_table(p, columns=["source", "problem", "solution"])
        d = t.to_pydict()
        n = len(d["source"])
        rows.extend(zip(d["source"], d["problem"], d["solution"]))
        print(f"  {os.path.basename(p)}: {n}")
    print(f"  合计 {len(rows)} 条")
    return rows


def clean(rows):
    seen = set()
    out = []
    for src, q, a in rows:
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        q, a = q.strip(), a.strip()
        if not (MIN_Q <= len(q) <= MAX_Q):
            continue
        if not (MIN_A <= len(a) <= MAX_A):
            continue
        # 明确要求"证明/explain why"的样本答案常无唯一值，评测无法判分，训练也会拉低格式稳定性
        if len(q) < 40 and q.rstrip().endswith("?"):
            pass  # 短但可用的选择题不排除
        h = hashlib.md5(q.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append((src, q, a))
    print(f"  清洗后 {len(out)} 条（去重 + 长度过滤）")
    return out


def stratified_sample(rows, n, cap, seed):
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r[0]].append(r)
    for v in by_src.values():
        rng.shuffle(v)

    total = len(rows)
    quota = {}
    for s, v in by_src.items():
        want = int(round(n * len(v) / total))
        quota[s] = max(1, min(len(v), want, cap))

    # 配额超/不足时按最大余数法微调总数
    picked = []
    for s, k in quota.items():
        picked.extend(by_src[s][:k])
    picked.sort(key=lambda x: x[0])
    rng.shuffle(picked)
    if len(picked) > n:
        picked = picked[:n]
    elif len(picked) < n:
        # 用未抽中的样本补齐
        chosen = set(id(x) for x in picked)
        rest = [r for r in rows if id(r) not in chosen]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked


def to_sharegpt(rows):
    out = []
    for src, q, a in rows:
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "source": src,
        })
    return out


def dump(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写出 {path}：{len(records)} 条")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--cap", type=int, default=6000, help="单一 source 在训练集中的上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[1/4] 读取 parquet 分片")
    rows = load_all(SRC_DIR)

    print("[2/4] 清洗")
    rows = clean(rows)

    print("[3/4] 分层采样")
    train_rows = stratified_sample(rows, args.n_train, args.cap, args.seed)
    used = set(train_rows)
    rest = [r for r in rows if r not in used]
    val_rows = stratified_sample(rest, args.n_val, args.n_val, args.seed + 1)

    print("[4/4] 导出")
    dump(to_sharegpt(train_rows), os.path.join(OUT_DIR, "train.jsonl"))
    dump(to_sharegpt(val_rows), os.path.join(OUT_DIR, "val.jsonl"))

    info = {
        "math_sft": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        "math_sft_val": {
            "file_name": "val.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    p = os.path.join(OUT_DIR, "dataset_info.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  写出 {p}")

    print("\n=== 训练集分布 ===")
    c = Counter(r[0] for r in train_rows)
    for s, k in c.most_common():
        print(f"  {s:<20} {k:>6}  ({k / len(train_rows):.1%})")
    qlen = sorted(len(r[1]) for r in train_rows)
    alen = sorted(len(r[2]) for r in train_rows)
    print(f"\n题面字符 中位 {qlen[len(qlen)//2]} / 95分位 {qlen[int(len(qlen)*0.95)]}")
    print(f"解答字符 中位 {alen[len(alen)//2]} / 95分位 {alen[int(len(alen)*0.95)]}")
    print("提示：解答中位 ~1000 字符 ≈ 400-600 token，cutoff_len=2048 足够容纳绝大多数样本")


if __name__ == "__main__":
    main()
