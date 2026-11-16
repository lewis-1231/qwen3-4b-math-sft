# -*- coding: utf-8 -*-
"""
eval.py — CMath 中文数学基准评测（答案准确率）

用法：
    # 基座（基线）
    python eval.py --model /mnt/workspace/models/qwen3-4b \
                   --data ../data/cmath/cmath_test.jsonl --out ../results/base.csv

    # 基座 + LoRA
    python eval.py --model /mnt/workspace/models/qwen3-4b \
                   --adapter /mnt/workspace/out/qwen3-4b-math-lora \
                   --data ../data/cmath/cmath_test.jsonl --out ../results/sft.csv

说明：
  * 指标是「答案准确率」，与输出语言无关——这是本次唯一客观、可对标公开榜单的口径
  * 答案抽取对三种结尾形式都兼容：\\boxed{X}、答案：X、正文最后一个数字
  * 每题只判一次，不做多种子投票，报的是保守值
"""
import argparse
import json
import os
import re
import sys
from fractions import Fraction

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "请解答下面的数学题。用中文作答，并在最后用 \\boxed{} 给出最终答案。"


def to_num(s):
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace("$", "").replace(",", "").replace("，", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("%", "")
    s = s.strip("。.")
    if not s:
        return None
    try:
        if "/" in s and not re.search(r"[a-zA-Z]", s):
            return float(Fraction(s))
        return float(s)
    except Exception:
        return None


def extract_answer(text):
    if not text:
        return ""
    for pat in (r"\\boxed\{([^{}]*)\}",
                r"答案\s*[:：]\s*([^\n。]+)",
                r"最终答案\s*[:：]\s*([^\n。]+)",
                r"[Ff]inal\s+answer\s*(?:is|:)\s*([^\n.]+)"):
        m = re.findall(pat, text)
        if m:
            return m[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else ""


def is_correct(pred, golden):
    p, g = to_num(pred), to_num(golden)
    if p is None or g is None:
        return str(pred).strip() == str(golden).strip(), p
    return abs(p - g) < 1e-4, p


def build_inputs(tokenizer, questions):
    texts = []
    for q in questions:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}]
        try:
            t = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            t = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 题（调试用）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} 题  ({os.path.basename(args.data)})")

    print(f"[model] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        device_map="auto", attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        print(f"[adapter] {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    preds = []
    texts = build_inputs(tok, [r["question"] for r in rows])
    for i in range(0, len(texts), args.batch):
        chunk = texts[i: i + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1536).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(rows[i: i + args.batch], gen):
            pred = extract_answer(g)
            ok, pnum = is_correct(pred, r["golden"])
            preds.append({
                "grade": r["grade"], "reasoning_step": r.get("reasoning_step"),
                "question": r["question"], "golden": r["golden"],
                "pred_raw": pred, "pred_num": pnum, "correct": bool(ok),
                "output_tail": g[-300:].replace("\n", " "),
            })
        done = min(i + args.batch, len(texts))
        if done % (args.batch * 10) == 0 or done == len(texts):
            acc = sum(x["correct"] for x in preds) / len(preds)
            print(f"  {done}/{len(texts)}  acc={acc:.4f}", flush=True)

    df = pd.DataFrame(preds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n=== 结果 ===")
    print(f"总体准确率: {df['correct'].mean():.4f}  ({int(df['correct'].sum())}/{len(df)})")
    print("\n分年级：")
    g = df.groupby("grade")["correct"].agg(["mean", "count"])
    for k, row in g.iterrows():
        print(f"  grade {k}: {row['mean']:.4f}  (n={int(row['count'])})")
    print("\n分推理步数：")
    s = df.groupby("reasoning_step")["correct"].agg(["mean", "count"])
    for k, row in s.iterrows():
        print(f"  step {k}: {row['mean']:.4f}  (n={int(row['count'])})")
    print(f"\n明细已保存: {args.out}")


if __name__ == "__main__":
    main()
