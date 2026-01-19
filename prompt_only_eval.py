# -*- coding: utf-8 -*-
"""
三分类评测脚本（模型输出只能是 1/2/3），但数据 score 是 1-5：
- 先把 score(1-5) 映射成 true(1-3)，再评测三分类。
- 严格解析模型输出：必须“完全等于” 1/2/3（允许空格），如 '1 or 2' 视为 invalid。
- invalid 默认当作错误（可用 --invalid_mode skip 选择跳过不计）。

运行示例：
python prompt_only_eval.py --model gpt2 --data "C:\...\scored_input2.json" --use_metrics --strict_output
python prompt_only_eval.py --model gpt2 --data "C:\...\scored_input2.json" --use_metrics --strict_output --invalid_mode skip
python prompt_only_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data .\scored_input2.json --use_metrics --load_in_4bit --trust_remote_code --strict_output
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ====== 解析相关 ======
STRICT_OUT_RE = re.compile(r"^\s*([123])\s*$")   # 严格：只能是单个 1/2/3
LOOSE_OUT_RE = re.compile(r"[123]")             # 宽松：找第一个 1/2/3
SCORE_1_5_RE = re.compile(r"([1-5])")           # score(1-5) 抽取


# ====== 数据读取（只认你的 keys：identifier/text/metrics/score） ======
REQUIRED_KEYS = ["identifier", "text", "metrics", "score"]


def load_json_list(path: str) -> List[Dict[str, Any]]:
    """读取 .json，顶层必须是 list；兼容 Windows BOM(utf-8-sig)."""
    p = Path(path)
    with p.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"输入文件顶层不是 list，而是 {type(obj)}")
    return obj


def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容两种结构：
    A) 直接：{identifier,text,metrics,score,...}
    B) 外层包一层：{data:{identifier,text,metrics,score,...}, ...外层字段...}
    规则：先取 data 内字段，再用外层字段覆盖（如果外层也有 score 等）。
    """
    if isinstance(rec, dict) and isinstance(rec.get("data"), dict):
        flat = dict(rec["data"])
        for k, v in rec.items():
            if k != "data":
                flat[k] = v
        return flat
    return rec


# ====== 标签处理 ======
def parse_model_label(text: str, strict: bool = True) -> int:
    """
    strict=True：必须完全等于 '1'/'2'/'3'（允许空格），否则 0
    strict=False：从文本中找第一个 1/2/3（更宽松）
    """
    s = (text or "")
    if strict:
        m = STRICT_OUT_RE.match(s)
        return int(m.group(1)) if m else 0
    m2 = LOOSE_OUT_RE.search(s)
    return int(m2.group(0)) if m2 else 0


def parse_score_1to5(score_value: Any) -> int:
    """从 score 字段抽取 1..5，抽不到返回 0。"""
    s = str(score_value).strip()
    m = SCORE_1_5_RE.search(s)
    return int(m.group(1)) if m else 0


def score_to_3class(score_1to5: int, low_max: int, mid_max: int) -> int:
    """
    把 1..5 映射到 1..3：
    - <= low_max  -> 1
    - <= mid_max  -> 2
    - >  mid_max  -> 3
    默认 low_max=2, mid_max=3 即：1-2->1, 3->2, 4-5->3
    """
    if score_1to5 <= 0:
        return 0
    if score_1to5 <= low_max:
        return 1
    if score_1to5 <= mid_max:
        return 2
    return 3


# ====== Prompt ======
def build_prompt(tokenizer, instruction: str, user_input: str) -> str:
    """
    强约束：只输出单个字符 1/2/3（不允许解释、不允许标点）
    """
    constraint = "Output EXACTLY one character: 1 or 2 or 3. No words, no punctuation, no explanation."

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a strict classifier."},
            {"role": "user", "content": f"{instruction}\n\n{user_input}\n\n{constraint}"},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return (
        f"{instruction}\n\n"
        f"{user_input}\n\n"
        f"{constraint}\n"
        "Label:"
    )


# ====== 推理 ======
@torch.inference_mode()
def predict_one(model, tokenizer, prompt: str, max_new_tokens: int, strict_output: bool) -> Tuple[int, str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # do_sample=False 时不要传 temperature，避免你看到的 warning
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    label = parse_model_label(gen_text, strict=strict_output)
    return label, gen_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="你的 JSON 文件（顶层 list；每条含 identifier/text/metrics/score）")
    parser.add_argument("--use_metrics", action="store_true", help="把 metrics 拼到输入里（不包含 score）")
    parser.add_argument("--max_new_tokens", type=int, default=2, help="建议 1-2，减少模型输出多余文字")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    # score(1-5) -> true(1-3) 的映射阈值
    parser.add_argument("--low_max", type=int, default=2)
    parser.add_argument("--mid_max", type=int, default=3)

    # 严格输出（推荐开）
    parser.add_argument("--strict_output", action="store_true", help="严格要求模型输出只能是单个 1/2/3")
    # invalid 怎么处理：error=当错；skip=跳过不计
    parser.add_argument("--invalid_mode", choices=["error", "skip"], default="error")

    args = parser.parse_args()

    if not (1 <= args.low_max < args.mid_max <= 5):
        raise ValueError("阈值必须满足：1 <= low_max < mid_max <= 5")

    instruction = "Given a Python test code snippet, classify it into 3 classes and output ONLY one label."

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # 兼容 gpt2

    quant_config = None
    if args.load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("你开启了 --load_in_4bit，但没有 bitsandbytes。")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading model: {args.model} (4bit={bool(quant_config)})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=quant_config,
    )
    model.eval()

    data = load_json_list(args.data)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    y_true: List[int] = []
    y_pred: List[int] = []
    invalid_outputs = 0

    for idx, rec in enumerate(data, 1):
        if not isinstance(rec, dict):
            raise ValueError(f"第 {idx} 条样本不是 dict，而是 {type(rec)}")

        rec = flatten_record(rec)

        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            raise KeyError(
                f"第 {idx} 条样本缺少 key: {missing}\n"
                f"该样本实际 keys={list(rec.keys())}"
            )

        identifier = rec["identifier"]
        text = rec["text"]
        metrics = rec["metrics"]
        score_raw = rec["score"]

        score_1to5 = parse_score_1to5(score_raw)
        if score_1to5 == 0:
            raise ValueError(f"第 {idx} 条样本 score={score_raw!r} 解析不到 1-5（identifier={identifier})")

        true_label = score_to_3class(score_1to5, low_max=args.low_max, mid_max=args.mid_max)
        if true_label == 0:
            raise ValueError(f"第 {idx} 条样本 score={score_raw!r} 无法映射到 3 类（identifier={identifier})")

        user_input = text
        if args.use_metrics:
            user_input = f"{text}\n\nMETRICS:\n{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}"

        prompt = build_prompt(tokenizer, instruction, user_input)
        pred_label, raw = predict_one(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            strict_output=args.strict_output
        )

        if pred_label == 0:
            invalid_outputs += 1
            if args.invalid_mode == "skip":
                # 跳过不计指标
                if idx <= 3:
                    print(f"\n[Sample {idx}] invalid_output raw_output={raw!r} (skipped)")
                continue
            # 当作错误：为了能算指标，给一个固定类别（1）
            pred_label = 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        if idx <= 3:
            print(f"\n[Sample {idx}]")
            print(f"identifier={identifier}")
            print(f"score(1-5)={score_1to5} -> true(1-3)={true_label} | pred={pred_label} raw_output={raw!r}")

    if not y_true:
        raise RuntimeError("没有可用于评测的样本（可能都被 skip 了或输入有问题）。")

    LABELS = [1, 2, 3]
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\n========== RESULTS ==========")
    print(f"N={len(y_true)}  invalid_model_outputs={invalid_outputs}")
    print(f"score->3class mapping: <= {args.low_max} =>1, <= {args.mid_max} =>2, > {args.mid_max} =>3")
    print(f"ACC={acc:.4f}")
    print(f"Macro-F1={mf1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred) labels=[1,2,3]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))


if __name__ == "__main__":
    main()
