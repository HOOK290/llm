# -*- coding: utf-8 -*-
import argparse
import json
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

r"""
usage:
python .\prompt_only_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data .\val_code.jsonl --load_in_4bit --trust_remote_code
python .\prompt_only_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data .\val_code_metrics.jsonl --load_in_4bit --trust_remote_code
"""

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


LABEL_RE = re.compile(r"[123]")

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def build_prompt(tokenizer, instruction: str, user_input: str) -> str:
    """
    Prefer chat template if available; otherwise fallback to plain prompt.
    We force the model to output ONLY 1/2/3.
    """
    # If tokenizer supports chat template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n\n{user_input}\n\nOutput ONLY one label: 1 or 2 or 3."},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Fallback (non-chat)
    return (
        f"{instruction}\n\n"
        f"{user_input}\n\n"
        "Output ONLY one label: 1 or 2 or 3.\n"
        "Label:"
    )

def parse_label(text: str) -> int:
    """
    Extract first occurrence of 1/2/3.
    Return 0 if not found (invalid).
    """
    m = LABEL_RE.search(text)
    if not m:
        return 0
    return int(m.group(0))

@torch.inference_mode()
def predict_one(model, tokenizer, prompt: str, max_new_tokens: int = 3) -> Tuple[int, str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # deterministic
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    label = parse_label(gen_text)
    return label, gen_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data", required=True, help="val_code.jsonl or val_code_metrics.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise evaluate first N samples")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization (recommended on 4060)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable if model requires it")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)

    quant_config = None
    if args.load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig not available. Please install bitsandbytes.")
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
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    model.eval()

    data = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    y_true, y_pred = [], []
    invalid = 0

    for i, ex in enumerate(data, 1):
        instruction = ex["instruction"]
        user_input = ex["input"]
        true_label = int(str(ex["output"]).strip())

        prompt = build_prompt(tokenizer, instruction, user_input)
        pred_label, raw = predict_one(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)

        if pred_label == 0:
            invalid += 1
            # treat invalid as wrong prediction; map to 1 (or keep 0 out of labels)
            # Here we map to 1 just to keep metrics computable; also count invalid separately.
            pred_label = 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        if i <= 3:
            print(f"\n[Sample {i}] true={true_label} pred={pred_label} raw_output={raw!r}")

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])

    print("\n========== RESULTS ==========")
    print(f"Model: {args.model}")
    print(f"Data : {args.data}")
    print(f"N    : {len(y_true)}  (invalid_outputs={invalid})")
    print(f"ACC  : {acc:.4f}")
    print(f"Macro-F1: {mf1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred) labels=[1,2,3]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[1, 2, 3], digits=4))

if __name__ == "__main__":
    main()
